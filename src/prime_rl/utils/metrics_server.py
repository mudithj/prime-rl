"""Prometheus metrics server for trainer observability.

Exposes training metrics at /metrics in Prometheus format.
Also exposes /health endpoint for Kubernetes liveness probes.

The metrics server runs in a background thread. Additionally, a separate
health-check process is spawned that serves /health independently of the
GIL — this ensures K8s liveness probes always get a response, even during
long forward/backward passes that block the Python thread.
"""

import multiprocessing
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING

from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest

if TYPE_CHECKING:
    from prime_rl.configs.shared import MetricsServerConfig


@dataclass
class RunStats:
    """Statistics for a single run/LoRA adapter."""

    run_id: str
    step: int
    total_tokens: int
    learning_rate: float
    ready: bool


def _health_process_main(host: str, port: int) -> None:
    """Minimal health server in a separate process. Immune to GIL."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok\n")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    HTTPServer((host, port), Handler).serve_forever()


class HealthServer:
    """Lightweight HTTP server exposing /health for Kubernetes liveness probes.

    Runs in a separate process so it always responds, even when the trainer
    holds the GIL during long forward/backward passes on large models.
    """

    def __init__(self, port: int, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self._process: multiprocessing.Process | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            logger.warning(f"{self.__class__.__name__} already started")
            return

        self._process = multiprocessing.Process(
            target=_health_process_main, args=(self.host, self.port), daemon=True
        )
        self._process.start()
        self._started = True
        logger.info(f"Health server started at http://{self.host}:{self.port}/health (pid={self._process.pid})")

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.kill()
            self._process = None
            self._started = False
            logger.info(f"{self.__class__.__name__} stopped")


class MetricsServer(HealthServer):
    """Prometheus metrics server with /metrics (thread) and /health (process).

    /metrics runs in a background thread — may be delayed during GIL-heavy work
    but Prometheus tolerates this (it retries on the next scrape interval).

    /health runs in a separate process via the parent HealthServer — always
    responsive for K8s liveness probes.

    Both share the same port number but serve on different ports:
    - metrics_port: /metrics + /health (thread, best-effort)
    - metrics_port + 1: /health only (process, always responsive)

    K8s probes should target metrics_port + 1 for guaranteed responsiveness.
    For backwards compatibility, /health on the main port also works when the
    GIL is available.
    """

    def __init__(self, config: "MetricsServerConfig"):
        # Health process runs on port + 1
        super().__init__(config.port + 1, config.host)
        self.config = config
        self._metrics_port = config.port

        self._registry = CollectorRegistry()
        self._step = Gauge("trainer_step", "Current training step", registry=self._registry)
        self._loss = Gauge("trainer_loss", "Current training loss", registry=self._registry)
        self._throughput = Gauge(
            "trainer_throughput_tokens_per_sec", "Training throughput in tokens/sec", registry=self._registry
        )
        self._last_step_ts = Gauge(
            "trainer_last_step_timestamp_seconds", "Unix timestamp of last step", registry=self._registry
        )
        self._grad_norm = Gauge("trainer_grad_norm", "Gradient norm", registry=self._registry)
        self._peak_mem = Gauge("trainer_peak_memory_gib", "Peak GPU memory in GiB", registry=self._registry)
        self._lr = Gauge("trainer_learning_rate", "Current learning rate", registry=self._registry)
        self._mfu = Gauge("trainer_mfu_percent", "Model FLOPS utilization %", registry=self._registry)
        self._entropy = Gauge("trainer_entropy", "Mean entropy", registry=self._registry)
        self._mismatch_kl = Gauge(
            "trainer_mismatch_kl", "KL divergence between trainer and inference model", registry=self._registry
        )
        self._kl_ent_ratio = Gauge("trainer_kl_ent_ratio", "Ratio of mismatch KL to entropy", registry=self._registry)
        self._zero_grad_ratio = Gauge(
            "trainer_zero_grad_ratio",
            "Fraction of tracked parameter elements with zero gradient",
            registry=self._registry,
        )
        # Aggregate run metrics
        self._runs_discovered = Gauge(
            "trainer_runs_discovered", "Number of run folders discovered", registry=self._registry
        )
        self._runs_active = Gauge("trainer_runs_active", "Number of runs with assigned slots", registry=self._registry)
        self._runs_ready = Gauge(
            "trainer_runs_ready", "Number of runs ready for gradient updates", registry=self._registry
        )
        self._runs_max = Gauge("trainer_runs_max", "Maximum run capacity", registry=self._registry)
        # Per-run metrics with labels
        self._run_step = Gauge("trainer_run_step", "Training step for run", ["run"], registry=self._registry)
        self._run_tokens = Gauge(
            "trainer_run_tokens", "Total tokens processed by run", ["run"], registry=self._registry
        )
        self._run_learning_rate = Gauge(
            "trainer_run_learning_rate", "Current learning rate for run", ["run"], registry=self._registry
        )
        self._run_ready = Gauge(
            "trainer_run_ready",
            "Whether run is ready for updates (1=ready, 0=not ready)",
            ["run"],
            registry=self._registry,
        )
        # Track known run labels for cleanup
        self._known_runs: set[str] = set()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        registry = self._registry

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    self._handle_metrics()
                elif self.path == "/health":
                    self._handle_health()
                else:
                    self.send_response(404)
                    self.end_headers()

            def _handle_metrics(self):
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                self.end_headers()
                self.wfile.write(generate_latest(registry))

            def _handle_health(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok\n")

            def log_message(self, format, *args):
                pass

        return Handler

    def start(self) -> None:
        if self._started:
            logger.warning("Metrics server already started")
            return

        # Start health process on port + 1 (GIL-free)
        super().start()

        # Start metrics thread on main port
        self._server = HTTPServer((self.host, self._metrics_port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"Metrics server started at http://{self.host}:{self._metrics_port}/metrics")
        logger.info(f"GIL-free health endpoint at http://{self.host}:{self.port}/health")

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            if self._thread is not None:
                self._thread.join(timeout=5.0)
            self._server = None
            self._thread = None
        super().stop()

    def update(
        self,
        step: int,
        loss: float,
        throughput: float,
        grad_norm: float,
        peak_memory_gib: float,
        learning_rate: float,
        mfu: float = 0.0,
        entropy: float = 0.0,
        mismatch_kl: float = 0.0,
        zero_grad_ratio: float = 0.0,
    ) -> None:
        """Update metrics after a training step."""
        self._step.set(step)
        self._loss.set(loss)
        self._throughput.set(throughput)
        self._grad_norm.set(grad_norm)
        self._peak_mem.set(peak_memory_gib)
        self._lr.set(learning_rate)
        self._mfu.set(mfu)
        self._entropy.set(entropy)
        self._mismatch_kl.set(mismatch_kl)
        self._zero_grad_ratio.set(zero_grad_ratio)
        if entropy > 0:
            self._kl_ent_ratio.set(mismatch_kl / entropy)
        self._last_step_ts.set(time.time())

    def update_runs(
        self,
        runs_discovered: int,
        runs_max: int,
        run_stats: list[RunStats],
    ) -> None:
        """Update run/LoRA metrics."""
        self._runs_discovered.set(runs_discovered)
        self._runs_active.set(len(run_stats))
        self._runs_ready.set(sum(1 for r in run_stats if r.ready))
        self._runs_max.set(runs_max)

        current_runs = {r.run_id for r in run_stats}

        removed_runs = self._known_runs - current_runs
        for run_id in removed_runs:
            self._run_step.remove(run_id)
            self._run_tokens.remove(run_id)
            self._run_learning_rate.remove(run_id)
            self._run_ready.remove(run_id)

        for run in run_stats:
            self._run_step.labels(run=run.run_id).set(run.step)
            self._run_tokens.labels(run=run.run_id).set(run.total_tokens)
            self._run_learning_rate.labels(run=run.run_id).set(run.learning_rate)
            self._run_ready.labels(run=run.run_id).set(1 if run.ready else 0)

        self._known_runs = current_runs
