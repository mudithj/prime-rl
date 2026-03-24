import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


class _DummyAfterEvent:
    def current_stream_wait(self) -> None:
        return None


class _DummyBuffer:
    def __init__(self) -> None:
        self.combine_calls: list[dict[str, object]] = []

    def combine(self, **kwargs):
        self.combine_calls.append(kwargs)
        return kwargs["x"].clone(), None, _DummyAfterEvent()


@pytest.fixture(scope="module")
def deepep_module():
    module_path = Path(__file__).resolve().parents[3] / "src/prime_rl/trainer/distributed/deepep.py"

    deep_ep = ModuleType("deep_ep")
    deep_ep_utils = ModuleType("deep_ep.utils")

    class _FakeBuffer:
        @staticmethod
        def set_num_sms(num_sms: int) -> None:
            return None

    class _FakeEventHandle:
        pass

    class _FakeEventOverlap:
        def __init__(self, handle=None) -> None:
            self.handle = handle

        def current_stream_wait(self) -> None:
            return None

    deep_ep.Buffer = _FakeBuffer
    deep_ep_utils.EventHandle = _FakeEventHandle
    deep_ep_utils.EventOverlap = _FakeEventOverlap

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "deep_ep", deep_ep)
    monkeypatch.setitem(sys.modules, "deep_ep.utils", deep_ep_utils)

    class _FakeLibrary:
        def __init__(self, *_args, **_kwargs) -> None:
            return None

        def define(self, *_args, **_kwargs) -> None:
            return None

    monkeypatch.setattr(torch.library, "Library", _FakeLibrary)
    monkeypatch.setattr(torch.library, "impl", lambda *_args, **_kwargs: (lambda func: func))
    monkeypatch.setattr(torch.library, "register_autograd", lambda *_args, **_kwargs: None)

    spec = importlib.util.spec_from_file_location("test_prime_rl_trainer_distributed_deepep", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    try:
        yield module
    finally:
        monkeypatch.undo()


@pytest.fixture
def deepep_runtime(deepep_module):
    buffer = _DummyBuffer()
    deepep_module._buffer = buffer
    deepep_module._handle_cache.clear()
    deepep_module._pending_combine_event = None
    try:
        yield deepep_module, buffer
    finally:
        deepep_module._buffer = None
        deepep_module._handle_cache.clear()
        deepep_module._pending_combine_event = None


def test_combine_op_pops_handle_under_no_grad(deepep_runtime) -> None:
    module, buffer = deepep_runtime
    handle = object()
    handle_id = torch.tensor([1], dtype=torch.int64)
    module._handle_cache[handle_id.item()] = handle

    x = torch.randn(2, 3, requires_grad=True)
    with torch.no_grad():
        combined = module._combine_op_impl(x, handle_id)

    assert torch.equal(combined, x)
    assert buffer.combine_calls[0]["handle"] is handle
    assert handle_id.item() not in module._handle_cache


def test_combine_op_pops_handle_when_input_does_not_require_grad(deepep_runtime) -> None:
    module, buffer = deepep_runtime
    handle = object()
    handle_id = torch.tensor([2], dtype=torch.int64)
    module._handle_cache[handle_id.item()] = handle

    x = torch.randn(2, 3)
    combined = module._combine_op_impl(x, handle_id)

    assert torch.equal(combined, x)
    assert buffer.combine_calls[0]["handle"] is handle
    assert handle_id.item() not in module._handle_cache


def test_combine_op_leaves_handle_for_setup_context_when_autograd_tracks_it(deepep_runtime) -> None:
    module, _ = deepep_runtime
    handle = object()
    handle_id = torch.tensor([3], dtype=torch.int64)
    module._handle_cache[handle_id.item()] = handle

    x = torch.randn(2, 3, requires_grad=True)
    combined = module._combine_op_impl(x, handle_id)

    assert module._handle_cache[handle_id.item()] is handle

    ctx = SimpleNamespace()
    module._combine_setup_context(ctx, (x, handle_id), combined)

    assert ctx.saved_handle is handle
    assert handle_id.item() not in module._handle_cache
