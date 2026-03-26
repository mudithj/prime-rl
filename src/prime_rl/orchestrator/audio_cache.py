import base64
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import verifiers as vf

_PARALLEL_DECODE_THRESHOLD = 4


def _collect_audio_keys_from_messages(messages: list) -> list[str]:
    """Extract audio base64 keys from OpenAI-style chat messages."""
    keys = []
    if not messages or not isinstance(messages, list):
        return keys
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "audio_url":
                    url = item.get("audio_url", {}).get("url", "")
                    if url.startswith("data:audio"):
                        keys.append(url.split(",", 1)[1])
    return keys


def _extract_audio_from_examples(
    examples: list[tuple[int, vf.RolloutOutput]],
) -> tuple[list[str], dict[int, list[list[int]]]]:
    """Extract unique audio base64 keys and per-step indices from rollout trajectories."""
    unique_keys: list[str] = []
    key_to_index: dict[str, int] = {}
    step_audio_indices_per_example: dict[int, list[list[int]]] = {}

    for eid, output in examples:
        trajectory = output.get("trajectory", [])
        if not trajectory:
            step_audio_indices_per_example[eid] = []
            continue

        step_audio_indices = []
        for step in trajectory:
            prompt = step.get("prompt")
            audio_keys = _collect_audio_keys_from_messages(prompt)
            indices = []
            for key in audio_keys:
                if key not in key_to_index:
                    key_to_index[key] = len(unique_keys)
                    unique_keys.append(key)
                indices.append(key_to_index[key])
            step_audio_indices.append(indices)

        step_audio_indices_per_example[eid] = step_audio_indices

    return unique_keys, step_audio_indices_per_example


def _decode_audio_b64_to_waveform(b64_data: str) -> tuple[np.ndarray, int]:
    """Decode base64 audio to a mono float32 waveform at 16kHz."""
    from pydub import AudioSegment

    audio_bytes = base64.b64decode(b64_data)
    seg = AudioSegment.from_file(BytesIO(audio_bytes))
    seg = seg.set_frame_rate(16000).set_channels(1)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    samples /= 2 ** (seg.sample_width * 8 - 1)
    return samples, 16000


class _AudioStore:
    """Holds per-unique-audio feature data, assembled lazily on demand."""

    def __init__(
        self,
        audio_bytes: list[bytes],
        audio_lengths: list[int],
        feature_size: int,
        time_dim: int,
    ):
        self.audio_bytes = audio_bytes
        self.audio_lengths = audio_lengths
        self.feature_size = feature_size
        self.time_dim = time_dim
        self._cache: dict[tuple[int, ...], tuple[bytes, list[int], list[int]]] = {}

    def assemble(self, indices: list[int]) -> tuple[bytes, list[int], list[int]]:
        cache_key = tuple(indices)
        if cache_key in self._cache:
            return self._cache[cache_key]
        feature_bytes = b"".join(self.audio_bytes[i] for i in indices)
        shape = [len(indices), self.feature_size, self.time_dim]
        lengths = [self.audio_lengths[i] for i in indices]
        result = (feature_bytes, shape, lengths)
        self._cache[cache_key] = result
        return result


def _preprocess_audio_batched(
    audio_keys: list[str],
    step_audio_indices_per_example: dict[int, list[list[int]]],
    processor,
) -> tuple["_AudioStore | None", dict[int, list[list[int]]]]:
    """Decode and preprocess all unique audio clips, returning an _AudioStore."""
    if not audio_keys or processor is None or not hasattr(processor, "feature_extractor"):
        return None, step_audio_indices_per_example

    if len(audio_keys) > _PARALLEL_DECODE_THRESHOLD:
        with ThreadPoolExecutor(max_workers=min(len(audio_keys), 8)) as pool:
            waveforms = list(pool.map(_decode_audio_b64_to_waveform, audio_keys))
    else:
        waveforms = [_decode_audio_b64_to_waveform(k) for k in audio_keys]

    feature_extractor = processor.feature_extractor
    audio_bytes_list: list[bytes] = []
    audio_lengths_list: list[int] = []
    time_dim: int | None = None

    for samples, sr in waveforms:
        out = feature_extractor(samples, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
        feats = out["input_features"]   # [1, feature_size, T]
        mask = out["attention_mask"]    # [1, T]
        length = int(mask.sum().item())
        if time_dim is None:
            time_dim = feats.shape[2]
        audio_bytes_list.append(feats.numpy().tobytes())
        audio_lengths_list.append(length)

    store = _AudioStore(
        audio_bytes=audio_bytes_list,
        audio_lengths=audio_lengths_list,
        feature_size=feature_extractor.feature_size,
        time_dim=time_dim or 3000,
    )
    return store, step_audio_indices_per_example


class AudioCache:
    """Result of building audio feature cache with per-step data."""

    def __init__(
        self,
        store: "_AudioStore | None",
        step_indices: dict[int, list[list[int]]],
        num_unique_examples: int,
        num_unique_audios: int,
        extract_time: float,
        preprocess_time: float,
    ):
        self._store = store
        self._step_indices = step_indices
        self.num_unique_examples = num_unique_examples
        self.num_unique_audios = num_unique_audios
        self.extract_time = extract_time
        self.preprocess_time = preprocess_time

    def get_for_step(
        self, cache_key: int, step_idx: int
    ) -> tuple[bytes | None, list[int] | None, list[int] | None]:
        if self._store is None:
            return None, None, None
        steps = self._step_indices.get(cache_key, [])
        if not steps or step_idx >= len(steps) or not steps[step_idx]:
            return None, None, None
        return self._store.assemble(steps[step_idx])


def build_audio_cache(rollouts: list[vf.RolloutOutput], processor) -> AudioCache:
    """Build audio feature cache by extracting and preprocessing audio from rollout prompts."""
    examples = [(idx, rollout) for idx, rollout in enumerate(rollouts)]
    unique_example_ids = {rollout["example_id"] for rollout in rollouts}

    extract_start = time.perf_counter()
    audio_keys, audio_indices = _extract_audio_from_examples(examples)
    num_unique_audios = len(audio_keys)
    extract_time = time.perf_counter() - extract_start

    preprocess_start = time.perf_counter()
    store, step_indices = _preprocess_audio_batched(audio_keys, audio_indices, processor)
    preprocess_time = time.perf_counter() - preprocess_start

    return AudioCache(
        store=store,
        step_indices=step_indices,
        num_unique_examples=len(unique_example_ids),
        num_unique_audios=num_unique_audios,
        extract_time=extract_time,
        preprocess_time=preprocess_time,
    )
