import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import httpx
import pytest

from prime_rl.utils.client import _is_retryable_lora_error, init_nccl_broadcast, load_lora_adapter


def test_is_retryable_lora_error_returns_true_for_404():
    response = MagicMock()
    response.status_code = 404
    error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_true_for_500():
    response = MagicMock()
    response.status_code = 500
    error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_false_for_400():
    response = MagicMock()
    response.status_code = 400
    error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is False


def test_is_retryable_lora_error_returns_false_for_non_http_error():
    assert _is_retryable_lora_error(ValueError("some error")) is False


def test_load_lora_adapter_succeeds_on_first_attempt():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
    )


def test_load_lora_adapter_retries_on_404_then_succeeds():
    mock_client = AsyncMock()

    error_response = MagicMock()
    error_response.status_code = 404
    success_response = MagicMock()
    success_response.raise_for_status = MagicMock()

    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.HTTPStatusError("Not found", request=MagicMock(), response=error_response)
        return success_response

    mock_client.post = mock_post

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    assert call_count == 2


def test_load_lora_adapter_raises_non_retryable_error_immediately():
    mock_client = AsyncMock()

    error_response = MagicMock()
    error_response.status_code = 400
    mock_client.post.side_effect = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=error_response)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    assert exc_info.value.response.status_code == 400
    assert mock_client.post.call_count == 1


def test_init_nccl_broadcast_omits_gpus_per_server_from_request_payload():
    mock_clients = [AsyncMock(), AsyncMock()]
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    for client in mock_clients:
        client.post.return_value = mock_response

    asyncio.run(
        init_nccl_broadcast(
            mock_clients,
            host="localhost",
            port=9000,
            timeout=30,
            inference_world_size=4,
            quantize_in_weight_transfer=True,
            delta_compression=True,
        )
    )

    expected_calls = [
        call(
            "/init_broadcaster",
            json={
                "host": "localhost",
                "port": 9000,
                "rank_offset": 0,
                "inference_world_size": 4,
                "timeout": 30,
                "quantize_in_weight_transfer": True,
                "delta_compression": True,
            },
        ),
        call(
            "/init_broadcaster",
            json={
                "host": "localhost",
                "port": 9000,
                "rank_offset": 2,
                "inference_world_size": 4,
                "timeout": 30,
                "quantize_in_weight_transfer": True,
                "delta_compression": True,
            },
        ),
    ]

    actual_calls = [client.post.call_args for client in mock_clients]
    assert actual_calls == expected_calls
    for _, kwargs in actual_calls:
        assert "gpus_per_server" not in kwargs["json"]
