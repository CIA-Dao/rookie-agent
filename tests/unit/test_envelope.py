import pytest
from pydantic import ValidationError

from my_agent.core.bus.envelope import (
    PARSE_ERROR,
    JsonRpcRequest,
    JsonRpcSuccess,
    make_error,
)


def test_request_roundtrip() -> None:
    """序列化再反序列化，字段不丢"""
    req = JsonRpcRequest(id="1", method="core.ping", params={"client": "test"})
    req2 = JsonRpcRequest.model_validate_json(req.model_dump_json())
    assert req2.id == "1"
    assert req2.method == "core.ping"
    assert req2.params == {"client": "test"}


def test_request_default_params() -> None:
    """不传 params 时默认是空字典"""
    req = JsonRpcRequest(id="1", method="x")
    assert req.params == {}


def test_request_missing_id_raises() -> None:
    """缺 id 字段应该报错"""
    with pytest.raises(ValidationError):
        JsonRpcRequest.model_validate({"jsonrpc": "2.0", "method": "x"})


def test_success_roundtrip() -> None:
    """成功响应序列化/反序列化"""
    resp = JsonRpcSuccess(id="1", result={"key": "value"})
    resp2 = JsonRpcSuccess.model_validate_json(resp.model_dump_json())
    assert resp2.id == "1"
    assert resp2.result == {"key": "value"}


def test_make_error_returns_correct_code() -> None:
    """make_error 造的 JsonRpcError 包含正确的错误码"""
    err = make_error("1", PARSE_ERROR, "Parse error")
    assert err.error.code == PARSE_ERROR
    assert err.id == "1"
