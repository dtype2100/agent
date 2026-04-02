"""선택적 API 키 검증 단위 테스트."""

from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from app.core.security import require_api_key_if_configured


def test_api_key_skipped_when_empty():
    req = Mock()
    req.headers = {}
    settings = Mock()
    settings.api_key = ""
    require_api_key_if_configured(req, settings)  # no raise


def test_api_key_rejects_missing_header():
    req = Mock()
    req.headers = Mock()
    req.headers.get = Mock(return_value=None)
    settings = Mock()
    settings.api_key = "expected-secret"
    with pytest.raises(HTTPException) as exc:
        require_api_key_if_configured(req, settings)
    assert exc.value.status_code == 401


def test_api_key_accepts_valid_header():
    req = Mock()
    req.headers = Mock()
    req.headers.get = Mock(return_value="expected-secret")
    settings = Mock()
    settings.api_key = "expected-secret"
    require_api_key_if_configured(req, settings)
