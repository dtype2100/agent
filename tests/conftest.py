import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings


@pytest.fixture(autouse=True)
def mock_mode_env(monkeypatch: pytest.MonkeyPatch):
    """API 키 없이 앱 스모크 테스트를 돌리기 위해 Mock 모드를 켠다."""
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("CORS_ORIGINS", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(mock_mode_env):
    """앱 import는 환경 설정 이후에만 수행한다."""
    from app.main import app

    with TestClient(app) as tc:
        yield tc
