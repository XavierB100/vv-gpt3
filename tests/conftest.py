import pytest

import web_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "uploads").mkdir()
    (tmp_path / "logs").mkdir()
    web_app.app.config.update(TESTING=True)
    return web_app.app.test_client()
