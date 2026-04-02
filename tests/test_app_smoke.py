"""FastAPI 앱 스모크 테스트 (MOCK_MODE)."""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["mock_mode"] is True


def test_rag_query(client):
    r = client.post("/rag/query", json={"query": "테스트 질문"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert body["query"] == "테스트 질문"


def test_agent_chat(client):
    r = client.post("/agent/chat", json={"message": "안녕"})
    assert r.status_code == 200
    body = r.json()
    assert "session_id" in body
    assert "answer" in body


def test_documents_index_texts(client):
    r = client.post(
        "/documents/texts",
        json={"texts": ["스모크 테스트용 문장입니다."]},
    )
    assert r.status_code == 200
    assert r.json()["indexed"] >= 1
