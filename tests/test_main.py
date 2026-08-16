from pathlib import Path

from fastapi.testclient import TestClient

from app import main, ollama_client, store


async def fake_embed(text: str) -> list[float]:
    return [0.1] * 8


async def fake_chat_stream(prompt: str, max_tokens: int):
    for token in ["Answer", " goes", " here"]:
        yield token


def make_client(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "app_password", "secret")
    monkeypatch.setattr(main.settings, "db_path", str(tmp_path / "vectors.db"))
    monkeypatch.setattr(main.settings, "docs_dir", str(tmp_path / "docs_corpus"))
    (tmp_path / "docs_corpus").mkdir()
    (tmp_path / "docs_corpus" / "note.md").write_text("cats are great pets", encoding="utf-8")

    monkeypatch.setattr(ollama_client, "embed", fake_embed)
    monkeypatch.setattr(ollama_client, "chat_stream", fake_chat_stream)
    monkeypatch.setattr(store, "DEFAULT_DIM", 8)

    return TestClient(main.app)


def test_health_returns_ok(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_chat_rejects_wrong_password(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        response = client.post("/chat", json={"question": "hi", "password": "wrong"})
        assert response.status_code == 401


def test_chat_streams_answer_with_correct_password(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        response = client.post("/chat", json={"question": "hi", "password": "secret"})
        assert response.status_code == 200
        assert response.text == "Answer goes here"


def test_chat_rejects_question_over_max_length(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        monkeypatch.setattr(main.settings, "max_question_len", 5)
        response = client.post("/chat", json={"question": "way too long", "password": "secret"})
        assert response.status_code == 400


def test_chat_remembers_recent_turns_in_same_session(tmp_path, monkeypatch):
    seen_prompts = []

    async def recording_chat_stream(prompt: str, max_tokens: int):
        seen_prompts.append(prompt)
        for token in ["Answer", " goes", " here"]:
            yield token

    with make_client(tmp_path, monkeypatch) as client:
        monkeypatch.setattr(main.ollama_client, "chat_stream", recording_chat_stream)

        client.post(
            "/chat",
            json={"question": "what pets are great?", "password": "secret", "session_id": "abc"},
        )
        client.post(
            "/chat",
            json={"question": "tell me more", "password": "secret", "session_id": "abc"},
        )

    assert "Recent conversation with this visitor" not in seen_prompts[0]
    assert "what pets are great?" in seen_prompts[1]
    assert "Answer goes here" in seen_prompts[1]


def test_chat_without_session_id_has_no_recap(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        client.post("/chat", json={"question": "hi", "password": "secret"})
        response = client.post("/chat", json={"question": "hi again", "password": "secret"})
        assert response.status_code == 200
