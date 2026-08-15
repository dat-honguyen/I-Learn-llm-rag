import os
from dataclasses import dataclass


@dataclass
class Settings:
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://ollama:11434")
    chat_model: str = os.environ.get("CHAT_MODEL", "llama3.2:3b-instruct-q4_K_M")
    embed_model: str = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    app_password: str = os.environ.get("APP_PASSWORD", "")
    cors_origin: str = os.environ.get("CORS_ORIGIN", "*")
    db_path: str = os.environ.get("DB_PATH", "data/vectors.db")
    docs_dir: str = os.environ.get("DOCS_DIR", "docs_corpus")
    top_k: int = int(os.environ.get("TOP_K", "4"))
    max_question_len: int = int(os.environ.get("MAX_QUESTION_LEN", "500"))
    max_output_tokens: int = int(os.environ.get("MAX_OUTPUT_TOKENS", "512"))


settings = Settings()
