from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin123"
    minio_bucket: str = "documents"
    minio_secure: bool = False

    # Groq
    groq_api_key: str = ""
    groq_chat_model: str = "llama-3.3-70b-versatile"
    groq_fast_model: str = "llama-3.1-8b-instant"
    groq_tts_model: str = "canopylabs/orpheus-v1-english"
    groq_tts_voice: str = "autumn"

    # Data dirs
    data_dir: str = "/app/data"
    chroma_dir: str = "/app/data/chroma"
    kuzu_dir: str = "/app/data/kuzu"

    # Embedding
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure dirs exist
for d in [settings.data_dir, settings.chroma_dir, settings.kuzu_dir]:
    Path(d).mkdir(parents=True, exist_ok=True)