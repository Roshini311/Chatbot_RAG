import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Research & Knowledge Assistant"
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    VECTOR_DB_DIR: str = Field(default="./data/vector_db", env="VECTOR_DB_DIR")
    RAW_DOCUMENTS_DIR: str = Field(default="./data/raw_documents", env="RAW_DOCUMENTS_DIR")
    DATABASE_URL: str = Field(default="sqlite:///./data/metadata.db", env="DATABASE_URL")
    MODEL_PATH: str = Field(default="./models/tf_classifier.h5", env="MODEL_PATH")
    TOKENIZER_PATH: str = Field(default="./models/tokenizer.pickle", env="TOKENIZER_PATH")
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
