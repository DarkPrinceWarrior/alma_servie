from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "alma_servie back"
    app_version: str = "0.1.0"
    environment: str = "local"
    debug: bool = True
    api_v1_prefix: str = "/api"

    data_root: Path = Path("/data")
    research_root: Path = Path("/workspace")
    uploads_root: Path = Path("/uploads")

    detection_mock: bool = True
    detection_mock_duration_seconds: float = 2.0

    postgres_user: str = "app"
    postgres_password: str = "app"
    postgres_db: str = "alma_servie"
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432

    database_url: str = "postgresql+asyncpg://app:app@127.0.0.1:5432/alma_servie"

    secret_key: str = "CHANGE_ME"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30

    admin_email: str | None = None
    admin_password: str | None = None
    admin_reset_password: bool = False

    refresh_cookie_secure: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
