"""Application settings, read from the environment / repo-root .env. Secrets are SecretStr and never logged."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore", case_sensitive=False)

    database_url: str = Field(repr=False)  # contains the password: never shown in reprs, logs or test output
    test_database_url: str | None = Field(default=None, repr=False)
    storage_root: Path = Path("data")

    featherless_api_key: SecretStr | None = None
    featherless_base_url: str = "https://api.featherless.ai/v1"
    featherless_text_model: str = "Qwen/Qwen3-14B"
    featherless_vision_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    # Qwen3 hybrid models otherwise spend their token budget on hidden reasoning and often return empty or corrupt JSON
    # (measured 2026-10-31: valid JSON 4/10 with thinking, 10/10 without; research/phase6/claim_model_comparison.json)
    featherless_disable_thinking: bool = True

    sample_fps: float = 5.0      # frames analysed per second of video (CPU budget; see HARDWARE_AND_COMPUTE_PLAN)
    max_frames: int = 3000       # cap on sampled frames per video
    thumb_width: int = 320

    # detection (Phase 4); weights are resolved only through models.lock.json
    models_dir: Path | None = None          # default: <storage_root>/models
    detector_model: str = "yolo26s"
    detector_imgsz: int = 640
    detector_conf: float = 0.15             # storage threshold (keep weak boxes for tracking); the API filters for display
    accident_model: str | None = "accident_yolov8s"   # set to empty to disable the accident cue
    accident_conf: float = 0.25
    # which model supplies the accident cue: "detr" (hilmantm/detr-traffic-accident-detection, pinned in models.lock.json), "yolo" (accident_model above) or "" (none)
    accident_backend: str = "detr"
    accident_detr_conf: float = 0.48
    detector_device: str = "auto"           # auto -> mps if available, else cpu

    max_video_mb: int = 500
    max_other_mb: int = 20

    @property
    def storage_path(self) -> Path:
        return self.storage_root if self.storage_root.is_absolute() else (REPO_ROOT / self.storage_root).resolve()

    @property
    def models_path(self) -> Path:
        return self.models_dir if self.models_dir else self.storage_path / "models"

    @property
    def featherless_configured(self) -> bool:
        return bool(self.featherless_api_key and self.featherless_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
