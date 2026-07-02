from pathlib import Path
from typing import Literal
import subprocess
import sys

import yaml
from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    # Paths
    raw_dir: Path = Field(default=Path("raw"), description="Directory for raw downloads")
    output_dir: Path = Field(default=Path("dataset"), description="Directory for final dataset")

    # Audio parameters
    sample_rate: int = Field(default=22050, ge=8000, le=48000)
    min_segment_dur: float = Field(default=5.0, ge=2.0, le=20.0)
    max_segment_dur: float = Field(default=20.0, ge=5.0, le=60.0)

    # Text filtering
    min_text_len: int = Field(default=10, ge=1, le=100)
    max_text_len: int = Field(default=500, ge=50, le=2000)

    # VTT parsing
    merge_gap: float = Field(default=0.5, ge=0.1, le=2.0, description="Max gap (s) to merge adjacent cues")
    fragment_threshold: float = Field(default=0.3, ge=0.01, le=1.0, description="Cues shorter than this are fragments")

    # Train/test split
    train_split: float = Field(default=0.9, ge=0.5, le=1.0)

    # YouTube
    youtube_caption_langs: list[str] = Field(default=["vi"])
    max_videos_per_channel: int = Field(default=5, ge=1, le=100)

    # System
    ffmpeg_path: str = Field(default="ffmpeg")
    device: Literal["cpu", "cuda"] = "cpu"

    @field_validator("ffmpeg_path")
    @classmethod
    def ffmpeg_must_exist(cls, v: str) -> str:
        try:
            subprocess.run([v, "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise ValueError(f"ffmpeg not found at '{v}'. Install ffmpeg and ensure it's on PATH.")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from a YAML file, merging with defaults."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data) if data else cls()

    @classmethod
    def from_cli_overrides(cls, **overrides) -> "Settings":
        """Create settings with specific overrides from CLI args."""
        return cls(**{k: v for k, v in overrides.items() if v is not None})


# Windows encoding fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
