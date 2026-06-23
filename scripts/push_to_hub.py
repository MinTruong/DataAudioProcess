"""
Upload a TTS dataset to HuggingFace Hub.

Usage:
  # Truyền trực tiếp CLI arguments
  python scripts/push_to_hub.py --parquet dataset/train.parquet --repo user/my-tts-dataset

  # Dùng config YAML
  python scripts/push_to_hub.py --config configs/upload.yaml

Config YAML mẫu (configs/upload.yaml):
  repo: user/my-tts-dataset
  dir: dataset
  token: hf_your_token_here        # optional, có thể dùng huggingface-cli login
  language: vi
  private: false
  commit_message: 'Add TTS dataset from audio books'
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
from huggingface_hub import HfApi, upload_folder


def upload_parquet_to_hub(
    parquet_path: str,
    repo_id: str,
    token: str | None = None,
    commit_message: str = "Add TTS dataset",
) -> None:
    """Upload a single Parquet file to HF Hub as a dataset."""
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=parquet_path,
        path_in_repo=Path(parquet_path).name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )


def upload_dataset_dir_to_hub(
    dataset_dir: str,
    repo_id: str,
    token: str | None = None,
) -> None:
    """Upload an entire dataset directory (parquet + audio/) to HF Hub."""
    upload_folder(
        folder_path=dataset_dir,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )


def create_dataset_card(repo_id: str, num_samples: int, language: str = "vi") -> str:
    """Generate a dataset card README.md for the HF Hub."""
    return f"""---
language: {language}
license: cc-by-nc-4.0
task:
  - text-to-speech
---

# TTS Dataset — {repo_id}

## Description
Vietnamese TTS dataset generated from YouTube audio books.

## Statistics
- Samples: {num_samples}
- Format: Parquet + WAV (mono, 22050Hz, 16-bit)

## Columns
| Column | Type | Description |
|--------|------|-------------|
| `audio` | struct | `{{path, bytes}}` — embedded audio compatible with HF Audio feature |
| `transcription` | string | Vietnamese text transcription |
| `file_name` | string | WAV filename (`audio/xxx.wav`) |

## Usage
```python
from datasets import load_dataset
dataset = load_dataset("{repo_id}")
```
"""


def load_config(config_path: str) -> dict:
    """Load upload config from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg else {}


def main():
    parser = argparse.ArgumentParser(description="Upload TTS dataset to HuggingFace Hub")
    parser.add_argument("--parquet", help="Path to Parquet file")
    parser.add_argument("--dir", help="Path to dataset directory (parquet + audio/)")
    parser.add_argument("--repo", help="HF repo ID (user/repo-name)")
    parser.add_argument("--token", help="HF token (default: use huggingface-cli login)")
    parser.add_argument("--private", action="store_true", help="Create private dataset repo")
    parser.add_argument("--config", help="Path to YAML config file (ghi đè = --parquet, --dir, --repo, --token, --private)")

    args = parser.parse_args()

    # Load config từ YAML trước
    cfg = {}
    if args.config:
        cfg = load_config(args.config)
        print(f"[Config] Loaded from {args.config}")

    # CLI args ghi đè config file
    parquet = args.parquet or cfg.get("parquet")
    dir_ = args.dir or cfg.get("dir")
    repo = args.repo or cfg.get("repo")
    token = args.token or cfg.get("token")
    private = args.private or cfg.get("private", False)
    language = cfg.get("language", "vi")
    commit_message = cfg.get("commit_message", "Add TTS dataset")

    if not repo:
        print("[Error] --repo is required (set in CLI or config YAML)")
        parser.print_help()
        sys.exit(1)

    api = HfApi(token=token)

    if parquet:
        upload_parquet_to_hub(parquet, repo, token, commit_message)
        df = pd.read_parquet(parquet)
        card = create_dataset_card(repo, len(df), language)
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
        print(f"[OK] Uploaded {parquet} to {repo}")
    elif dir_:
        upload_dataset_dir_to_hub(dir_, repo, token)
        parquet_files = list(Path(dir_).glob("*.parquet"))
        if parquet_files:
            df = pd.read_parquet(parquet_files[0])
            card = create_dataset_card(repo, len(df), language)
            api.upload_file(
                path_or_fileobj=card.encode(),
                path_in_repo="README.md",
                repo_id=repo,
                repo_type="dataset",
                commit_message="Add dataset card",
            )
        print(f"[OK] Uploaded {dir_} to {repo}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
