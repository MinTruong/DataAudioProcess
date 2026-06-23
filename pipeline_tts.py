"""
TTS Dataset Pipeline — YouTube audio books to TTS training datasets.

Usage:
  python pipeline_tts.py <video_url>
  python pipeline_tts.py <channel_url> --max-videos 3
  python pipeline_tts.py <video_url> --config configs/default.yaml
"""

import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tts_pipeline.cli import main

if __name__ == "__main__":
    main()
