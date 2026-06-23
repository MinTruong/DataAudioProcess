"""YouTube video and caption download using yt-dlp."""

import subprocess
from pathlib import Path

import yt_dlp

from tts_pipeline.config import Settings


def download_video_and_subs(
    video_url: str,
    output_dir: str | Path,
    settings: Settings | None = None,
) -> dict:
    """Download audio (WAV) and auto-caption from YouTube.

    Returns:
        dict with keys: video_id, title, audio_path, vtt_path
    """
    s = settings or Settings()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "writesubtitles": False,
        "writeautomaticsub": True,
        "subtitleslangs": s.youtube_caption_langs,
        "subtitlesformat": "vtt",
        "skip_download": False,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        video_id = info["id"]
        title = info.get("title", "unknown")

    wav_path = _resolve_wav(output_dir, video_id, s.sample_rate)
    vtt_path = _resolve_vtt(output_dir, video_id)

    return {
        "video_id": video_id,
        "title": title,
        "audio_path": str(wav_path),
        "vtt_path": str(vtt_path),
    }


def _resolve_wav(output_dir: Path, video_id: str, sample_rate: int) -> Path:
    """Find or convert audio to WAV."""
    wav_path = output_dir / f"{video_id}.wav"
    if wav_path.exists():
        return wav_path
    for f in output_dir.glob(f"{video_id}.*"):
        if f.suffix in (".wav", ".mp3", ".m4a", ".opus"):
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(f), "-ar", str(sample_rate), "-ac", "1", str(wav_path)],
                check=True, capture_output=True,
            )
            if f.suffix != ".wav":
                f.unlink()
            return wav_path
    raise FileNotFoundError(f"No audio file found for {video_id}")


def _resolve_vtt(output_dir: Path, video_id: str) -> Path:
    """Find the VTT caption file."""
    for p in output_dir.glob(f"{video_id}*.vtt"):
        content = p.read_text(encoding="utf-8", errors="replace")
        if "WEBVTT" in content:
            return p
    raise FileNotFoundError(f"No VTT caption found for {video_id}")


def get_channel_videos(channel_url: str, max_videos: int = 5) -> list[dict]:
    """List recent videos from a channel or playlist."""
    opts = {"quiet": True, "extract_flat": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    videos = []
    if "entries" in info:
        for entry in info["entries"][:max_videos]:
            if entry:
                videos.append({
                    "url": f"https://www.youtube.com/watch?v={entry['id']}",
                    "title": entry.get("title", "unknown"),
                })
    return videos
