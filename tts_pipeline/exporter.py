"""Audio cutting and Parquet dataset export.

Embedds audio bytes inside Parquet (compatible with HuggingFace datasets Audio feature),
while keeping WAV files on disk for manual adjustment.
"""

import io
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from tts_pipeline.config import Settings
from tts_pipeline.processor import clean_text


def cut_audio_segment(
    audio_path: str,
    output_path: str,
    start: float,
    end: float,
    sample_rate: int = 22050,
) -> tuple[bool, str]:
    """Cut one audio segment with ffmpeg.

    Returns:
        (success: bool, message_or_path: str)
    """
    if end - start < 2.0 or end - start > 20.0:
        return False, f"Duration {end - start:.1f}s out of range"
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(end - start),
        "-i", str(audio_path),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-sample_fmt", "s16",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True, str(output_path)
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode("utf-8", errors="replace")


def _read_wav_bytes(wav_path: str) -> bytes:
    """Read a WAV file as raw bytes."""
    with open(wav_path, "rb") as f:
        return f.read()


def _build_hf_audio_struct(path: str, audio_bytes: bytes) -> dict:
    """Build an audio struct compatible with HuggingFace datasets Audio feature.

    Format: {"path": "audio/file.wav", "bytes": b"..."}
    Must match ngochuyen_voice: bytes first, path second.
    """
    return {
        "bytes": audio_bytes,
        "path": path,
    }


def _export_split_parquet(
    df: pd.DataFrame,
    split_name: str,
    output_dir: Path,
    video_id: str,
    audio_dir: Path,
    sample_rate: int = 22050,
) -> Path:
    """Save one train/test split to Parquet with embedded audio bytes.

    The 'audio' column is stored as a struct {path, bytes} for HuggingFace datasets
    Audio feature compatibility, while 'audio_path' keeps the disk path for WAV access.
    """
    parquet_path = output_dir / f"{video_id}_{split_name}.parquet"

    # Chuyen audio column thanh struct: {bytes: binary, path: string}
    # Thu tu bytes truoc, path sau giong ngochuyen_voice
    audio_structs = []
    for _, row in df.iterrows():
        # file_name co dang "audio/xxx.wav", path day du cho WAV tren disk
        wav_path = output_dir / row["file_name"]
        if wav_path.exists():
            audio_bytes = _read_wav_bytes(str(wav_path))
        else:
            audio_bytes = b""
        # path: ensure forward slash cho HF viewer
        rel_path = row["audio"].replace("\\", "/")
        audio_structs.append(_build_hf_audio_struct(rel_path, audio_bytes))

    # Tao bang pyarrow, set schema metadata de HF nhan dien Audio feature
    table = pa.Table.from_pydict({
        "audio": pa.array(audio_structs, type=pa.struct([
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string()),
        ])),
        "transcription": pa.array(df["transcription"], type=pa.string()),
        "file_name": pa.array(df["file_name"], type=pa.string()),
    })

    # Ghi kem HF metadata de datasets.load_dataset nhan dien Audio feature
    hf_meta = {
        "info": {
            "features": {
                "audio": {"_type": "Audio", "sampling_rate": sample_rate},
                "transcription": {"_type": "Value", "dtype": "string"},
                "file_name": {"_type": "Value", "dtype": "string"},
            }
        }
    }
    table = table.replace_schema_metadata({"huggingface": json.dumps(hf_meta)})
    pq.write_table(table, parquet_path)
    return parquet_path


def export_dataset(
    segments: list[dict],
    output_dir: str | Path,
    video_id: str,
    audio_source_path: str,
    settings: Settings | None = None,
    split_train_test: bool = True,
) -> dict:
    """Cut audio segments and export dataset as Parquet.

    Args:
        segments: List of {start, end, text} dicts.
        output_dir: Root output directory.
        video_id: YouTube video ID (used for naming).
        audio_source_path: Path to the full audio WAV file.
        settings: Pipeline settings (controls sample_rate, train_split).
        split_train_test: If True, split into train/test Parquet files.

    Returns:
        dict with {video_id, total_segments, success, fail, parquet_paths: str|None}
    """
    s = settings or Settings()
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    success = 0
    fail = 0

    for idx, seg in enumerate(tqdm(segments, desc=f"Cut audio [{video_id}]")):
        dur = seg.get("duration", seg["end"] - seg["start"])
        text = clean_text(seg["text"])

        if dur < s.min_segment_dur or dur > s.max_segment_dur:
            fail += 1
            continue
        if not text or len(text) < s.min_text_len or len(text) > s.max_text_len:
            fail += 1
            continue

        audio_filename = f"{video_id}_{idx:05d}.wav"
        audio_path = audio_dir / audio_filename

        ok, _ = cut_audio_segment(
            audio_source_path, str(audio_path),
            seg["start"], seg["end"], s.sample_rate,
        )
        if ok:
            # Path.relative_to() tra ve backslash tren Windows, can chuan hoa sang /
            rel_path = str(audio_path.relative_to(output_dir)).replace("\\", "/")
            rows.append({
                "audio": rel_path,
                "transcription": text,
                "file_name": f"audio/{audio_filename}",
            })
            success += 1
        else:
            fail += 1

    result = {
        "video_id": video_id,
        "total_segments": len(segments),
        "success": success,
        "fail": fail,
        "parquet_paths": None,
    }

    if not rows:
        print(f"\n[!] No valid segments for {video_id}")
        return result

    df = pd.DataFrame(rows)

    if split_train_test and s.train_split < 1.0:
        n_train = int(len(df) * s.train_split)
        train_df = df.iloc[:n_train]
        test_df = df.iloc[n_train:]
        train_path = _export_split_parquet(
            train_df, "train", output_dir, video_id, audio_dir, s.sample_rate
        )
        test_path = _export_split_parquet(
            test_df, "test", output_dir, video_id, audio_dir, s.sample_rate
        )
        result["parquet_paths"] = {"train": str(train_path), "test": str(test_path)}
        print(f"\n[OK] Train: {len(train_df)}, Test: {len(test_df)}")
    else:
        path = _export_split_parquet(df, "dataset", output_dir, video_id, audio_dir, s.sample_rate)
        result["parquet_paths"] = str(path)
        print(f"\n[OK] Dataset: {path} ({len(df)} samples)")

    return result
