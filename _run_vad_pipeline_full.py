#!/usr/bin/env python
"""Run full VAD pipeline - cleans old data, runs, prints results."""
import sys, os, shutil

sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(__file__))

# Clean old data
for f in os.listdir("dataset_test"):
    fp = os.path.join("dataset_test", f)
    if f.startswith("T_zgDuLSIYU"):
        os.remove(fp)
        print(f"Removed {fp}")

# Run pipeline
from tts_pipeline.cli import run_pipeline
from tts_pipeline.config import Settings

s = Settings()
result = run_pipeline(
    "https://www.youtube.com/watch?v=T_zgDuLSIYU",
    raw_dir="raw",
    output_dir="dataset_test",
    settings=s,
)
print(f"\nResult: {result['success']} segments")
print(f"Paths: {result['parquet_paths']}")
