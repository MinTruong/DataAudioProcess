"""Batch process multiple YouTube URLs and merge results."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tts_pipeline.cli import run_pipeline
from tts_pipeline.config import Settings


def main():
    """Read URLs from stdin or file, process each, output summary."""
    if len(sys.argv) > 1 and sys.argv[1] == "--urls":
        urls = [sys.argv[2]]
    elif len(sys.argv) > 1 and sys.argv[1] == "--file":
        with open(sys.argv[2], "r") as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        print("Usage: python scripts/batch_process.py --url <URL>")
        print("       python scripts/batch_process.py --file <urls.txt>")
        sys.exit(1)

    settings = Settings()
    results = []

    for url in urls:
        r = run_pipeline(url, settings=settings)
        results.append(r)

    summary_path = Path(settings.output_dir) / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = sum(r["success"] for r in results)
    failed = sum(r["fail"] for r in results)
    print(f"\nBatch complete: {total} samples, {failed} failures")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
