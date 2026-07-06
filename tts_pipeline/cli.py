"""CLI entry point for the TTS pipeline."""

import argparse
from pathlib import Path

from tts_pipeline.config import Settings
from tts_pipeline.downloader import download_video_and_subs, get_channel_videos
from tts_pipeline.parser import parse_vtt_full, merge_cues, export_segments_to_vtt, load_vtt_segments
from tts_pipeline.processor import (
    clean_text,
)
from tts_pipeline.exporter import export_dataset
from tts_pipeline.punctuator import punctuate_vtt_file


def run_pipeline(
    video_url: str,
    raw_dir: str | None = None,
    output_dir: str | None = None,
    settings: Settings | None = None,
) -> dict:
    """Run the full pipeline for a single video URL."""
    s = settings or Settings()
    raw = raw_dir or str(s.raw_dir)
    out = output_dir or str(s.output_dir)

    print(f"\n{'='*60}")
    print(f"[PROCESS] {video_url}")
    print(f"{'='*60}")

    # Step 1: Download
    print("\n[Step 1] Download video + caption...")
    video_info = download_video_and_subs(video_url, raw, s)
    print(f"   ID: {video_info['video_id']}")
    print(f"   Title: {video_info['title']}")

    # Step 2: Parse + merge
    print("\n[Step 2] Parse caption & merge cues...")
    raw_cues = parse_vtt_full(video_info["vtt_path"])
    print(f"   Raw cues: {len(raw_cues)}")
    merged = merge_cues(raw_cues, s)
    print(f"   After merge: {len(merged)}")

    for seg in merged:
        seg["text"] = clean_text(seg["text"])

    # Export merged segments → VTT file để inspect
    merged_vtt = str(Path(raw) / f"{video_info['video_id']}_merged.vtt")
    export_segments_to_vtt(merged, merged_vtt)
    print(f"   Saved merged VTT: {merged_vtt}")

    # Punctuation restoration
    from tts_pipeline.punctuator import _split_segment as _do_split
    merged = [_do_split(s)[0] for s in merged]
    punct_vtt = str(Path(raw) / f"{video_info['video_id']}_punctuated.vtt")
    export_segments_to_vtt(merged, punct_vtt)
    print(f"   Saved punctuated VTT: {punct_vtt}")

    # Step 3: VAD speech detection + re-segment
    print("\n[Step 3] VAD speech detection...")
    from tts_pipeline.vad import get_speech_intervals
    from tts_pipeline.vad_aligner import group_vad_intervals, align_text_to_groups

    if s.vad_enabled:
        speech = get_speech_intervals(
            video_info["audio_path"],
            sample_rate=s.sample_rate,
            threshold=s.vad_threshold,
            min_speech_dur=s.vad_min_speech_dur,
            min_silence_dur=s.vad_min_silence_dur,
        )
        print(f"   Speech intervals: {len(speech)}")

        groups = group_vad_intervals(speech, s.min_segment_dur, s.max_segment_dur)
        print(f"   After grouping: {len(groups)}")

        segments = align_text_to_groups(groups, raw_cues, punctuation=True)
        print(f"   After text alignment: {len(segments)}")
    else:
        from tts_pipeline.processor import fix_time_overlaps, segment_by_content
        merged = fix_time_overlaps(merged)
        segments = segment_by_content(merged, s.min_segment_dur, s.max_segment_dur)
        segments = fix_time_overlaps(segments)

    segments = [seg for seg in segments
                if seg["text"]
                and len(seg["text"]) >= s.min_text_len
                and (seg["end"] - seg["start"]) >= s.min_segment_dur]
    print(f"   After filter: {len(segments)}")

    # Step 4: Export
    print("\n[Step 4] Cut audio & export dataset...")
    result = export_dataset(
        segments, out, video_info["video_id"],
        video_info["audio_path"], s,
        split_train_test=True,
    )

    print(f"\n[OK] Raw: {raw}")
    print(f"[OK] Dataset: {out}")
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="TTS Dataset from YouTube Audio Books")
    parser.add_argument("url", help="YouTube video URL or channel URL")
    parser.add_argument("--max-videos", type=int, default=1, help="Max videos for channel")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--raw", default=None, help="Raw download directory")
    parser.add_argument("--config", default=None, help="Path to YAML config file")

    args = parser.parse_args(argv)

    # Load config
    if args.config:
        settings = Settings.from_yaml(args.config)
    else:
        settings = Settings.from_cli_overrides(
            raw_dir=args.raw,
            output_dir=args.output,
        )

    # Detect channel/playlist
    last_seg = args.url.rstrip("/").split("/")[-1]
    if "@" in last_seg or "/playlist" in args.url or "/@" in args.url:
        print(f"[Channel] Getting {args.max_videos} videos...")
        videos = get_channel_videos(args.url, args.max_videos)
        print(f"   Found {len(videos)} videos:")
        for v in videos:
            print(f"   - {v['title']}")
        all_results = []
        for v in videos:
            result = run_pipeline(
                v["url"], raw_dir=args.raw,
                output_dir=args.output, settings=settings,
            )
            all_results.append(result)
        print(f"\n{'='*60}")
        print("[SUMMARY]:")
        for r in all_results:
            print(f"   {r['video_id']}: {r['success']} segments")
    else:
        run_pipeline(
            args.url, raw_dir=args.raw,
            output_dir=args.output, settings=settings,
        )


if __name__ == "__main__":
    main()
