"""Quick check VAD pipeline output quality."""
import pyarrow.parquet as pq, io, wave, statistics

tab = pq.read_table("dataset_test/T_zgDuLSIYU_train.parquet")
texts = tab.column("transcription").to_pylist()
audios = tab.column("audio").to_pylist()

durs = []
for a in audios:
    with io.BytesIO(a["bytes"]) as buf:
        with wave.open(buf) as w:
            durs.append(w.getnframes() / w.getframerate())

no_punct = sum(1 for t in texts if not t or t[-1] not in ".!?")
print(f"Train: {len(tab)} rows")
print(f"Mean: {statistics.mean(durs):.1f}s, Min: {min(durs):.1f}s, Max: {max(durs):.1f}s")
print(f"Under 5s: {len([d for d in durs if d<5.0])}, Over 20s: {len([d for d in durs if d>20.0])}")
print(f"No punct: {no_punct}")

t2 = pq.read_table("dataset_test/T_zgDuLSIYU_test.parquet")
print(f"Test: {len(t2)} rows")
