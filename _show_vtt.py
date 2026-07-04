"""Read VTT file and print content."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('raw/iUGFXuxHNAA.vi.vtt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines[:60]):
    print(f'{i+1:4d}: {line.rstrip()}')
