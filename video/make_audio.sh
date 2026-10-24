#!/usr/bin/env bash
# Generate one narration clip per segment with macOS `say` (voice Daniel, en_GB) and record durations.
set -euo pipefail
cd "$(dirname "$0")"
RATE="${RATE:-178}"
mkdir -p audio
python3 - "$RATE" <<'PY'
import json, subprocess, sys
rate = sys.argv[1]
segs = json.load(open("segments.json"))
out = {}
for s in segs:
    aiff = f"audio/{s['id']}.aiff"
    subprocess.run(["say", "-v", "Daniel", "-r", rate, "-o", aiff, s["text"]], check=True)
    d = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", aiff],
                       capture_output=True, text=True, check=True).stdout.strip()
    out[s["id"]] = round(float(d), 3)
    print(f"{s['id']:18s} {out[s['id']]:6.2f}s")
json.dump(out, open("audio/durations.json", "w"), indent=1)
print("total speech", round(sum(out.values()), 1), "s")
PY
