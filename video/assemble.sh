#!/usr/bin/env bash
# Prepend the entrant's own face-to-camera intro to the demo body.
# Usage: video/assemble.sh path/to/my_intro.(mov|mp4|webm) [output.mp4]
# Normalises the intro to 1920x1080 @30fps (letterboxed, no stretching), H.264 + AAC 48 kHz stereo;
# adds a silent track if the clip has no audio. Output defaults to video/hazardlens_final.mp4.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FFMPEG="${FFMPEG:-$(command -v ffmpeg || echo /opt/homebrew/bin/ffmpeg)}"
FFPROBE="${FFPROBE:-$(command -v ffprobe || echo /opt/homebrew/bin/ffprobe)}"
INTRO="${1:?usage: assemble.sh <intro clip> [output.mp4]}"
OUT="${2:-$HERE/hazardlens_final.mp4}"
BODY="$HERE/hazardlens_demo_body.mp4"
[ -f "$BODY" ] || { echo "missing $BODY (run build.py first)" >&2; exit 1; }
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# Encoder settings identical to build.py so the two parts can be joined losslessly with the concat demuxer.
ENC="-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 30 -video_track_timescale 15360 -c:a aac -b:a 192k -ar 48000 -ac 2"
VF="scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps=30,format=yuv420p"
if [ -n "$("$FFPROBE" -v error -select_streams a -show_entries stream=index -of csv=p=0 "$INTRO")" ]; then
  "$FFMPEG" -loglevel error -y -i "$INTRO" -vf "$VF" -af "aresample=48000,aformat=channel_layouts=stereo" \
    $ENC -shortest "$TMP/intro.mp4"
else
  "$FFMPEG" -loglevel error -y -i "$INTRO" -f lavfi -i anullsrc=r=48000:cl=stereo -vf "$VF" \
    -map 0:v -map 1:a $ENC -shortest "$TMP/intro.mp4"
fi
printf "file '%s'\nfile '%s'\n" "$TMP/intro.mp4" "$BODY" > "$TMP/list.txt"
"$FFMPEG" -loglevel error -y -f concat -safe 0 -i "$TMP/list.txt" -c copy -movflags +faststart "$OUT"
D="$("$FFPROBE" -v error -show_entries format=duration -of csv=p=0 "$OUT")"
echo "wrote $OUT (${D}s)"
awk -v d="$D" 'BEGIN{ if (d > 300) { print "WARNING: longer than 5:00 - trim your intro"; exit 1 } }'
