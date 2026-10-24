#!/usr/bin/env python3
"""Write narration.md from segments.json, with body timecodes taken from the rendered segment clips."""
import json, os, subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)))
segs = json.load(open("segments.json"))
def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True, check=True).stdout)
fmt = lambda t: f"{int(t // 60)}:{t % 60:04.1f}"
L = ["# HazardLens demo video: narration", "",
     "First-person script, one block per segment. The synthetic voice (macOS `say -v Daniel`) reads exactly this text;",
     "you can re-record any block in your own voice and keep the same timing (each block has ~0.5 s lead-in and ~0.8 s tail).",
     "Times are for `hazardlens_demo_body.mp4`; add the length of your intro clip for the final video.", "",
     "## 0. Entrant intro (you, on camera, 15–20 s, before the body)", "",
     "> Hi, I'm <your name>, a solo developer from <where you are>. This is HazardLens, an OpenCV 5 agent that looks for",
     "> trip and slip hazards, the kind that cause falls in care homes, and it asks a human before raising an alarm.", ""]
t = 0.0
for i, s in enumerate(segs, 1):
    d = dur(f"build/seg/{s['id']}.mp4")
    L += [f"## {i}. {s['shows']}  ({fmt(t)}–{fmt(t + d)})", "", f"> {s['text']}", ""]
    t += d
L += [f"Body total: {fmt(t)}. With a 20 s intro the final video is {fmt(t + 20)} (limit 5:00).", ""]
open("narration.md", "w").write("\n".join(L))
print("narration.md written; body", fmt(t))
