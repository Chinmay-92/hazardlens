#!/usr/bin/env python3
"""Cut the screencast into per-segment clips, add slides, lay each narration clip in place and concatenate.

Inputs: segments.json, audio/*.aiff + audio/durations.json (make_audio.sh), slides/*.png (render_slides.js),
build/app/{frames,timeline.json} (record_app.js). Outputs: hazardlens_demo_body.mp4, intro_placeholder.mp4.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
FFMPEG = os.environ.get("FFMPEG", "/opt/homebrew/bin/ffmpeg")
W, H, FPS = 1920, 1080, 30
VENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS)]
AENC = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
SLIDE_FOR = {"01_title": "title", "09_architecture": "architecture", "10_workflow": "workflow",
             "11_results": "results", "12_failures": "failures", "13_close": "close"}

segs = json.load(open("segments.json"))
dur = json.load(open("audio/durations.json"))
tl = json.load(open("build/app/timeline.json"))
LEAD, TAIL = tl["lead"], tl["tail"]
app_segs = {s["id"]: s for s in tl["segments"]}
frames = tl["frames"]
os.makedirs("build/seg", exist_ok=True)


def run(cmd):
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", *cmd], check=True)


def audio_filter(length):
    ms = int(LEAD * 1000)
    return f"[1:a]aresample=48000,aformat=channel_layouts=stereo,adelay={ms}|{ms},apad,atrim=0:{length:.3f}[a]"


outs = []
for s in segs:
    sid = s["id"]
    out = f"build/seg/{sid}.mp4"
    aiff = f"audio/{sid}.aiff"
    if sid in SLIDE_FOR:
        length = LEAD + dur[sid] + TAIL
        fade = f"fade=t=in:st=0:d=0.35,fade=t=out:st={length - 0.35:.3f}:d=0.35"
        run(["-loop", "1", "-framerate", str(FPS), "-t", f"{length:.3f}", "-i", f"slides/{SLIDE_FOR[sid]}.png", "-i", aiff,
             "-filter_complex", f"[0:v]scale={W}:{H},format=yuv420p,{fade}[v];{audio_filter(length)}",
             "-map", "[v]", "-map", "[a]", "-t", f"{length:.3f}", *VENC, *AENC, out])
    else:
        a = app_segs[sid]
        t0, t1, length = a["start"], a["end"], a["len"]
        # last frame at/before t0 holds until the next frame; frames inside the window follow their timestamps
        idx = [i for i, f in enumerate(frames) if f["t"] <= t0]
        first = idx[-1] if idx else 0
        chosen = [frames[first]] + [f for f in frames[first + 1:] if f["t"] < t1]
        lst = f"build/seg/{sid}.txt"
        with open(lst, "w") as fh:
            for i, f in enumerate(chosen):
                st = max(f["t"], t0)
                en = chosen[i + 1]["t"] if i + 1 < len(chosen) else t1
                fh.write(f"file '{os.path.join(HERE, 'build/app/frames', f['file'])}'\nduration {max(en - st, 0.001):.4f}\n")
            fh.write(f"file '{os.path.join(HERE, 'build/app/frames', chosen[-1]['file'])}'\n")
        run(["-f", "concat", "-safe", "0", "-i", lst, "-i", aiff,
             "-filter_complex", f"[0:v]scale={W}:{H}:flags=lanczos,fps={FPS},format=yuv420p[v];{audio_filter(length)}",
             "-map", "[v]", "-map", "[a]", "-t", f"{length:.3f}", *VENC, *AENC, out])
    outs.append(out)
    print(f"{sid:18s} done")

with open("build/concat.txt", "w") as fh:
    fh.writelines(f"file '{os.path.join(HERE, o)}'\n" for o in outs)
run(["-f", "concat", "-safe", "0", "-i", "build/concat.txt", "-c", "copy", "-movflags", "+faststart", "hazardlens_demo_body.mp4"])

# 15 s placeholder for the entrant's face-to-camera intro (silent stereo track so concat stays uniform)
run(["-loop", "1", "-framerate", str(FPS), "-t", "15", "-i", "slides/intro_placeholder.png",
     "-f", "lavfi", "-t", "15", "-i", "anullsrc=r=48000:cl=stereo",
     "-vf", f"scale={W}:{H},format=yuv420p", "-t", "15", *VENC, *AENC, "-movflags", "+faststart", "intro_placeholder.mp4"])
print("wrote hazardlens_demo_body.mp4 and intro_placeholder.mp4")
