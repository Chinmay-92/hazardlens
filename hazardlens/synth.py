"""Deterministic synthetic floor-scene generator (no third-party or copyrighted imagery).

Each scene is a top-down-ish phone photo of a walkway floor with optional trip/slip
hazards, distractors (soft shadows, floor seams) and capture degradations (dark, blur).
Ground truth (hazard boxes + expected agent outcome) is emitted alongside each image.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2 as cv
import numpy as np

W, H = 640, 480

HIGH_SEVERITY_TYPES = {"cable", "spill"}


@dataclass
class Scene:
    name: str
    kind: str
    floor: str
    hazards: list = field(default_factory=list)  # [{"type", "bbox": [x,y,w,h]}]
    degradation: str = "none"
    distractor: str = "none"
    expected_action: str = "clear"

    def to_json(self) -> dict:
        return asdict(self)


def expected_action_for(hazards: list, degradation: str) -> str:
    if degradation == "blurry":
        return "recapture"
    types = {h["type"] for h in hazards}
    if types & HIGH_SEVERITY_TYPES:
        return "escalate"
    if types:
        return "work_order"
    return "clear"


# ---------------------------------------------------------------- floors

def _floor_carpet(rng: np.random.Generator) -> np.ndarray:
    base = np.array(rng.choice([[96, 110, 128], [70, 90, 110], [120, 120, 112], [88, 104, 92]]), np.float32)
    base += rng.uniform(-10, 10, 3)
    img = np.ones((H, W, 3), np.float32) * base
    noise = rng.normal(0, 6, (H, W, 1)).astype(np.float32)
    img += cv.GaussianBlur(noise, (0, 0), 1.0)[..., None]
    return img


def _floor_wood(rng: np.random.Generator) -> np.ndarray:
    tone = np.array([60, 105, 150], np.float32) + rng.uniform(-12, 12, 3)
    img = np.ones((H, W, 3), np.float32) * tone
    plank_h = int(rng.integers(70, 110))
    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
    grain = 6 * np.sin(xs / rng.uniform(8, 14) + 3 * np.sin(ys / 23.0))
    img += grain[..., None]
    for y0 in range(0, H, plank_h):
        img[y0:y0 + plank_h] *= rng.uniform(0.92, 1.06)
        img[max(0, y0 - 1):y0 + 1] *= 0.72  # thin seam (straight, full width) - distractor
    img += rng.normal(0, 3, (H, W, 1)).astype(np.float32)
    return img


def _floor_tile(rng: np.random.Generator) -> np.ndarray:
    tone = np.array(rng.choice([[200, 200, 196], [176, 186, 190], [150, 160, 170]]), np.float32)
    img = np.ones((H, W, 3), np.float32) * tone
    step = int(rng.integers(110, 150))
    grout = tone * 0.82
    for x in range(int(rng.integers(0, step)), W, step):
        img[:, x:x + 3] = grout
    for y in range(int(rng.integers(0, step)), H, step):
        img[y:y + 3, :] = grout
    img += rng.normal(0, 3, (H, W, 1)).astype(np.float32)
    return img


FLOORS = {"carpet": _floor_carpet, "wood": _floor_wood, "tile": _floor_tile}


def _lighting(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = rng.uniform(0.3, 0.7) * W, rng.uniform(0.2, 0.6) * H
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / np.hypot(W, H)
    gain = 1.08 - 0.35 * r
    return img * gain[..., None]


# ---------------------------------------------------------------- hazards

def _bbox_from_mask(mask: np.ndarray) -> list[int]:
    x, y, w, h = cv.boundingRect(cv.findNonZero(mask))
    return [int(x), int(y), int(w), int(h)]


def _place(rng, margin=40):
    return float(rng.uniform(margin, W - margin)), float(rng.uniform(margin + 20, H - margin))


def _add_cable(img, rng):
    mask = np.zeros((H, W), np.uint8)
    p0 = np.array(_place(rng, 20))
    pts = [p0]
    ang = rng.uniform(0, 2 * np.pi)
    length = rng.uniform(260, 520)
    n = 40
    for _ in range(n):
        ang += rng.normal(0, 0.12)
        nxt = pts[-1] + (length / n) * np.array([np.cos(ang), np.sin(ang)])
        nxt = np.clip(nxt, [5, 5], [W - 6, H - 6])
        pts.append(nxt)
    poly = np.array(pts, np.int32).reshape(-1, 1, 2)
    thick = int(rng.integers(6, 11))
    cv.polylines(mask, [poly], False, 255, thick, cv.LINE_AA)
    colour = np.array(rng.choice([[25, 25, 25], [235, 235, 235], [30, 30, 140], [40, 40, 40]]), np.float32)
    m = (mask.astype(np.float32) / 255.0)[..., None]
    img[:] = img * (1 - m) + colour * m
    return {"type": "cable", "bbox": _bbox_from_mask(mask)}


def _add_clutter(img, rng):
    mask = np.zeros((H, W), np.uint8)
    cx, cy = _place(rng, 90)
    w, h = rng.uniform(50, 140), rng.uniform(40, 110)
    ang = rng.uniform(0, 180)
    if rng.random() < 0.5:
        box = cv.boxPoints(((cx, cy), (w, h), ang)).astype(np.int32)
        cv.fillConvexPoly(mask, box, 255, cv.LINE_AA)
    else:
        cv.ellipse(mask, (int(cx), int(cy)), (int(w / 2), int(h / 2)), ang, 0, 360, 255, -1, cv.LINE_AA)
    colour = np.array(rng.choice([[40, 60, 200], [200, 80, 30], [30, 160, 60], [20, 20, 20], [180, 40, 160], [0, 200, 230]]), np.float32)
    shade = cv.GaussianBlur(mask, (0, 0), 6).astype(np.float32) / 255.0
    m = (mask.astype(np.float32) / 255.0)[..., None]
    obj = colour * (0.75 + 0.25 * shade[..., None])
    img[:] = img * (1 - m) + obj * m
    return {"type": "clutter", "bbox": _bbox_from_mask(mask)}


def _add_spill(img, rng):
    mask = np.zeros((H, W), np.uint8)
    cx, cy = _place(rng, 100)
    for _ in range(int(rng.integers(3, 7))):
        r = int(rng.uniform(25, 55))
        ox, oy = rng.normal(0, 30, 2)
        cv.circle(mask, (int(cx + ox), int(cy + oy)), r, 255, -1, cv.LINE_AA)
    soft = cv.GaussianBlur(mask, (0, 0), 3).astype(np.float32) / 255.0
    # glossy liquid: lifts luminance, desaturates, specular highlight
    target = np.array([235, 238, 240], np.float32)
    alpha = 0.55 * soft[..., None]
    img[:] = img * (1 - alpha) + target * alpha
    spec = np.zeros((H, W), np.uint8)
    cv.ellipse(spec, (int(cx - 10), int(cy - 8)), (18, 7), rng.uniform(0, 180), 0, 360, 255, -1)
    spec_f = cv.GaussianBlur(spec, (0, 0), 4).astype(np.float32)[..., None] / 255.0
    img[:] = img * (1 - spec_f) + 255 * spec_f
    return {"type": "spill", "bbox": _bbox_from_mask((soft > 0.3).astype(np.uint8) * 255)}


HAZARDS = {"cable": _add_cable, "clutter": _add_clutter, "spill": _add_spill}


def _add_shadow(img, rng):
    mask = np.zeros((H, W), np.float32)
    cx, cy = _place(rng, 120)
    cv.ellipse(mask, (int(cx), int(cy)), (int(rng.uniform(70, 130)), int(rng.uniform(40, 90))),
               rng.uniform(0, 180), 0, 360, 1.0, -1)
    mask = cv.GaussianBlur(mask, (0, 0), 18)
    img *= (1 - 0.45 * mask)[..., None]


# ---------------------------------------------------------------- scenes

def make_scene(name: str, kind: str, seed: int) -> tuple[np.ndarray, Scene]:
    """kind: comma-free recipe string like 'carpet:cable+clutter|dark|shadow'."""
    rng = np.random.default_rng(seed)
    floor_part, *mods = kind.split("|")
    floor, _, haz = floor_part.partition(":")
    img = FLOORS[floor](rng)
    img = _lighting(img, rng)
    scene = Scene(name=name, kind=kind, floor=floor)
    if "shadow" in mods:
        _add_shadow(img, rng)
        scene.distractor = "shadow"
    for h in [h for h in haz.split("+") if h]:
        scene.hazards.append(HAZARDS[h](img, rng))
    img = np.clip(img, 0, 255)
    if "dark" in mods:
        img = img * rng.uniform(0.18, 0.28)
        img += rng.normal(0, 2.0, img.shape)
        scene.degradation = "dark"
    if "blurry" in mods:
        img = cv.GaussianBlur(img, (0, 0), rng.uniform(7, 10))
        scene.degradation = "blurry"
    img = np.clip(img, 0, 255).astype(np.uint8)
    scene.expected_action = expected_action_for(scene.hazards, scene.degradation)
    return img, scene


# 60-scene evaluation recipe (balanced across floors, hazards and failure modes)
RECIPE: list[str] = (
    ["carpet:", "wood:", "tile:"] * 3 + ["carpet:"]                      # 10 clear
    + ["carpet:cable", "wood:cable", "tile:cable", "carpet:cable"] * 2    # 8 cable
    + ["carpet:clutter", "wood:clutter", "tile:clutter", "wood:clutter"] * 2  # 8 clutter
    + ["carpet:spill", "wood:spill", "tile:spill", "carpet:spill"] * 2    # 8 spill
    + ["carpet:cable+clutter", "wood:spill+clutter", "tile:cable+spill",
       "carpet:clutter+clutter", "wood:cable+clutter", "tile:clutter+spill"]  # 6 multi
    + ["carpet:|shadow", "wood:|shadow", "tile:|shadow"] * 2              # 6 shadow distractor
    + ["carpet:cable|dark", "wood:clutter|dark", "tile:spill|dark",
       "carpet:clutter|dark", "wood:cable|dark", "carpet:|dark"]          # 6 dark
    + ["carpet:cable|blurry", "wood:|blurry", "tile:clutter|blurry", "carpet:spill|blurry"]  # 4 blurry
    + ["carpet:cable|shadow", "wood:clutter|shadow", "tile:cable|shadow", "carpet:spill|shadow"]  # 4 hazard+shadow
)


def generate_dataset(out_dir: Path, seed: int = 2026) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    for i, kind in enumerate(RECIPE):
        name = f"scene_{i:03d}.jpg"
        img, scene = make_scene(name, kind, seed * 1000 + i)
        cv.imwrite(str(out_dir / name), img, [cv.IMWRITE_JPEG_QUALITY, 88])
        labels.append(scene.to_json())
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=1))
    return labels
