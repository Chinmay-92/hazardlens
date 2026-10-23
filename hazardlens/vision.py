"""OpenCV 5 perception tools used by the HazardLens agent.

Every public function is a *tool*: it takes an image (BGR uint8) plus JSON-serialisable
arguments and returns a JSON-serialisable dict, so the same functions can be called by
the deterministic planner, by an LLM via tool-use, or over HTTP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2 as cv
import numpy as np

WORK_W = 640  # analysis width; boxes are mapped back to original coordinates

# Quality thresholds (tuned on the synthetic dev split; see docs/TECHNICAL_REPORT.md)
DARK_L = 70.0           # mean L* (0-255 OpenCV scale) below this -> too dark
LOW_CONTRAST = 4.0     # std of L* below this -> flat / washed out
BLUR_SHARPNESS = 40.0   # 90th percentile tile Laplacian variance below this -> blurry
GLARE_FRAC = 0.25       # fraction of saturated highlights above this -> overexposed


def opencv_version() -> str:
    return cv.__version__


def _resize(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if w == WORK_W:
        return img, 1.0
    s = WORK_W / float(w)
    return cv.resize(img, (WORK_W, max(1, int(round(h * s)))), interpolation=cv.INTER_AREA), s


# ------------------------------------------------------------------ quality

def assess_quality(img: np.ndarray) -> dict:
    """Exposure, contrast, sharpness and glare check. Returns metrics + verdict."""
    small, _ = _resize(img)
    lab = cv.cvtColor(small, cv.COLOR_BGR2Lab)
    L = lab[..., 0].astype(np.float32)
    gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
    lap = cv.Laplacian(gray, cv.CV_64F, ksize=3)
    # sharpness = 90th percentile of per-tile Laplacian variance (robust to plain floors)
    th, tw = gray.shape[0] // 6, gray.shape[1] // 8
    tiles = [lap[y:y + th, x:x + tw].var() for y in range(0, th * 6, th) for x in range(0, tw * 8, tw)]
    sharp = float(np.percentile(tiles, 90))
    hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)
    glare = float(np.mean((hsv[..., 2] > 250) & (hsv[..., 1] < 25)))
    m = {
        "mean_luminance": round(float(L.mean()), 2),
        "contrast": round(float(L.std()), 2),
        "sharpness": round(sharp, 2),
        "glare_fraction": round(glare, 4),
    }
    issues = []
    if m["mean_luminance"] < DARK_L:
        issues.append("dark")
    if m["contrast"] < LOW_CONTRAST:
        issues.append("low_contrast")
    if sharp < BLUR_SHARPNESS:
        issues.append("blurry")
    if glare > GLARE_FRAC:
        issues.append("overexposed")
    m["issues"] = issues
    # enhancement can fix exposure/contrast but not motion/defocus blur or blown highlights
    if "blurry" in issues or "overexposed" in issues:
        m["verdict"] = "unusable"
    elif issues:
        m["verdict"] = "enhance"
    else:
        m["verdict"] = "ok"
    return m


def enhance(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """Low-light recovery: denoise -> gamma lift -> CLAHE on L* channel."""
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    mean_l = float(lab[..., 0].mean())
    out = cv.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21) if mean_l < DARK_L else img.copy()
    target = 0.5
    gamma = float(np.clip(math.log(target) / math.log(max(mean_l / 255.0, 1e-3)), 0.25, 1.0))
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
    out = cv.LUT(out, lut)
    lab = cv.cvtColor(out, cv.COLOR_BGR2Lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    out = cv.cvtColor(lab, cv.COLOR_Lab2BGR)
    return out, {"gamma": round(gamma, 3), "clahe_clip": 2.0, "denoised": mean_l < DARK_L}


# ------------------------------------------------------------------ floor model

@dataclass
class FloorModel:
    centers: np.ndarray  # (k,3) Lab centres considered "floor"
    weights: np.ndarray  # channel weights (L down-weighted for illumination robustness)

    def distance(self, lab: np.ndarray) -> np.ndarray:
        x = lab.reshape(-1, 1, 3).astype(np.float32)
        d = np.sqrt((((x - self.centers[None]) * self.weights) ** 2).sum(-1)).min(1)
        return d.reshape(lab.shape[:2])


def fit_floor_model(lab: np.ndarray, k: int = 4, min_share: float = 0.18) -> FloorModel:
    """k-means on Lab pixels; every cluster covering >= min_share of the frame is floor."""
    small = cv.resize(lab, (80, 60), interpolation=cv.INTER_AREA).reshape(-1, 3).astype(np.float32)
    crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv.setRNGSeed(7)
    _, labels, centers = cv.kmeans(small, k, None, crit, 3, cv.KMEANS_PP_CENTERS)
    share = np.bincount(labels.ravel(), minlength=k) / float(len(labels))
    keep = share >= min_share
    if not keep.any():
        keep[np.argmax(share)] = True
    return FloorModel(centers=centers[keep], weights=np.array([0.45, 1.0, 1.0], np.float32))


def _illum_normalise(lab: np.ndarray) -> np.ndarray:
    """Divide out slow illumination gradients from L* (vignetting, uneven lighting).

    A robust 2-D quadratic surface is fitted to a downsampled L* map with cv.solve
    (SVD) and two rounds of MAD outlier rejection, so local objects/spills are *not*
    absorbed into the illumination model (unlike a large median/Gaussian background).
    """
    L = lab[..., 0].astype(np.float32)
    h, w = L.shape
    sh, sw = 48, 64
    Ls = cv.resize(L, (sw, sh), interpolation=cv.INTER_AREA).ravel()
    ys, xs = np.mgrid[0:sh, 0:sw].astype(np.float32)
    xs, ys = (xs / sw - 0.5).ravel(), (ys / sh - 0.5).ravel()
    A = np.stack([np.ones_like(xs), xs, ys, xs * xs, ys * ys, xs * ys], 1)
    keep = np.ones(len(Ls), bool)
    coef = None
    for _ in range(3):
        ok, coef = cv.solve(A[keep], Ls[keep].reshape(-1, 1), flags=cv.DECOMP_SVD)
        resid = Ls - (A @ coef).ravel()
        mad = float(np.median(np.abs(resid[keep] - np.median(resid[keep])))) + 1e-3
        keep = np.abs(resid - np.median(resid[keep])) < 2.5 * 1.4826 * mad + 2.0
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = gx / w - 0.5, gy / h - 0.5
    c = coef.ravel()
    bg = c[0] + c[1] * gx + c[2] * gy + c[3] * gx * gx + c[4] * gy * gy + c[5] * gx * gy
    Ln = np.clip(L / np.maximum(bg, 1.0) * float(np.median(L)), 0, 255)
    out = lab.copy()
    out[..., 0] = Ln.astype(np.uint8)
    return out


# ------------------------------------------------------------------ detection

def _classify(comp_mask: np.ndarray, lab: np.ndarray, floor: FloorModel) -> tuple[str, dict]:
    cnts, _ = cv.findContours(comp_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv.contourArea)
    area = float(np.count_nonzero(comp_mask))
    (_, _), (rw, rh), _ = cv.minAreaRect(cnt)
    elong = max(rw, rh) / max(1.0, min(rw, rh))
    fill = area / max(1.0, rw * rh)
    dist = cv.distanceTransform(comp_mask, cv.DIST_L2, 3)
    thickness = float(2.0 * dist[comp_mask > 0].mean()) if area else 0.0
    hull = cv.convexHull(cnt)
    solidity = area / max(1.0, cv.contourArea(hull))
    px = lab[comp_mask > 0].astype(np.float32)
    L, a, b = px[:, 0].mean(), px[:, 1].mean() - 128, px[:, 2].mean() - 128
    fL = float(floor.centers[:, 0].mean())
    chroma = float(math.hypot(a, b))
    feats = {"elongation": round(elong, 2), "fill": round(fill, 3), "thickness_px": round(thickness, 2),
             "solidity": round(solidity, 3), "delta_L": round(float(L - fL), 1), "chroma": round(chroma, 1)}
    if thickness < 10.0 or (thickness < 14.0 and (elong > 3.5 or solidity < 0.45)):
        return "cable", feats
    if L - fL > 18 and chroma < 20 and solidity > 0.6:
        return "spill", feats
    return "clutter", feats


def _straight_span(comp_mask: np.ndarray) -> bool:
    """Floor seams / grout: thin straight structures spanning most of the frame."""
    h, w = comp_mask.shape
    x, y, bw, bh = cv.boundingRect(comp_mask)
    if not (bw > 0.85 * w or bh > 0.85 * h):
        return False
    pts = cv.findNonZero(comp_mask).reshape(-1, 2).astype(np.float32)
    vx, vy, x0, y0 = cv.fitLine(pts, cv.DIST_L2, 0, 0.01, 0.01).ravel()
    resid = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
    return float(np.percentile(resid, 90)) < 6.0


def detect_hazards(img: np.ndarray, sensitivity: float = 0.5) -> dict:
    """Segment non-floor regions and classify them as cable / clutter / spill candidates."""
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    small, s = _resize(img)
    blur = cv.GaussianBlur(small, (5, 5), 0)
    lab = _illum_normalise(cv.cvtColor(blur, cv.COLOR_BGR2Lab))
    floor = fit_floor_model(lab)
    d = floor.distance(lab)
    spread = float(np.median(d) + 3.0 * np.median(np.abs(d - np.median(d))))
    thr = max(spread, 26.0 - 12.0 * sensitivity)
    mask = (d > thr).astype(np.uint8) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9)))
    n, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    min_area = (0.0035 - 0.002 * sensitivity) * mask.size
    hh, ww = mask.shape
    dets, rejected = [], []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        if _straight_span(comp):
            rejected.append({"bbox": _scale_box([x, y, w, h], s), "reason": "straight_floor_seam"})
            continue
        htype, feats = _classify(comp, lab, floor)
        contrast = float(d[labels == i].mean() / thr)
        conf = 1.0 / (1.0 + math.exp(-2.2 * (contrast - 1.6)))
        conf *= min(1.0, 0.6 + area / (0.01 * mask.size))
        cx = (x + w / 2) / ww
        dets.append({
            "id": f"h{len(dets) + 1}",
            "type": htype,
            "bbox": _scale_box([x, y, w, h], s),
            "area_fraction": round(area / mask.size, 4),
            "confidence": round(float(np.clip(conf, 0, 1)), 3),
            "in_walk_path": bool(0.2 <= cx <= 0.8),
            "features": feats,
        })
    dets.sort(key=lambda dd: -dd["area_fraction"])
    return {"detections": dets, "rejected": rejected, "threshold": round(thr, 2),
            "floor_clusters": int(len(floor.centers)), "sensitivity": sensitivity}


def _scale_box(b, s):
    return [int(round(v / s)) for v in b]


# ------------------------------------------------------------------ active perception

def inspect_region(img: np.ndarray, bbox: list[int], pad: float = 0.6) -> dict:
    """Zoom into a candidate, build a *local* floor model from the surrounding ring and
    test shadow vs object vs liquid. Used by the agent to confirm or reject candidates."""
    H, W = img.shape[:2]
    x, y, w, h = bbox
    px, py = int(w * pad) + 12, int(h * pad) + 12
    x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return {"confirmed": False, "reason": "empty_crop"}
    scale = 2.0 if max(crop.shape[:2]) < 400 else 1.0
    crop = cv.resize(crop, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    lab = cv.cvtColor(cv.GaussianBlur(crop, (5, 5), 0), cv.COLOR_BGR2Lab).astype(np.float32)
    ch, cw = lab.shape[:2]
    inner = np.zeros((ch, cw), np.uint8)
    ix0, iy0 = int((x - x0) * scale), int((y - y0) * scale)
    cv.rectangle(inner, (ix0, iy0), (ix0 + int(w * scale), iy0 + int(h * scale)), 255, -1)
    ring = cv.bitwise_not(cv.dilate(inner, np.ones((15, 15), np.uint8)))
    if np.count_nonzero(ring) < 50:
        ring = np.zeros_like(inner)
        cv.rectangle(ring, (0, 0), (cw - 1, ch - 1), 255, 6)
    ref = np.median(lab[ring > 0], axis=0)
    dLab = lab - ref
    dL = dLab[..., 0]
    dC = np.sqrt(dLab[..., 1] ** 2 + dLab[..., 2] ** 2)
    ref_chroma = float(math.hypot(ref[1] - 128, ref[2] - 128))
    obj = ((np.abs(dL) * 0.45) ** 2 + dC ** 2) ** 0.5 > 12
    obj &= inner > 0
    frac = float(obj.mean() / max(1e-6, (inner > 0).mean()))
    if frac < 0.03:
        return {"confirmed": False, "reason": "no_local_contrast", "object_fraction": round(frac, 3)}
    # Shadow test: darker, chroma scales proportionally (hue preserved), soft boundary.
    region = lab[obj]
    mean_dL = float(np.mean(region[:, 0] - ref[0]))
    reg_chroma = float(np.hypot(region[:, 1].mean() - 128, region[:, 2].mean() - 128))
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    gx, gy = cv.Sobel(gray, cv.CV_32F, 1, 0), cv.Sobel(gray, cv.CV_32F, 0, 1)
    mag = cv.magnitude(gx, gy)
    edge = cv.morphologyEx(obj.astype(np.uint8) * 255, cv.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    edge_strength = float(mag[edge].mean()) if edge.any() else 0.0
    ratio_l = (ref[0] + mean_dL) / max(1.0, ref[0])
    hue_shift = float(math.degrees(abs(math.atan2(region[:, 2].mean() - 128, region[:, 1].mean() - 128)
                                       - math.atan2(ref[2] - 128, ref[1] - 128))))
    hue_shift = min(hue_shift, 360 - hue_shift)
    shadow_like = (mean_dL < -8 and edge_strength < 60 and
                   abs(reg_chroma - ref_chroma * ratio_l) < 6 and (hue_shift < 25 or ref_chroma < 6))
    # Hough evidence for linear structure (cables)
    edges = cv.Canny(gray, 40, 120)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=int(20 * scale), maxLineGap=6)
    n_lines = 0 if lines is None else int(len(lines))
    evidence = {"object_fraction": round(frac, 3), "mean_delta_L": round(mean_dL, 1),
                "edge_strength": round(edge_strength, 1), "chroma_ref": round(ref_chroma, 1),
                "chroma_region": round(reg_chroma, 1), "hue_shift_deg": round(hue_shift, 1),
                "hough_segments": n_lines, "zoom": scale}
    if shadow_like:
        return {"confirmed": False, "reason": "shadow_like", **evidence}
    conf = float(np.clip(0.4 + 0.5 * min(1.0, edge_strength / 120.0) + 0.3 * min(1.0, frac), 0, 1))
    return {"confirmed": True, "reason": "local_object_evidence", "confidence": round(conf, 3), **evidence}


# ------------------------------------------------------------------ rendering

COLOURS = {"cable": (0, 0, 255), "spill": (255, 160, 0), "clutter": (0, 200, 255)}


def annotate(img: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = img.copy()
    for d in detections:
        x, y, w, h = d["bbox"]
        status = d.get("status", "candidate")
        col = COLOURS.get(d["type"], (255, 255, 255)) if status != "rejected" else (160, 160, 160)
        thick = 3 if status == "confirmed" else 1
        cv.rectangle(out, (x, y), (x + w, y + h), col, thick, cv.LINE_AA)
        label = f'{d["id"]} {d["type"]} {status}'
        cv.putText(out, label, (x + 2, max(14, y - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(out, label, (x + 2, max(14, y - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv.LINE_AA)
    return out


def iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0
