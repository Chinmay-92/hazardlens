#!/usr/bin/env python
"""Evaluate HazardLens on the synthetic benchmark.

Compares a single-shot OpenCV pipeline (detect once, act on every candidate) against the
agentic loop (quality gate -> enhance/recapture -> detect -> glare-triggered re-detect ->
inspect_region -> severity-gated action). Optionally evaluates the LLM planner.

Usage:
  python scripts/evaluate.py                       # rule planners only (deterministic)
  python scripts/evaluate.py --with-llm            # also run Claude planner (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2 as cv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hazardlens import vision as V  # noqa: E402
from hazardlens.actions import ActionStore  # noqa: E402
from hazardlens.agent import RulePlanner, run_inspection, result_payload  # noqa: E402
from hazardlens.synth import generate_dataset  # noqa: E402

OUTCOMES = ["clear", "work_order", "escalate", "recapture"]


def tags(kind: str) -> list[str]:
    floor_part, *mods = kind.split("|")
    haz = floor_part.split(":")[1]
    t = list(mods)
    hs = [h for h in haz.split("+") if h]
    if not hs:
        t.append("no_hazard")
    elif len(hs) > 1:
        t.append("multi")
    else:
        t.append(hs[0])
    t.append("floor_" + floor_part.split(":")[0])
    return t


def match(gt: list[dict], pred: list[dict], thr: float = 0.3):
    used, tp, type_ok = set(), 0, 0
    for g in gt:
        best, bi = 0.0, None
        for i, p in enumerate(pred):
            if i in used:
                continue
            o = V.iou(g["bbox"], p["bbox"])
            if o > best:
                best, bi = o, i
        if bi is not None and best >= thr:
            used.add(bi)
            tp += 1
            type_ok += pred[bi]["type"] == g["type"]
    return tp, len(pred) - len(used), len(gt) - tp, type_ok


def evaluate(config: str, labels: list[dict], data_dir: Path, out_dir: Path, planner: str = "rule",
             save_failures: bool = True) -> dict:
    rp = {"single_shot": RulePlanner(high_sensitivity_on_glare=False, use_inspection=False,
                                     use_quality_gate=False),
          "agentic": RulePlanner(), "llm": RulePlanner()}[config]
    rows, conf = [], Counter()
    TP = FP = FN = TYPE_OK = 0
    lat, steps = [], []
    fail_dir = out_dir / "failures" / config
    if save_failures:
        fail_dir.mkdir(parents=True, exist_ok=True)
        for f in fail_dir.glob("*.jpg"):
            f.unlink()
    for lab in labels:
        img = cv.imread(str(data_dir / lab["name"]))
        t0 = time.perf_counter()
        s = run_inspection(img, ActionStore(trace_bucket="", alert_topic=""), location=lab["name"],
                           planner=planner, rule_planner=rp)
        ms = (time.perf_counter() - t0) * 1000
        lat.append(ms)
        steps.append(len(s.trace))
        got, exp = s.outcome, lab["expected_action"]
        conf[(exp, got)] += 1
        pred = s.confirmed if got != "recapture" else []
        if exp != "recapture" and got != "recapture":
            tp, fp, fn, tok = match(lab["hazards"], pred)
            TP, FP, FN, TYPE_OK = TP + tp, FP + fp, FN + fn, TYPE_OK + tok
        row = {"name": lab["name"], "kind": lab["kind"], "tags": tags(lab["kind"]), "expected": exp, "got": got,
               "correct": got == exp, "steps": len(s.trace), "ms": round(ms, 1),
               "tools": [t["tool"] for t in s.trace],
               "confirmed": [(d["type"], d["bbox"]) for d in s.confirmed],
               "rejected": [(d["type"], d["inspection"].get("reason")) for d in s.detections.values()
                            if d["status"] == "rejected"],
               "gt": [(g["type"], g["bbox"]) for g in lab["hazards"]],
               "guardrails": [t["guardrail"] for t in s.trace if t.get("guardrail")]}
        rows.append(row)
        if not row["correct"] and save_failures:
            pay = result_payload(s, include_image=False)
            ann = V.annotate(s.working, pay["detections"])
            for g in lab["hazards"]:
                x, y, w, h = g["bbox"]
                cv.rectangle(ann, (x, y), (x + w, y + h), (0, 255, 0), 1, cv.LINE_4)
            cv.imwrite(str(fail_dir / lab["name"]), ann, [cv.IMWRITE_JPEG_QUALITY, 80])
        if config == "agentic" and lab["name"] in ("scene_043.jpg", "scene_046.jpg", "scene_036.jpg",
                                                    "scene_052.jpg"):
            (out_dir / "traces").mkdir(exist_ok=True)
            (out_dir / "traces" / lab["name"].replace(".jpg", ".json")).write_text(
                json.dumps(result_payload(s, include_image=False), indent=1, default=str))
    n = len(rows)
    esc = [r for r in rows if r["expected"] == "escalate"]
    non_esc = [r for r in rows if r["expected"] in ("clear", "work_order")]
    rec = [r for r in rows if r["expected"] == "recapture"]
    by_tag = defaultdict(list)
    for r in rows:
        for t in r["tags"]:
            by_tag[t].append(r["correct"])
    prec = TP / max(1, TP + FP)
    recall = TP / max(1, TP + FN)
    return {
        "config": config, "n_images": n,
        "outcome_accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "escalation_recall": round(sum(r["got"] == "escalate" for r in esc) / max(1, len(esc)), 3),
        "unsafe_misses": [r["name"] for r in esc if r["got"] in ("clear", "work_order")],
        "false_escalation_rate": round(sum(r["got"] == "escalate" for r in non_esc) / max(1, len(non_esc)), 3),
        "recapture_recall": round(sum(r["got"] == "recapture" for r in rec) / max(1, len(rec)), 3),
        "unneeded_recaptures": [r["name"] for r in rows if r["got"] == "recapture" and r["expected"] != "recapture"],
        "detection": {"tp": TP, "fp": FP, "fn": FN, "precision": round(prec, 3), "recall": round(recall, 3),
                      "f1": round(2 * prec * recall / max(1e-9, prec + recall), 3),
                      "type_accuracy_on_matches": round(TYPE_OK / max(1, TP), 3), "iou_threshold": 0.3},
        "accuracy_by_tag": {t: round(sum(v) / len(v), 3) for t, v in sorted(by_tag.items())},
        "confusion": {f"{e}->{g}": c for (e, g), c in sorted(conf.items())},
        "mean_steps": round(statistics.mean(steps), 2),
        "latency_ms": {"mean": round(statistics.mean(lat), 1),
                       "p95": round(sorted(lat)[int(0.95 * (n - 1))], 1)},
        "failures": [r for r in rows if not r["correct"]],
        "rows": rows,
    }


def write_markdown(results: list[dict], path: Path, seed: int = 2026) -> None:
    L = ["# HazardLens evaluation results", "",
         f"OpenCV {V.opencv_version()} · synthetic benchmark (60 images, seed {seed}) · "
         "generated by `python scripts/evaluate.py`", "",
         "| Metric | " + " | ".join(r["config"] for r in results) + " |",
         "|---|" + "---|" * len(results)]

    def row(label, f):
        L.append(f"| {label} | " + " | ".join(str(f(r)) for r in results) + " |")
    row("Outcome accuracy (exact action)", lambda r: r["outcome_accuracy"])
    row("Escalation recall (cable/spill → human)", lambda r: r["escalation_recall"])
    row("Unsafe misses (high hazard not escalated)", lambda r: len(r["unsafe_misses"]))
    row("False escalation rate", lambda r: r["false_escalation_rate"])
    row("Recapture recall (blurry photos)", lambda r: r["recapture_recall"])
    row("Detection precision @IoU0.3", lambda r: r["detection"]["precision"])
    row("Detection recall @IoU0.3", lambda r: r["detection"]["recall"])
    row("Detection F1", lambda r: r["detection"]["f1"])
    row("Type accuracy (matched)", lambda r: r["detection"]["type_accuracy_on_matches"])
    row("Mean tool calls / run", lambda r: r["mean_steps"])
    row("Latency mean / p95 (ms, laptop CPU)", lambda r: f'{r["latency_ms"]["mean"]} / {r["latency_ms"]["p95"]}')
    L += ["", "## Accuracy by scene tag", "", "| Tag | " + " | ".join(r["config"] for r in results) + " |",
          "|---|" + "---|" * len(results)]
    for t in results[0]["accuracy_by_tag"]:
        L.append(f"| {t} | " + " | ".join(str(r["accuracy_by_tag"].get(t, "-")) for r in results) + " |")
    for r in results:
        L += ["", f"## Failure cases — {r['config']} ({len(r['failures'])})", "",
              "| Image | Scene recipe | Expected | Got | Confirmed | Ground truth |", "|---|---|---|---|---|---|"]
        for f in r["failures"]:
            L.append(f"| {f['name']} | `{f['kind']}` | {f['expected']} | {f['got']} | "
                     f"{', '.join(t for t, _ in f['confirmed']) or '-'} | {', '.join(t for t, _ in f['gt']) or '-'} |")
        L.append("")
        L.append(f"Annotated failure images: `failures/{r['config']}/` next to this file (green = ground truth).")
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "samples"))
    ap.add_argument("--out", default=str(ROOT / "eval"))
    ap.add_argument("--regenerate", action="store_true", help="re-create the synthetic dataset")
    ap.add_argument("--seed", type=int, default=2026, help="dataset seed (2026 = dev set, others = held-out)")
    ap.add_argument("--with-llm", action="store_true")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.regenerate or not (data / "labels.json").exists():
        generate_dataset(data, seed=a.seed)
    labels = json.loads((data / "labels.json").read_text())
    results = [evaluate("single_shot", labels, data, out), evaluate("agentic", labels, data, out)]
    if a.with_llm:
        results.append(evaluate("llm", labels, data, out, planner="llm"))
    (out / "results.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "rows"} for r in results], indent=1, default=str))
    (out / "per_image.json").write_text(json.dumps({r["config"]: r["rows"] for r in results}, indent=1,
                                                   default=str))
    write_markdown(results, out / "RESULTS.md", seed=a.seed)
    for r in results:
        print(f"{r['config']:12s} acc={r['outcome_accuracy']:.3f} esc_recall={r['escalation_recall']:.3f} "
              f"unsafe_misses={len(r['unsafe_misses'])} false_esc={r['false_escalation_rate']:.3f} "
              f"det_P={r['detection']['precision']:.3f} det_R={r['detection']['recall']:.3f} "
              f"steps={r['mean_steps']} lat_ms={r['latency_ms']['mean']}")


if __name__ == "__main__":
    main()
