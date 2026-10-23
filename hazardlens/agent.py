"""HazardLens agent: perception -> decision -> action loop with human-in-the-loop control.

The *executor* (Toolbox) owns state, safety policy and guardrails. A *planner* only
chooses the next tool call. Two planners share the exact same tools:
  * RulePlanner - deterministic, auditable policy (default; used when no LLM is configured)
  * LLMPlanner  - Claude via the Anthropic API or Amazon Bedrock (tool use)
If the LLM fails mid-run, the RulePlanner takes over from the current state.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import cv2 as cv
import numpy as np

from . import vision as V
from .actions import ActionStore

log = logging.getLogger("hazardlens.agent")

MAX_STEPS = 14
MAX_INSPECTIONS = 6
HIGH_TYPES = {"cable", "spill"}

TOOL_SPECS: list[dict] = [
    {"name": "assess_image_quality",
     "description": "OpenCV quality gate on the current working image: luminance, contrast, sharpness "
                    "(tile Laplacian variance), specular glare. Returns issues and a verdict ok|enhance|unusable.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "enhance_image",
     "description": "Low-light recovery (non-local-means denoise, gamma, CLAHE on L*). Replaces the working "
                    "image and returns the post-enhancement quality report.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "detect_hazards",
     "description": "Segment non-floor regions (k-means floor model in Lab + robust illumination fit) and "
                    "classify candidates as cable|clutter|spill. Higher sensitivity finds fainter regions "
                    "(e.g. clear liquids) at the cost of more false candidates.",
     "input_schema": {"type": "object", "properties": {
         "sensitivity": {"type": "number", "minimum": 0, "maximum": 1}}, "required": ["sensitivity"]}},
    {"name": "inspect_region",
     "description": "Active perception: zoom into one candidate, build a local floor reference and test "
                    "shadow vs physical object. Confirms or rejects the candidate.",
     "input_schema": {"type": "object", "properties": {"hazard_id": {"type": "string"}},
                      "required": ["hazard_id"]}},
    {"name": "create_work_order",
     "description": "File a routine (non-urgent) tidy-up work order for confirmed LOW/MEDIUM severity hazards.",
     "input_schema": {"type": "object", "properties": {
         "hazard_ids": {"type": "array", "items": {"type": "string"}},
         "note": {"type": "string"}}, "required": ["hazard_ids", "note"]}},
    {"name": "request_human_approval",
     "description": "Queue an urgent action (alert on-shift staff to remove a trip/slip hazard) for HIGH "
                    "severity hazards. Nothing is sent until a human approves it.",
     "input_schema": {"type": "object", "properties": {
         "hazard_ids": {"type": "array", "items": {"type": "string"}},
         "reason": {"type": "string"}}, "required": ["hazard_ids", "reason"]}},
    {"name": "request_recapture",
     "description": "Ask the user to retake the photo (with guidance) when the image cannot be trusted.",
     "input_schema": {"type": "object", "properties": {"guidance": {"type": "string"}},
                      "required": ["guidance"]}},
    {"name": "log_clear",
     "description": "Record that the walkway is clear. Refused if any confirmed hazard exists.",
     "input_schema": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}},
    {"name": "finish",
     "description": "End the inspection with a one-paragraph summary for the user.",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]
TOOL_NAMES = {t["name"] for t in TOOL_SPECS}
TERMINAL = {"create_work_order", "request_human_approval", "request_recapture", "log_clear"}


def severity(det: dict) -> str:
    if det["type"] in HIGH_TYPES:
        return "high"
    return "medium" if det.get("in_walk_path") else "low"


@dataclass
class RunState:
    run_id: str
    original: np.ndarray
    working: np.ndarray
    location: str = ""
    quality: dict | None = None
    post_quality: dict | None = None
    enhanced: bool = False
    detect_calls: list = field(default_factory=list)
    detections: dict = field(default_factory=dict)   # id -> det (with status)
    inspected: set = field(default_factory=set)
    actions: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    summary: str = ""
    finished: bool = False
    planner_used: list = field(default_factory=list)

    @property
    def confirmed(self) -> list[dict]:
        return [d for d in self.detections.values() if d["status"] == "confirmed"]

    @property
    def outcome(self) -> str:
        kinds = {a["kind"] for a in self.actions}
        for k, o in (("recapture", "recapture"), ("approval", "escalate"), ("work_order", "work_order")):
            if k in kinds:
                return o
        return "clear"


class Toolbox:
    """Executes tool calls against a RunState and enforces policy/guardrails."""

    def __init__(self, state: RunState, store: ActionStore):
        self.s = state
        self.store = store

    # -- dispatch -------------------------------------------------------
    def call(self, name: str, args: dict, planner: str, rationale: str = "") -> dict:
        t0 = time.perf_counter()
        guard = None
        if name not in TOOL_NAMES:
            result = {"error": f"unknown tool {name}"}
        elif len(self.s.trace) >= MAX_STEPS and name != "finish":
            result = {"error": "step budget exhausted; call finish"}
        else:
            try:
                result, guard = getattr(self, f"_t_{name}")(args or {})
            except Exception as exc:  # tool failure is observable, never fatal
                log.exception("tool %s failed", name)
                result = {"error": f"{type(exc).__name__}: {exc}"}
        step = {"i": len(self.s.trace) + 1, "planner": planner, "tool": name, "args": args or {},
                "rationale": rationale, "result": result, "ms": round((time.perf_counter() - t0) * 1000, 1)}
        if guard:
            step["guardrail"] = guard
            log.warning(json.dumps({"event": "guardrail", "run_id": self.s.run_id, "tool": name, "detail": guard}))
        self.s.trace.append(step)
        log.info(json.dumps({"event": "tool_call", "run_id": self.s.run_id, "step": step["i"], "tool": name,
                             "planner": planner, "ms": step["ms"]}))
        return result

    # -- perception tools ----------------------------------------------
    def _t_assess_image_quality(self, a):
        q = V.assess_quality(self.s.working)
        if self.s.enhanced:
            self.s.post_quality = q
        else:
            self.s.quality = q
        return q, None

    def _t_enhance_image(self, a):
        if self.s.enhanced:
            return {"error": "already enhanced"}, None
        self.s.working, params = V.enhance(self.s.original)
        self.s.enhanced = True
        self.s.post_quality = V.assess_quality(self.s.working)
        return {"params": params, "post_quality": self.s.post_quality}, None

    def _t_detect_hazards(self, a):
        sens = float(a.get("sensitivity", 0.5))
        r = V.detect_hazards(self.s.working, sens)
        self.s.detect_calls.append(sens)
        new = []
        for d in r["detections"]:
            # merge with existing candidates (re-detections at other sensitivities)
            if any(V.iou(d["bbox"], e["bbox"]) > 0.3 for e in self.s.detections.values()):
                continue
            d = dict(d, id=f"h{len(self.s.detections) + 1}", status="candidate", severity=severity(d))
            self.s.detections[d["id"]] = d
            new.append(d)
        brief = [{k: d[k] for k in ("id", "type", "bbox", "confidence", "in_walk_path", "severity")} for d in new]
        return {"new_candidates": brief, "rejected_patterns": r["rejected"], "threshold": r["threshold"],
                "sensitivity": sens}, None

    def _t_inspect_region(self, a):
        hid = a.get("hazard_id")
        d = self.s.detections.get(hid)
        if d is None:
            return {"error": f"no candidate {hid}"}, None
        if len(self.s.inspected) >= MAX_INSPECTIONS:
            return {"error": "inspection budget exhausted"}, None
        r = V.inspect_region(self.s.working, d["bbox"])
        self.s.inspected.add(hid)
        d["status"] = "confirmed" if r["confirmed"] else "rejected"
        d["inspection"] = r
        return {"hazard_id": hid, "type": d["type"], "severity": d["severity"], **r}, None

    # -- action tools (policy enforced here, not in the planner) -------
    def _resolve(self, ids) -> list[dict]:
        return [self.s.detections[i] for i in ids or [] if i in self.s.detections
                and self.s.detections[i]["status"] == "confirmed"]

    def _t_create_work_order(self, a):
        dets = self._resolve(a.get("hazard_ids"))
        if not dets:
            return {"error": "work orders need confirmed hazard ids (inspect_region first)"}, None
        high = [d for d in dets if d["severity"] == "high"]
        guard = None
        if high:
            guard = "high-severity hazards cannot be filed as routine work orders; routed to human approval"
            self._approval(high, "auto-routed by guardrail: " + a.get("note", ""))
            dets = [d for d in dets if d["severity"] != "high"]
            if not dets:
                return {"status": "rerouted_to_approval"}, guard
        wo = self.store.create_work_order(self.s.run_id, dets, a.get("note", ""), self.s.location)
        self.s.actions.append({"kind": "work_order", "id": wo["id"], "hazard_ids": [d["id"] for d in dets]})
        return {"status": "created", "work_order": wo["id"]}, guard

    def _approval(self, dets, reason):
        ap = self.store.request_approval(self.s.run_id, dets, reason, self.s.location)
        self.s.actions.append({"kind": "approval", "id": ap["id"], "hazard_ids": [d["id"] for d in dets]})
        return ap

    def _t_request_human_approval(self, a):
        dets = self._resolve(a.get("hazard_ids"))
        if not dets:
            return {"error": "approval requests need confirmed hazard ids"}, None
        ap = self._approval(dets, a.get("reason", ""))
        return {"status": "pending_human_approval", "approval_id": ap["id"]}, None

    def _t_request_recapture(self, a):
        self.s.actions.append({"kind": "recapture", "guidance": a.get("guidance", "")})
        return {"status": "recapture_requested"}, None

    def _t_log_clear(self, a):
        if self.s.confirmed:
            return {"error": "refused: confirmed hazards exist"}, "log_clear refused with confirmed hazards"
        self.s.actions.append({"kind": "clear", "note": a.get("note", "")})
        return {"status": "logged_clear"}, None

    def _t_finish(self, a):
        guard = self._post_run_guardrails()
        self.s.summary = a.get("summary", "")
        self.s.finished = True
        return {"status": "finished", "outcome": self.s.outcome}, guard

    def _post_run_guardrails(self) -> str | None:
        """Safety net: every confirmed HIGH hazard must be in front of a human."""
        if any(a["kind"] == "recapture" for a in self.s.actions):
            return None
        covered = {i for a in self.s.actions if a["kind"] == "approval" for i in a["hazard_ids"]}
        missing = [d for d in self.s.confirmed if d["severity"] == "high" and d["id"] not in covered]
        if missing:
            self._approval(missing, "auto-escalated by guardrail at finish")
            return f"guardrail escalated {[d['id'] for d in missing]} to human approval"
        return None


# ====================================================================== planners

class RulePlanner:
    """Deterministic policy. Every branch depends on an OpenCV observation."""
    name = "rule"

    def __init__(self, high_sensitivity_on_glare: bool = True, use_inspection: bool = True,
                 use_quality_gate: bool = True):
        self.glare_rerun = high_sensitivity_on_glare
        self.inspect = use_inspection
        self.quality_gate = use_quality_gate

    def next(self, s: RunState) -> tuple[str, dict, str]:
        terminal_done = any(a["kind"] in ("recapture", "clear", "work_order", "approval") for a in s.actions)
        if terminal_done:
            return "finish", {"summary": summarise(s)}, "action taken; closing run"
        if self.quality_gate:
            if s.quality is None:
                return "assess_image_quality", {}, "always gate on image quality first"
            if s.quality["verdict"] == "unusable":
                return "request_recapture", {"guidance": guidance(s.quality)}, \
                    f"quality issues {s.quality['issues']} cannot be fixed by enhancement"
            if s.quality["verdict"] == "enhance" and not s.enhanced:
                return "enhance_image", {}, f"quality issues {s.quality['issues']} are recoverable"
            if s.enhanced and s.post_quality and "dark" in s.post_quality["issues"]:
                return "request_recapture", {"guidance": "Still too dark after enhancement: switch on the "
                                                         "lights or use the torch."}, "enhancement failed"
        if not s.detect_calls:
            return "detect_hazards", {"sensitivity": 0.5}, "image usable; run standard detection"
        if self.glare_rerun and s.enhanced and 1.0 not in s.detect_calls:
            return "detect_hazards", {"sensitivity": 1.0}, \
                "low-light image was enhanced (contrast compressed by noise): re-detect at maximum sensitivity"
        if (self.glare_rerun and s.quality and s.quality["glare_fraction"] > 0 and 0.9 not in s.detect_calls
                and not any(d["type"] == "spill" and d["status"] != "rejected" for d in s.detections.values())):
            return "detect_hazards", {"sensitivity": 0.9}, \
                "specular highlights seen but no spill candidate: re-detect at high sensitivity for clear liquid"
        if self.inspect:
            for d in sorted(s.detections.values(), key=lambda x: x["confidence"]):
                if d["status"] == "candidate" and len(s.inspected) < MAX_INSPECTIONS:
                    return "inspect_region", {"hazard_id": d["id"]}, \
                        f"verify {d['type']} candidate (conf {d['confidence']}) before acting"
        else:
            for d in s.detections.values():
                if d["status"] == "candidate":
                    d["status"] = "confirmed"
        high = [d["id"] for d in s.confirmed if d["severity"] == "high"]
        other = [d["id"] for d in s.confirmed if d["severity"] != "high"]
        if high:
            return "request_human_approval", {"hazard_ids": high, "reason": "confirmed trip/slip hazard"}, \
                "high severity (cable/spill) needs a human-approved urgent alert"
        if other:
            return "create_work_order", {"hazard_ids": other, "note": "clutter on walkway"}, \
                "medium/low severity -> routine work order"
        return "log_clear", {"note": "no confirmed hazards"}, "all candidates rejected or none found"


def guidance(q: dict) -> str:
    tips = {"blurry": "hold the phone still and tap to focus", "overexposed": "avoid pointing at the light / "
            "turn off flash", "dark": "turn on the lights", "low_contrast": "step back to include more floor"}
    return "Please retake the photo: " + "; ".join(tips[i] for i in q["issues"] if i in tips) + "."


def summarise(s: RunState) -> str:
    c = s.confirmed
    rej = [d for d in s.detections.values() if d["status"] == "rejected"]
    if any(a["kind"] == "recapture" for a in s.actions):
        return "The photo could not be analysed reliably, so I asked for a retake."
    if not c:
        extra = f" {len(rej)} candidate(s) were checked and dismissed (e.g. shadows)." if rej else ""
        return "Walkway looks clear." + extra
    parts = ", ".join(f"{d['type']} ({d['severity']})" for d in c)
    return f"Found {len(c)} hazard(s): {parts}. High-severity items are waiting for human approval."


SYSTEM_PROMPT = """You are HazardLens, a trip-and-slip hazard inspection agent for care homes and \
homes of older adults. You see the floor ONLY through OpenCV tools. Work step by step:
1. Call assess_image_quality first. If verdict is 'unusable' call request_recapture. If 'enhance', \
call enhance_image; if the post-enhancement report still says 'dark', request_recapture.
2. Call detect_hazards (start at sensitivity 0.5). If the quality report shows glare_fraction > 0 and \
no spill candidate was found, call detect_hazards again at 0.9 (clear liquids are faint). If the \
image was enhanced for low light, also call detect_hazards at 1.0 (noise compresses contrast).
3. Call inspect_region on every candidate before acting on it; rejected candidates (e.g. shadows) must \
be ignored.
4. Confirmed cable/spill = HIGH severity -> request_human_approval. Confirmed clutter -> create_work_order. \
No confirmed hazards -> log_clear.
5. Call finish with a short, plain-English summary for a care worker.
Never claim a hazard that was not confirmed. Never alert anyone directly: humans approve urgent actions."""


class LLMPlanner:
    """Claude tool-use planner (Anthropic API or Amazon Bedrock)."""
    name = "llm"

    def __init__(self, client: Any, model: str, max_tokens: int = 1024):
        self.client, self.model, self.max_tokens = client, model, max_tokens

    def run(self, tb: Toolbox, s: RunState) -> None:
        msgs: list[dict] = [{"role": "user", "content":
                             f"Inspect the floor photo for location '{s.location or 'unspecified'}'. "
                             f"Image size {s.original.shape[1]}x{s.original.shape[0]}."}]
        while not s.finished and len(s.trace) < MAX_STEPS:
            resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                               system=SYSTEM_PROMPT, tools=TOOL_SPECS, messages=msgs)
            msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in resp.content]})
            uses = [b for b in resp.content if b.type == "tool_use"]
            text = " ".join(b.text for b in resp.content if b.type == "text").strip()
            if not uses:
                tb.call("finish", {"summary": text or summarise(s)}, self.name, "model ended turn")
                break
            results = []
            for u in uses:
                r = tb.call(u.name, dict(u.input), self.name, text[:300])
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": json.dumps(r)[:4000]})
                if s.finished:
                    break
            msgs.append({"role": "user", "content": results})


def make_llm_planner() -> LLMPlanner | None:
    provider = os.getenv("LLM_PROVIDER", "auto").lower()
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return None
    if provider == "bedrock":
        model = os.getenv("BEDROCK_MODEL_ID")
        if not model:
            return None
        return LLMPlanner(anthropic.AnthropicBedrock(aws_region=os.getenv("AWS_REGION", "eu-west-2")), model)
    if provider in ("auto", "anthropic") and os.getenv("ANTHROPIC_API_KEY"):
        return LLMPlanner(anthropic.Anthropic(), os.getenv("HAZARDLENS_MODEL", "claude-sonnet-5-5"))
    return None


# ====================================================================== entrypoint

def run_inspection(img: np.ndarray, store: ActionStore, location: str = "", planner: str = "auto",
                   rule_planner: RulePlanner | None = None) -> RunState:
    if img is None or img.ndim != 3:
        raise ValueError("expected a BGR colour image")
    s = RunState(run_id=uuid.uuid4().hex[:12], original=img, working=img.copy(), location=location[:120])
    tb = Toolbox(s, store)
    llm = make_llm_planner() if planner in ("auto", "llm") else None
    if llm is not None:
        s.planner_used.append("llm")
        try:
            llm.run(tb, s)
        except Exception as exc:
            log.warning("LLM planner failed (%s); falling back to rule planner", exc)
            s.trace.append({"i": len(s.trace) + 1, "planner": "system", "tool": "fallback", "args": {},
                            "rationale": f"LLM error: {type(exc).__name__}", "result": {}, "ms": 0})
    rp = rule_planner or RulePlanner()
    while not s.finished:
        if "rule" not in s.planner_used:
            s.planner_used.append("rule")
        name, args, why = rp.next(s)
        tb.call(name, args, rp.name, why)
        if len(s.trace) > MAX_STEPS + 2:  # hard stop
            tb.call("finish", {"summary": summarise(s)}, "system", "hard step limit")
    return s


def result_payload(s: RunState, include_image: bool = True) -> dict:
    dets = list(s.detections.values())
    out = {
        "run_id": s.run_id, "outcome": s.outcome, "summary": s.summary, "planners": s.planner_used,
        "opencv_version": V.opencv_version(), "quality": s.quality, "post_quality": s.post_quality,
        "enhanced": s.enhanced,
        "detections": [{k: v for k, v in d.items() if k != "features"} for d in dets],
        "actions": s.actions, "trace": s.trace,
    }
    if include_image:
        ann = V.annotate(s.working, dets)
        ok, buf = cv.imencode(".jpg", ann, [cv.IMWRITE_JPEG_QUALITY, 85])
        out["annotated_jpeg_b64"] = base64.b64encode(buf.tobytes()).decode() if ok else None
    return out
