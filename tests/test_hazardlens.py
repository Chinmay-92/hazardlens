import io
import json
from types import SimpleNamespace

import cv2 as cv
import numpy as np
import pytest
from fastapi.testclient import TestClient

from hazardlens import vision as V
from hazardlens.actions import ActionStore
from hazardlens.agent import LLMPlanner, RulePlanner, RunState, Toolbox, run_inspection, summarise
from hazardlens.synth import make_scene


def scene(kind, seed=1):
    return make_scene("t.jpg", kind, seed)


@pytest.fixture
def store():
    return ActionStore(trace_bucket="", alert_topic="")


# ------------------------------------------------------------------ vision

def test_opencv_major_version_is_5():
    assert V.opencv_version().startswith("5.")


def test_quality_flags_dark_and_blurry():
    img, _ = scene("carpet:|dark")
    assert "dark" in V.assess_quality(img)["issues"]
    img, _ = scene("wood:|blurry")
    assert V.assess_quality(img)["verdict"] == "unusable"
    img, _ = scene("tile:")
    assert V.assess_quality(img)["verdict"] == "ok"


def test_enhance_brightens_dark_image():
    img, _ = scene("carpet:clutter|dark")
    out, params = V.enhance(img)
    assert V.assess_quality(out)["mean_luminance"] > 2 * V.assess_quality(img)["mean_luminance"]
    assert params["denoised"]


@pytest.mark.parametrize("kind,typ", [("carpet:cable", "cable"), ("wood:clutter", "clutter"),
                                      ("carpet:spill", "spill")])
def test_detects_and_classifies(kind, typ):
    img, sc = scene(kind, seed=5)
    dets = V.detect_hazards(img)["detections"]
    assert dets, "expected a candidate"
    best = max(dets, key=lambda d: V.iou(d["bbox"], sc.hazards[0]["bbox"]))
    assert V.iou(best["bbox"], sc.hazards[0]["bbox"]) > 0.3
    assert best["type"] == typ


def test_clear_floor_has_no_candidates():
    for k in ("carpet:", "wood:", "tile:"):
        img, _ = scene(k, seed=3)
        assert V.detect_hazards(img)["detections"] == []


def test_inspect_region_rejects_shadow_confirms_object():
    img, _ = scene("carpet:|shadow", seed=11)
    dets = V.detect_hazards(img)["detections"]
    for d in dets:
        assert V.inspect_region(img, d["bbox"])["confirmed"] is False
    img, sc = scene("wood:clutter", seed=4)
    assert V.inspect_region(img, sc.hazards[0]["bbox"])["confirmed"] is True


def test_iou():
    assert V.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert V.iou([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0


# ------------------------------------------------------------------ agent

def tools(s):
    return [t["tool"] for t in s.trace]


def test_agent_escalates_cable_to_human(store):
    img, _ = scene("carpet:cable", seed=2)
    s = run_inspection(img, store, planner="rule")
    assert s.outcome == "escalate"
    assert "inspect_region" in tools(s)
    assert list(store.approvals.values())[0]["status"] == "pending"
    assert store.notifications == []  # nothing sent without approval


def test_agent_shadow_is_dismissed_after_inspection(store):
    img, _ = scene("carpet:|shadow", seed=11)
    s = run_inspection(img, store, planner="rule")
    assert s.outcome == "clear"
    assert any(d["status"] == "rejected" for d in s.detections.values())


def test_vision_output_changes_tool_sequence(store):
    """The core agentic property: different visual evidence -> different tool calls."""
    dark, _ = scene("carpet:cable|dark", seed=2)
    blurry, _ = scene("carpet:cable|blurry", seed=2)
    a = run_inspection(dark, store, planner="rule")
    b = run_inspection(blurry, store, planner="rule")
    assert "enhance_image" in tools(a) and a.outcome == "escalate"
    assert tools(b)[:3] == ["assess_image_quality", "request_recapture", "finish"]


def test_guardrail_reroutes_high_severity_work_order(store):
    img, sc = scene("carpet:cable", seed=2)
    s = RunState(run_id="t", original=img, working=img.copy())
    tb = Toolbox(s, store)
    tb.call("detect_hazards", {"sensitivity": 0.5}, "test")
    hid = next(iter(s.detections))
    tb.call("inspect_region", {"hazard_id": hid}, "test")
    tb.call("create_work_order", {"hazard_ids": [hid], "note": "x"}, "test")
    assert s.trace[-1].get("guardrail")
    assert s.outcome == "escalate"


def test_log_clear_refused_with_confirmed_hazard(store):
    img, _ = scene("wood:clutter", seed=4)
    s = RunState(run_id="t", original=img, working=img.copy())
    tb = Toolbox(s, store)
    tb.call("detect_hazards", {"sensitivity": 0.5}, "test")
    tb.call("inspect_region", {"hazard_id": "h1"}, "test")
    r = tb.call("log_clear", {"note": "fine"}, "test")
    assert "error" in r


def test_finish_guardrail_escalates_unhandled_high_hazard(store):
    img, _ = scene("carpet:spill", seed=5)
    s = RunState(run_id="t", original=img, working=img.copy())
    tb = Toolbox(s, store)
    tb.call("detect_hazards", {"sensitivity": 0.5}, "test")
    for hid in list(s.detections):
        tb.call("inspect_region", {"hazard_id": hid}, "test")
    tb.call("finish", {"summary": "done"}, "test")  # LLM 'forgot' to act
    assert s.outcome == "escalate" and s.trace[-1].get("guardrail")


# ------------------------------------------------------------------ LLM planner (fake client)

class _Block(SimpleNamespace):
    def model_dump(self, exclude_none=True):
        return {k: v for k, v in vars(self).items() if v is not None}


class FakeClient:
    """Scripted Claude: calls tools, but tries to file the cable as a routine work order."""

    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0
        self.messages = self

    def create(self, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("API unavailable")
        last = kw["messages"][-1]["content"]
        state = json.loads(last[0]["content"]) if isinstance(last, list) else {}
        script = [("assess_image_quality", {}), ("detect_hazards", {"sensitivity": 0.5}),
                  ("inspect_region", {"hazard_id": "h1"}),
                  ("create_work_order", {"hazard_ids": ["h1"], "note": "cable"}),
                  ("finish", {"summary": "done"})]
        name, args = script[min(self.calls - 1, len(script) - 1)]
        return SimpleNamespace(content=[_Block(type="text", text=f"step {self.calls} {list(state)[:2]}"),
                                        _Block(type="tool_use", id=f"tu{self.calls}", name=name, input=args)])


def test_llm_planner_with_guardrail(store, monkeypatch):
    img, _ = scene("carpet:cable", seed=2)
    import hazardlens.agent as A
    monkeypatch.setattr(A, "make_llm_planner", lambda: LLMPlanner(FakeClient(), "claude-sonnet-5-5"))
    s = run_inspection(img, store, planner="llm")
    assert s.planner_used == ["llm"]
    assert s.outcome == "escalate"  # guardrail rerouted the work order
    assert any(t.get("guardrail") for t in s.trace)


def test_llm_failure_falls_back_to_rules(store, monkeypatch):
    img, _ = scene("wood:clutter", seed=4)
    import hazardlens.agent as A
    monkeypatch.setattr(A, "make_llm_planner", lambda: LLMPlanner(FakeClient(fail=True), "m"))
    s = run_inspection(img, store, planner="llm")
    assert s.planner_used == ["llm", "rule"]
    assert any(t["tool"] == "fallback" for t in s.trace)
    assert s.outcome == "work_order"


# ------------------------------------------------------------------ API

@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from hazardlens import app as appmod
    appmod.store.__init__(trace_bucket="", alert_topic="")
    return TestClient(appmod.app), appmod


def test_api_end_to_end_with_approval(client):
    c, appmod = client
    assert c.get("/healthz").json()["opencv"].startswith("5.")
    img, _ = scene("carpet:cable", seed=2)
    ok, buf = cv.imencode(".jpg", img)
    r = c.post("/api/inspect", files={"file": ("f.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
               data={"location": "Corridor B", "planner": "rule"})
    assert r.status_code == 200
    j = r.json()
    assert j["outcome"] == "escalate" and j["annotated_jpeg_b64"]
    ap = c.get("/api/approvals").json()["approvals"][0]
    assert ap["status"] == "pending"
    monkey_token = "s3cret"
    appmod.APPROVER_TOKEN = monkey_token
    try:
        assert c.post(f"/api/approvals/{ap['id']}", json={"approve": True}).status_code == 401
        r = c.post(f"/api/approvals/{ap['id']}", json={"approve": True, "reviewer": "nurse"},
                   headers={"x-approver-token": monkey_token})
        assert r.json()["status"] == "approved"
    finally:
        appmod.APPROVER_TOKEN = ""
    assert c.get("/api/approvals").json()["notifications"][-1]["channel"] == "log"


def test_api_rejects_bad_input(client):
    c, _ = client
    r = c.post("/api/inspect", files={"file": ("x.jpg", io.BytesIO(b"not an image"), "image/jpeg")})
    assert r.status_code == 415
    assert c.post("/api/inspect_sample/../../etc/passwd").status_code == 404
    assert c.post("/api/inspect_sample/scene_043.jpg?planner=rule").json()["outcome"] == "clear"


# ------------------------------------------------------------------ summary text

def _state_with(dets, actions):
    img = np.zeros((8, 8, 3), np.uint8)
    s = RunState(run_id="t", original=img, working=img)
    s.detections = {d["id"]: d for d in dets}
    s.actions = actions
    return s


def test_summary_clutter_only_mentions_work_order_not_approval():
    s = _state_with([{"id": "h1", "type": "clutter", "severity": "medium", "status": "confirmed"}],
                    [{"kind": "work_order", "id": "WO-1", "hazard_ids": ["h1"]}])
    text = summarise(s)
    assert "routine work order" in text
    assert "human approval" not in text


def test_summary_high_severity_mentions_pending_approval():
    s = _state_with([{"id": "h1", "type": "cable", "severity": "high", "status": "confirmed"}],
                    [{"kind": "approval", "id": "AP-1", "hazard_ids": ["h1"]}])
    text = summarise(s)
    assert "waiting for human approval" in text
    assert "work order" not in text


def test_clutter_scene_end_to_end_summary(store):
    img, _ = scene("carpet:clutter")
    r = run_inspection(img, store=store, planner="rule")
    assert r.outcome == "work_order"
    assert "human approval" not in r.summary
    assert "routine work order" in r.summary
