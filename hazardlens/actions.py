"""Action sinks: work orders, human-approval queue, notifications and trace persistence.

Local mode keeps everything in memory. On AWS, set:
  TRACE_BUCKET   -> run traces are written to s3://TRACE_BUCKET/traces/<run_id>.json (SSE-S3)
  ALERT_TOPIC_ARN -> approved urgent alerts are published to this SNS topic
Nothing is ever sent to a person without an explicit human approval.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid

log = logging.getLogger("hazardlens.actions")


def _brief(dets):
    return [{"id": d["id"], "type": d["type"], "severity": d["severity"], "bbox": d["bbox"]} for d in dets]


class ActionStore:
    def __init__(self, trace_bucket: str | None = None, alert_topic: str | None = None):
        self._lock = threading.Lock()
        self.work_orders: dict[str, dict] = {}
        self.approvals: dict[str, dict] = {}
        self.notifications: list[dict] = []
        self.trace_bucket = trace_bucket if trace_bucket is not None else os.getenv("TRACE_BUCKET")
        self.alert_topic = alert_topic if alert_topic is not None else os.getenv("ALERT_TOPIC_ARN")

    def create_work_order(self, run_id, dets, note, location):
        wo = {"id": "WO-" + uuid.uuid4().hex[:6].upper(), "run_id": run_id, "hazards": _brief(dets),
              "note": note[:300], "location": location, "created": time.time(), "status": "open"}
        with self._lock:
            self.work_orders[wo["id"]] = wo
        return wo

    def request_approval(self, run_id, dets, reason, location):
        ap = {"id": "AP-" + uuid.uuid4().hex[:6].upper(), "run_id": run_id, "hazards": _brief(dets),
              "proposed_action": "alert_on_shift_staff", "reason": reason[:300], "location": location,
              "created": time.time(), "status": "pending"}
        with self._lock:
            self.approvals[ap["id"]] = ap
        return ap

    def decide(self, approval_id: str, approve: bool, reviewer: str) -> dict:
        with self._lock:
            ap = self.approvals.get(approval_id)
            if ap is None:
                raise KeyError(approval_id)
            if ap["status"] != "pending":
                return ap
            ap["status"] = "approved" if approve else "rejected"
            ap["reviewer"] = reviewer[:60]
            ap["decided"] = time.time()
        if approve:
            self._notify(ap)
        return ap

    def _notify(self, ap):
        msg = (f"[HazardLens] Please check {ap['location'] or 'the reported area'}: "
               + ", ".join(h["type"] for h in ap["hazards"]) + f" (approval {ap['id']}).")
        record = {"approval_id": ap["id"], "message": msg, "channel": "log"}
        if self.alert_topic:
            try:
                import boto3
                boto3.client("sns").publish(TopicArn=self.alert_topic, Message=msg, Subject="HazardLens alert")
                record["channel"] = "sns"
            except Exception as exc:  # observable failure; approval still recorded
                record["channel"] = f"sns_failed:{type(exc).__name__}"
                log.error("SNS publish failed: %s", exc)
        self.notifications.append(record)
        log.info(json.dumps({"event": "notification", **record}))

    def persist_trace(self, payload: dict) -> str | None:
        if not self.trace_bucket:
            return None
        try:
            import boto3
            key = f"traces/{payload['run_id']}.json"
            body = {k: v for k, v in payload.items() if k != "annotated_jpeg_b64"}  # no imagery stored
            boto3.client("s3").put_object(Bucket=self.trace_bucket, Key=key, Body=json.dumps(body).encode(),
                                          ContentType="application/json", ServerSideEncryption="AES256")
            return f"s3://{self.trace_bucket}/{key}"
        except Exception as exc:
            log.error("trace persist failed: %s", exc)
            return None
