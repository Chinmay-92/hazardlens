# HazardLens: agentic trip & slip hazard inspection with OpenCV 5 on AWS

Entry for the **OpenCV AI Competition 2026, powered by AWS** ([Devpost](https://opencv26.devpost.com/)).
Tracks: Overall and **Agentic Vision Award**.
Final submission deadline: **26 Oct 2026, 23:59 PT = Tue 27 Oct 2026, 06:59 UK time (GMT)**.

You photograph a walkway. An agent built on OpenCV 5 tools then works through these steps:

1. Checks whether the photo can be trusted (exposure, blur, glare).
2. Enhances it, or asks for a retake.
3. Detects candidate cables, clutter and spills.
4. Zooms into each candidate to rule out shadows.
5. Acts on what it finds. Clutter becomes a routine work order. Cables and spills become an urgent
   alert that **waits for a human to approve it** before anything is sent through Amazon SNS.

| | Single-shot OpenCV | **HazardLens agent** |
|---|---|---|
| Correct final action (60 synthetic scenes) | 73.3% | **95.0%** |
| Held-out set (seed 99) | 71.7% | **93.3%** |
| Shadow distractors handled | 40% | **100%** |
| Blurry photos sent back for a retake | 0% | **100%** |

Docs: [Technical report](docs/TECHNICAL_REPORT.md) · [Architecture](docs/architecture.md) ·
[Agent workflow](docs/agent-workflow.md) · [Eval results](eval/RESULTS.md) · [Submission text + video script](SUBMISSION.md) · [Human TODO](TODO.md)

## Quick start (local)

Needs Python 3.13. Tested on macOS with OpenCV `5.0.0` (`opencv-python-headless==5.0.0.93`).

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest                       # 19 tests
.venv/bin/python scripts/evaluate.py             # writes eval/RESULTS.md, eval/results.json, failure images
.venv/bin/uvicorn hazardlens.app:app --port 8080 # open http://localhost:8080
```

Click a demo scene, or upload or take a photo. The page shows the annotated image, each tool call
with the reason it was made, and the approval queue.

### Optional: Claude as the planner

```bash
cp .env.example .env    # then set ANTHROPIC_API_KEY (never commit it)
export ANTHROPIC_API_KEY=...            # model defaults to claude-sonnet-5-5 (HAZARDLENS_MODEL)
.venv/bin/python scripts/evaluate.py --with-llm
```

Without a key, the deterministic `RulePlanner` is used. If the LLM fails mid-run, the rule planner
takes over from the current state. The trace records which planner made each call.

## Docker

```bash
docker buildx build --platform linux/arm64 -t hazardlens:local --load .
docker run -p 8080:8080 hazardlens:local
```

## Deploy to AWS (CDK)

ECS Fargate on **Graviton (ARM64)** behind an ALB, plus an S3 trace bucket, an SNS alert topic,
Secrets Manager, CloudWatch alarms and metric filters. See `infra/hazardlens_infra/stack.py`.

```bash
cd infra
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
npm install -g aws-cdk@2.1146.1          # CDK CLI
export AWS_PROFILE=<your-profile> CDK_DEFAULT_REGION=eu-west-2
cdk bootstrap                             # once per account/region
cdk deploy                                # rule planner only (default)
# or: cdk deploy -c llm=bedrock -c bedrockModelId=<bedrock model id> -c alertEmail=you@example.com
# or: cdk deploy -c llm=anthropic  (after creating secret 'hazardlens/anthropic-api-key')
```

Outputs are `Url`, `TraceBucket`, `AlertTopic` and `ApproverTokenSecret`. Read the token with
`aws secretsmanager get-secret-value --secret-id <ApproverTokenSecret>`.
Tear down with `cdk destroy`.

## Repository layout

```
hazardlens/vision.py    OpenCV 5 tools: quality gate, enhancement, detection, active inspection
hazardlens/agent.py     executor + guardrails, RulePlanner, Claude LLMPlanner, tool schema
hazardlens/actions.py   work orders, approval queue, SNS alerts, S3 traces
hazardlens/app.py       FastAPI API + static UI (hazardlens/static/index.html)
hazardlens/synth.py     deterministic synthetic scene generator (no third-party images)
scripts/evaluate.py     benchmark: single-shot vs agent, metrics, failure images, traces
tests/                  pytest suite
infra/                  AWS CDK v2 stack (Python)
data/samples/           60 synthetic evaluation images + labels.json
eval/                   results, per-image rows, failure images, example traces
docs/                   technical report, architecture and agent workflow diagrams
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Health check, OpenCV version, whether an LLM is configured |
| POST | `/api/inspect` | multipart `file`, `location`, `planner=auto\|rule\|llm` |
| POST | `/api/inspect_sample/{scene_XXX.jpg}` | Run on a bundled demo scene |
| GET | `/api/approvals` | Approval queue, work orders, notifications |
| POST | `/api/approvals/{id}` | `{"approve": true, "reviewer": "name"}` + `X-Approver-Token` header |

## Responsible use

HazardLens supports human walkway checks; it does not replace them. Photos are processed in memory and
never stored, and no one is alerted without human approval. Limitations and failure cases are listed
in the [technical report](docs/TECHNICAL_REPORT.md#8-limitations).
