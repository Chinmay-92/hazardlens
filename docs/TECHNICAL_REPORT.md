# HazardLens: technical report

*Agentic trip-and-slip hazard inspection with OpenCV 5 on AWS.*
OpenCV AI Competition 2026 entry · solo entrant (UK) · tracks: Overall + Agentic Vision Award

---

## 1. Problem

Falls are a major cause of injury and loss of independence for older adults. The WHO lists falls
as the second leading cause of unintentional injury deaths worldwide. Many falls indoors happen because
of things on the floor: trailing cables, clutter, spilled liquid. Care homes, hospital wards and
supported-living homes already do walkway checks, but the checks are manual, rarely written down, and
they depend on someone noticing a hazard and acting on it.

A phone camera can see these hazards. The hard part is what comes after: deciding whether the photo
can be trusted, whether a dark blob is a shadow or a bag, and whether to page a busy member of staff.
HazardLens treats that as an **agentic perception-decision-action problem**, with a human kept in
control of every urgent action.

## 2. Users

| User | Need | How HazardLens helps |
|---|---|---|
| Care workers / ward staff | Quick, consistent walkway checks | One photo gives a clear outcome: clear, tidy-up work order, urgent alert pending approval, or retake |
| Duty manager | Control over who gets interrupted | Urgent alerts go only through a token-protected approval queue |
| Family carers / older adults at home | A second pair of eyes | Plain-English summary and retake guidance, accessible UI |
| Facilities / safety leads | An audit trail | JSON trace per run (S3), CloudWatch metrics for escalations and guardrail interventions |

## 3. Architecture

See [architecture.md](architecture.md) (system and sequence diagrams) and
[agent-workflow.md](agent-workflow.md) (agent loop).

- **Frontend**: one accessible HTML page (keyboard navigable, `aria-live` status, works in light and
  dark mode, opens the phone camera via `capture="environment"`).
- **API**: FastAPI (`hazardlens/app.py`). Uploads are size and pixel-limited and re-decoded with
  `cv.imdecode`, which strips EXIF/GPS. Errors return without stack traces.
- **Agent executor**: `hazardlens/agent.py` `Toolbox`. It owns state, budgets and guardrails.
- **Planners**: `RulePlanner` (deterministic, the default) and `LLMPlanner` (Claude tool use through the
  Anthropic API, model `claude-sonnet-5-5`, or Amazon Bedrock). Both use the same nine-tool schema.
- **Perception**: `hazardlens/vision.py`, built only on OpenCV 5 and NumPy.
- **Actions**: `hazardlens/actions.py`. Work orders, approval queue, SNS notification after approval,
  S3 trace persistence.

## 4. OpenCV 5 implementation

The environment is pinned to `opencv-python-headless==5.0.0.93` (OpenCV 5.0.0, `cv.__version__ == "5.0.0"`,
checked by a unit test). Wheels are available for `manylinux_2_28_aarch64`, so the same container runs on
Graviton.

| Tool | OpenCV 5 operations | Purpose |
|---|---|---|
| `assess_quality` | `cvtColor` (Lab/HSV/Gray), `Laplacian` (tile-wise variance, 90th percentile) | Measures exposure, sharpness and specular glare, then returns a verdict: ok / enhance / unusable |
| `enhance` | `fastNlMeansDenoisingColored`, `LUT` (adaptive gamma), `createCLAHE` on L* | Low-light recovery |
| `detect_hazards` | `GaussianBlur`, Lab, **robust quadratic illumination fit with `cv.solve(DECOMP_SVD)`** and MAD outlier rejection, **`kmeans` multi-modal floor model** (every cluster covering ≥18% of the frame counts as floor, which handles two-tone planks), weighted Lab distance, adaptive threshold, `morphologyEx`, `connectedComponentsWithStats`, `findContours`, `minAreaRect`, `convexHull`, `distanceTransform` (stroke thickness), `fitLine` (rejects straight full-span floor seams and grout) | Proposes candidates and classifies them as cable, clutter or spill |
| `inspect_region` | Crop + `resize` (×2 cubic), local floor reference from a ring (`dilate`/`bitwise_not`), shadow test (luminance drop, preserved hue and chroma ratio, weak boundary gradient from `Sobel`/`magnitude`/`MORPH_GRADIENT`), `Canny` + `HoughLinesP` evidence | Active perception: confirms or rejects one candidate |
| `annotate` | `rectangle`, `putText`, `imencode` | Visual output in the UI |

Design notes:
- **Why a quadratic illumination fit?** A large median or Gaussian background also absorbs big, bright
  spills. In our first version this caused 6 of 8 spills to be missed. The robust polynomial models
  vignetting and uneven lighting without absorbing local objects.
- **Why inspect each region separately?** A global detector cannot tell a soft shadow from a dark rug
  using one threshold. Zooming in and comparing the region with its immediate surroundings can.
  This is the step that removes all six shadow false positives in the benchmark.
- **Why a quality gate?** Detection on a blurred photo can still "find" hazards, as the single-shot
  baseline shows. Refusing to decide on untrustworthy pixels is a safety feature.

## 5. Agentic design

The full loop is in [agent-workflow.md](agent-workflow.md). OpenCV output changes later tool calls in
four places:

1. Quality verdict → `enhance_image` or `request_recapture` or continue.
2. Image was enhanced → a second `detect_hazards` call at sensitivity 1.0.
3. Specular glare seen but no spill candidate → `detect_hazards` at 0.9 (looking for clear liquid).
4. Each candidate → `inspect_region`. The result (confirmed or shadow-like) decides the final action:
   `request_human_approval`, `create_work_order` or `log_clear`.

**Human control.** The agent cannot alert anyone directly. HIGH-severity hazards (cables, spills) create a
*pending approval*. A person with the approver token (stored in Secrets Manager) approves it, and only then
is an SNS message published. Three executor-side guardrails apply whichever planner is driving: work orders
for HIGH hazards are re-routed to approval; `log_clear` is refused while confirmed hazards exist; `finish`
auto-escalates any HIGH hazard that has not been put in front of a human. Unit tests cover all three. One
test uses a scripted "LLM" that tries to file a cable as a routine work order.

**Failure handling.** Tool exceptions become observable `error` results and never crash the run. If the LLM
errors, the trace records a `fallback` step and the rule planner resumes from the current state (tested).
Step and inspection budgets prevent loops.

## 6. AWS deployment

AWS CDK v2 (Python), in `infra/`. The stack synthesizes locally (`cdk synth` equivalent verified). It has
**not yet been deployed** for this report; see TODO.md.

| Component | Service | Notes |
|---|---|---|
| Compute | ECS Fargate, **ARM64 (Graviton)**, 1 vCPU / 2 GB, 1 task | Container built from the repo `Dockerfile` for `linux/arm64`, non-root user, health check |
| Ingress | Application Load Balancer | `/healthz` target health check, ingress CIDR set by context |
| Network | VPC with 2 public subnets, **no NAT gateway** | Keeps hackathon running costs low |
| Traces | S3 | SSE-S3, block public access, TLS enforced, 30-day expiry, **no images stored** |
| Alerts | SNS topic | Published only after human approval; optional email subscription |
| Secrets | Secrets Manager | Generated approver token; optional existing `hazardlens/anthropic-api-key` |
| LLM | Anthropic API or Amazon Bedrock (`-c llm=bedrock -c bedrockModelId=...`), or rule-only | IAM `bedrock:InvokeModel` granted only in Bedrock mode |
| Observability | CloudWatch Logs (1-month retention), Container Insights, metric filters `Escalations` / `GuardrailInterventions`, alarms on target 5xx and unhealthy hosts → ops SNS | |
| Safety | ECS deployment circuit breaker with rollback | |

Reproducibility: pinned `requirements.txt` (compiled with `uv pip compile`), pinned CDK libraries,
deterministic synthetic data (seeded), deterministic default planner, and one command each for
`make test`, `make eval` and `make eval-heldout`.

## 7. Evaluation

**Dataset.** 60 synthetic floor photos (`hazardlens/synth.py`, seed 2026, committed in `data/samples/`).
The set covers carpet, wood and tile floors with vignetting, plus cables, clutter, spills, multi-hazard
scenes, soft-shadow distractors, low light and motion blur. Each image has ground-truth boxes and an
expected outcome. A second set (seed 99) is regenerated on demand and used as a held-out check. All
images are procedurally generated, so there are no third-party or personal images.

**Baseline.** "Single-shot": the same OpenCV detector, run once at default sensitivity, acting on every
candidate. It has no quality gate, enhancement, re-detection or inspection.

**Results (dev set, seed 2026)**. Full tables are in `eval/RESULTS.md`.

| Metric | Single-shot | **HazardLens agent** |
|---|---|---|
| Outcome accuracy (exact action) | 0.733 | **0.950** |
| Escalation recall (cable/spill reach a human) | 0.778 | **0.889** |
| False escalation rate | 0.000 | 0.000 |
| Recapture recall (blurry photos) | 0.000 | **1.000** |
| Detection precision / recall @IoU 0.3 | 0.787 / 0.822 | **0.952 / 0.889** |
| Shadow-distractor scenes correct | 0.40 | **1.00** |
| Low-light scenes correct | 0.50 | **0.83** |
| Mean tool calls per run | 3 | 5.1 |
| Latency per run, laptop CPU (mean) | ~30–50 ms | ~55–95 ms |

**Held-out (seed 99):** single-shot 0.717 → agent **0.933** outcome accuracy, escalation recall
0.778 → 0.852, false escalation 0.034 → 0.000 (`eval/heldout/RESULTS.md`).

**Failure cases** (annotated images in `eval/failures/agentic/`):
- *Clear liquid on light tiles* (dev `scene_028`, `scene_039`, `scene_048`): a whitish spill on near-white
  tile barely changes the colour. The glare-triggered re-detection catches some cases but not these.
- *Cable touching clutter* (held-out `scene_034`, `scene_038`): the two merge into one component, which
  is classified as clutter. The result is a work order, not an escalation.
- *Very dark cable on dark carpet* (held-out `scene_046`): still missed after enhancement and re-detection.

**Agentic-specific evaluation.** Task success = exact outcome above. Failure handling and guardrails are
covered by `tests/test_hazardlens.py` (19 tests: LLM fallback, three guardrails, approval token, bad
input, path traversal). Observability: every run has a step-by-step trace with timings (`eval/traces/*.json`).

**Honesty notes.** The thresholds were tuned on the dev set. The held-out set was looked at once, and that
led to one policy change (low-light re-detection at sensitivity 1.0). After that change, held-out accuracy
rose from 0.917 to 0.933. The LLM planner was tested with a scripted fake client but has **not** been
benchmarked against the live API in this report (`python scripts/evaluate.py --with-llm` does this when a
key is available). Synthetic results are **not** evidence of real-world accuracy.

## 8. Limitations

- Synthetic-only evaluation. Real floors (patterned carpets, rugs, reflections, feet in frame) will be
  harder. A small consented real-photo pilot is the obvious next step.
- Classical CV with a single-photo, top-down assumption. There is no depth, so the agent cannot judge
  how much something sticks up from the floor (trip height). A wide-angle corridor view is out of scope.
- Spill detection relies on brightness and glare. Clear water on light floors is a known blind spot.
- The approval queue lives in memory, in one task. A restart loses pending items (S3 traces remain).
  Production would use DynamoDB.
- HTTP only until a domain and ACM certificate are added. The approver token is a shared secret, not
  per-user identity (Cognito would fix this).

## 9. Responsible use

- **Assistive, not a safety system.** HazardLens supports human walkway checks. It does not replace
  them, and a "clear" result is not a guarantee. The UI and summaries say what was checked.
- **Human in the loop.** No person is contacted without explicit human approval. Every approval or
  rejection is recorded with the reviewer's name.
- **Privacy.** Images are processed in memory and not stored. EXIF/GPS is stripped. Only JSON traces
  (boxes, metrics, decisions) go to S3, with 30-day expiry. The tool is meant for floors, not people.
  There is no face or person analysis, and staff should avoid photographing residents.
- **Bias and failure modes are published**, including the classes of floor and hazard where it fails.
- **Data rights.** All evaluation imagery is generated by this project's code.
- **Security.** Non-root container, private encrypted bucket, least-privilege IAM grants, Secrets Manager,
  upload limits, no stack traces returned to clients.

## 10. Reproduce

```bash
make setup         # Python 3.13 venv + pinned deps
make test          # 19 pytest tests
make eval          # dev-set metrics → eval/RESULTS.md
make eval-heldout  # held-out metrics → eval/heldout/RESULTS.md
make run           # http://localhost:8080
cd infra && cdk synth   # CloudFormation for the AWS stack
```
