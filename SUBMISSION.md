# Devpost submission draft: HazardLens

> Draft only. Nothing here has been submitted. Fill in the `<...>` placeholders after deploying and recording.

**Project name:** HazardLens
**Tagline:** An OpenCV 5 agent that finds trip and slip hazards, checks its own evidence, and asks a person before raising an alert.
**Special award consideration:** Agentic Vision Award
**Repository URL:** `<judge-accessible GitHub URL or private repo with judge access>`
**Working web endpoint:** `<ALB Url output from cdk deploy>`
**Video:** `<YouTube unlisted / Vimeo URL, ≤ 5:00>`
**Built with:** opencv, python, fastapi, aws-ecs-fargate, aws-graviton, amazon-s3, amazon-sns, aws-secrets-manager, amazon-cloudwatch, aws-cdk, claude, amazon-bedrock, docker

---

## Inspiration
Falls are one of the main ways older people get hurt and lose their independence, and indoor falls are
often caused by things on the floor: a trailing charger cable, a bag left in a corridor, a spilled drink.
Care homes do walkway checks, but they are manual and seldom recorded. We wanted a phone-camera tool that
does more than label a picture. It should decide what to do next, and it should know when it cannot
trust what it sees.

## What it does
A carer photographs a walkway. An agent built on OpenCV 5 tools then:
1. **Checks the photo.** It measures exposure, sharpness and glare. Blurry photos get a retake request
   with specific advice. Dark photos are enhanced.
2. **Finds candidates.** It segments anything that isn't floor and classifies it as cable, clutter or spill.
3. **Double-checks.** It zooms into each candidate and compares it with the nearby floor to reject shadows.
4. **Acts on what it found.** Clutter becomes a routine work order. Cables and spills become an urgent alert
   that **waits for human approval** before anything is sent to staff through Amazon SNS.

The UI shows the annotated photo, every tool call with the reason for it, and the approval queue.

## How we built it
- **OpenCV 5.0** (`opencv-python-headless==5.0.0.93`) does all the perception:
  - Lab colour space, with a robust quadratic illumination fit (`cv.solve`, SVD)
  - a k-means multi-modal floor model
  - connected components, contour shape analysis (`minAreaRect`, `convexHull`, `distanceTransform`)
    and `fitLine` to reject floor seams
  - CLAHE, NL-means denoising, Laplacian sharpness
  - for zoomed inspection: Sobel/morphological-gradient edge tests and `HoughLinesP`
- **Agent:** an executor with typed tools, budgets and three guardrails. The planner is either a
  deterministic policy or **Claude** (`claude-sonnet-5-5`) via tool use, through the Anthropic API or
  Amazon Bedrock. If the LLM fails, the rule planner takes over automatically.
- **AWS (CDK v2):**
  - ECS Fargate on **Graviton/ARM64** behind an ALB
  - S3 for JSON decision traces (encrypted, private, 30-day expiry, never images)
  - SNS for human-approved alerts
  - Secrets Manager for the approver token
  - CloudWatch logs, Container Insights, alarms and metric filters for escalations and guardrail interventions

## Challenges
- A large spill can look like "background" to a median-filter illumination model. Switching to a robust
  polynomial fit took spill detection from 2 of 8 to 6 of 8 on the dev set.
- Telling soft shadows from dark objects needs a *local* comparison. That is why the agent zooms in
  instead of tightening one global threshold.
- Deciding where the LLM's autonomy stops. The agent cannot page a person: it can only request it.

## Accomplishments
On 60 synthetic scenes, the agent picks the right final action **95.0%** of the time, against **73.3%**
for a single-shot OpenCV pipeline. On a held-out set the figures are **93.3% vs 71.7%**. It removes
every shadow false alarm, sends every blurry photo back for a retake, and has **zero false escalations**.
We also publish the cases it gets wrong.

## What we learned
In this project, the extra value of an agent came from **choosing which OpenCV tool to run next**:
re-detecting at higher sensitivity after low-light enhancement, looking harder for liquid when glare is
present, and verifying before acting. Generating language was not where the gains came from.

## What's next
- A consented real-world pilot with a care provider
- Depth from a second frame, to estimate how high a trip hazard sticks up
- DynamoDB-backed approvals and Cognito per-user identity
- A COOL/Graviton benchmark of the OpenCV workload

## Testing instructions (for the Devpost field)
1. Open `<endpoint URL>`. Click any demo scene on the left. Each one is synthetic and runs the full agent.
   Try scene 1 (shadow, dismissed), scene 2 (dark, enhanced, then a cable escalated) and scene 4 (blurry,
   retake requested).
2. Read the decision trace on the right. Each step shows the OpenCV tool, its result and the reason for the
   next call.
3. In "Human approvals", enter the approver token `<shared privately with judges>` and approve an alert.
   The alert is published to SNS and shown as "Last alert".
4. Local alternative:
   `pip install -r requirements-dev.txt && pytest && python scripts/evaluate.py && uvicorn hazardlens.app:app`
   (see README).

---

# Video script (target 4:30, hard limit 5:00)

| Time | Shot | Voice-over |
|---|---|---|
| 0:00–0:20 | Face to camera (team requirement) | "Hi, I'm `<name>`, a solo developer from the UK. This is HazardLens, an OpenCV 5 agent that looks for trip and slip hazards, the kind that cause falls in care homes, and it asks a human before raising an alarm." |
| 0:20–0:45 | Photos of a cable / spill (stock-free, own footage or the synthetic scenes) | "Walkway checks are manual and rarely recorded. A phone can see a cable. The hard part is deciding whether to trust the photo, whether that dark patch is a shadow, and whether to interrupt a busy nurse." |
| 0:45–1:45 | Live endpoint: click shadow scene → trace | "Here's a scene with a soft shadow. Quality check: fine. Detection finds a candidate. Instead of acting, the agent calls inspect_region. It zooms in, compares the patch with the nearby floor, sees the hue is unchanged and the edges are soft, and rejects it as a shadow. Result: clear. A single-shot pipeline would have filed a work order." |
| 1:45–2:30 | Dark cable scene → trace → approval | "This photo is too dark. The agent enhances it with denoising, gamma and CLAHE. Because it enhanced the image, it runs detection again at maximum sensitivity. It finds a cable, confirms it, and since cables are high severity, it *requests* approval. Nothing has been sent yet. I approve as duty manager, and the alert goes out through Amazon SNS." |
| 2:30–2:50 | Blurry scene | "Blurry photo: the agent refuses to guess and asks for a retake, with advice." |
| 2:50–3:30 | Architecture diagram (docs/architecture.md) | "Under the hood: FastAPI on ECS Fargate on Graviton, deployed with CDK. OpenCV 5 tools; a planner that's either a deterministic policy or Claude via tool use, falling back automatically; S3 traces without images; Secrets Manager; CloudWatch alarms and metrics for escalations and guardrail interventions." |
| 3:30–4:15 | eval/RESULTS.md table + a failure image | "On 60 synthetic scenes the agent picks the right action 95% of the time, against 73% for single-shot OpenCV. On held-out scenes it's 93% vs 72%, with zero false escalations. Here's where it fails: clear liquid on white tiles, and a cable touching a box. We publish those cases." |
| 4:15–4:40 | Face / closing | "HazardLens assists people; it doesn't replace them. Photos aren't stored, and no one is alerted without approval. Next step: a consented pilot with a care provider. Thanks!" |

Recording tips: 1080p, screen at 125% zoom, keep under 5:00 including titles, upload as public or unlisted.
