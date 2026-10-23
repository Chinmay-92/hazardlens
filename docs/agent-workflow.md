# HazardLens agent workflow (perception → decision → action)

Every decision node below branches on an **OpenCV 5 output**. That is what qualifies HazardLens
for the Agentic Vision path: the visual evidence decides which tool runs next and which action
(or human request) follows.

```mermaid
flowchart TD
    A([Photo received]) --> Q["assess_image_quality<br/>L* mean, tile Laplacian variance,<br/>specular glare fraction"]
    Q -->|"verdict = unusable<br/>(blur / blown highlights)"| R["request_recapture<br/>with specific guidance"]
    Q -->|"verdict = enhance (dark)"| E["enhance_image<br/>NL-means denoise → gamma → CLAHE"]
    E -->|"still dark"| R
    E --> D1
    Q -->|"verdict = ok"| D1["detect_hazards(0.5)<br/>Lab k-means floor model + robust<br/>illumination fit → components → cable/clutter/spill"]
    D1 -->|"image was enhanced"| D2["detect_hazards(1.0)<br/>recover low-light contrast"]
    D1 -->|"glare seen but no spill candidate"| D3["detect_hazards(0.9)<br/>look harder for clear liquid"]
    D2 --> I
    D3 --> I
    D1 --> I{"Any unverified<br/>candidate?"}
    I -->|yes| IR["inspect_region(id)<br/>zoom ×2, local floor ring,<br/>shadow test, Sobel edges, HoughLinesP"]
    IR -->|"shadow_like / no contrast"| X["candidate rejected"] --> I
    IR -->|"confirmed"| C["candidate confirmed"] --> I
    I -->|no| S{"Severity of<br/>confirmed hazards"}
    S -->|"cable or spill = HIGH"| H["request_human_approval<br/>(nothing sent yet)"]
    S -->|"clutter = MEDIUM/LOW"| W["create_work_order"]
    S -->|"none confirmed"| L["log_clear"]
    H --> F["finish + post-run guardrail<br/>(every HIGH hazard must reach a human)"]
    W --> F
    L --> F
    R --> F
    H -.-> HU{{"Human approver<br/>(token-protected)"}}
    HU -->|approve| N["SNS alert to on-shift staff"]
    HU -->|reject| NR["closed, no alert"]
```

## Planners and control

| Concern | Mechanism |
|---|---|
| Who decides | `RulePlanner` (deterministic, default) or `LLMPlanner` (Claude `claude-sonnet-5-5` via Anthropic API, or Bedrock). Both call the *same* tool schema. |
| LLM failure | Any API error → logged `fallback` step → `RulePlanner` resumes from the current state. |
| Autonomy limits | Max 14 tool calls, max 6 inspections. Agent can only *request* urgent alerts. |
| Guardrail 1 | `create_work_order` with a HIGH hazard is re-routed to human approval. |
| Guardrail 2 | `log_clear` is refused while confirmed hazards exist. |
| Guardrail 3 | On `finish`, any confirmed HIGH hazard not in front of a human is auto-escalated. |
| Human control | `/api/approvals/{id}` needs `X-Approver-Token` (Secrets Manager) when configured. Only approved items reach SNS. |
| Observability | Each tool call → JSON log line (`run_id`, tool, planner, ms) + full trace in the response and in S3. CloudWatch metric filters count `Escalations` and `GuardrailInterventions`. |

## Example traces (from `eval/traces/`)

**Shadow dismissed (`scene_043`)**: quality ok → `detect_hazards(0.5)` finds a "clutter" candidate (conf 0.26)
→ `inspect_region(h1)` says `shadow_like` → `log_clear`. The single-shot pipeline files a needless work order.

**Low light (`scene_046`)**: quality `dark` → `enhance_image` → `detect_hazards(0.5)` → `detect_hazards(1.0)`
(because the image was enhanced) → `inspect_region` confirms a cable → `request_human_approval`.

**Blurry (`scene_052`)**: quality `unusable` (sharpness 7.5 < 40) → `request_recapture` with "hold the phone
still" guidance. No detection is attempted, so no decision is made on untrustworthy pixels.
