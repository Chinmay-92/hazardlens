# HazardLens demo video: narration

First-person script, one block per segment. The synthetic voice (macOS `say -v Daniel`) reads exactly this text;
you can re-record any block in your own voice and keep the same timing (each block has ~0.5 s lead-in and ~0.8 s tail).
Times are for `hazardlens_demo_body.mp4`; add the length of your intro clip for the final video.

## 0. Entrant intro (you, on camera, 15–20 s, before the body)

> Hi, I'm <your name>, a solo developer from <where you are>. This is HazardLens, an OpenCV 5 agent that looks for
> trip and slip hazards, the kind that cause falls in care homes, and it asks a human before raising an alarm.

## 1. Title card  (0:00.0–0:15.4)

> This is HazardLens: an OpenCV 5 agent that finds trip and slip hazards, checks its own evidence, and asks a person before raising an alert. I built it for the OpenCV AI Competition 2026.

## 2. App home page and demo scenes  (0:15.4–0:40.1)

> Falls are one of the main ways older people get hurt, and many indoor falls start with something on the floor: a trailing cable, a bag in a corridor, a spilled drink. Care homes check walkways by hand, and the checks are rarely recorded. A phone can see a cable. The hard part is deciding whether to trust the photo, whether a dark patch is just a shadow, and whether to interrupt a busy nurse.

## 3. Upload a photo, spill detected, zoom verification, approval requested  (0:40.1–1:10.9)

> Let me upload a photo of a carpeted corridor, add the location, and press Inspect. The agent first checks exposure, sharpness and glare. The photo is fine, so it runs detection and finds a spill candidate. Instead of acting straight away, it calls inspect region. It zooms in two times, compares the patch with a ring of nearby floor, and confirms the spill. Spills are high severity, so the agent requests human approval. Nothing has been sent yet.

## 4. Shadow scene: candidate rejected after zoom, logged clear  (1:10.9–1:27.2)

> Now a scene with a soft shadow. Detection finds a candidate that a single-shot pipeline would turn into a work order. The agent zooms in, sees that the hue is unchanged and the edges are soft, and rejects it as a shadow. The walkway is logged as clear.

## 5. Dark scene: enhance, re-detect at full sensitivity, cable confirmed, approval requested  (1:27.2–1:53.8)

> This photo is too dark. The quality gate says enhance, so the agent runs denoising, gamma and CLAHE, and finds a cable candidate. Because the image was enhanced, it then re-runs detection at maximum sensitivity, to catch anything the noise hid. That is an OpenCV output choosing the next tool. Nothing new turns up, so it confirms the cable with a zoomed inspection and queues it for approval.

## 6. Blurry scene: retake requested  (1:53.8–2:08.8)

> This one is blurry. The Laplacian sharpness score is far below the threshold, and enhancement cannot fix blur, so the agent refuses to guess. It asks for a retake, and makes no decision on pixels it cannot trust.

## 7. Clutter scene: routine work order  (2:08.8–2:21.0)

> Clutter, like a bag or a box left in a corridor, is medium severity. The agent confirms it and files a routine work order by itself. Nobody gets interrupted for this.

## 8. Human approval queue: approve the cable alert  (2:21.0–2:41.1)

> Urgent alerts are different. Here is the approval queue. As the duty manager, I review the cable alert and approve it. Only now is the alert sent to on-shift staff. Running locally it is just logged; in the cloud deployment it goes out through Amazon SNS, and approving needs a token kept in Secrets Manager.

## 9. System architecture diagram  (2:41.1–3:10.2)

> Under the hood, a FastAPI app runs on ECS Fargate on Graviton, behind a load balancer, deployed with AWS CDK. The agent executor owns the state, the budgets and the guardrails. The planner is either a deterministic rule policy or Claude through tool use. If the LLM fails, the rule policy takes over from the current state. Decision traces go to S3 without images, and CloudWatch counts escalations and guardrail interventions.

## 10. Agent workflow diagram  (3:10.2–3:33.3)

> Here is the agent loop. Every branch depends on an OpenCV 5 output: the quality verdict, whether the image was enhanced, whether glare was seen, and the zoomed inspection result. The agent can only request urgent alerts. Three guardrails make sure every high-severity hazard reaches a human, and that it never logs clear while a confirmed hazard exists.

## 11. Evaluation results: agent vs single-shot OpenCV  (3:33.3–4:01.4)

> On sixty synthetic scenes, the agent chooses the correct final action ninety-five percent of the time, against seventy-three percent for single-shot OpenCV. On a held-out set it is ninety-three versus seventy-two. It handles every shadow distractor, sends every blurry photo back for a retake, and makes no false escalations. The cost is about five tool calls and sixty milliseconds per photo on a laptop CPU.

## 12. Failure cases  (4:01.4–4:22.6)

> And here is where it fails. All three misses are on tile: clear liquid on light tiles, a spill touching a box that is reported only as clutter, and a spill in a dark photo. Escalation recall is eighty-nine percent, so three urgent hazards were missed. I publish these cases, and they are why a person still does the walk.

## 13. Closing card  (4:22.6–4:37.6)

> HazardLens assists people; it does not replace them. Photos are analysed in memory and never stored, and no one is alerted without human approval. Next step: a consented pilot with a care provider. Thanks for watching.

Body total: 4:37.6. With a 20 s intro the final video is 4:57.6 (limit 5:00).
