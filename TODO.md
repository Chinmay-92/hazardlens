# Human TODO: HazardLens submission

Deadline: **Mon 26 Oct 2026, 23:59 PT = Tue 27 Oct, 06:59 UK time**. UK clocks go back on 25 Oct. Aim to submit by Mon evening UK time.

## Must do
- [ ] **Register** on https://opencv26.devpost.com/ if you haven't, and accept the official rules (check the age/eligibility block; the UK is eligible).
- [ ] **Start Docker** and build locally: `docker buildx build --platform linux/arm64 -t hazardlens:local --load .`, then `docker run -p 8080:8080 hazardlens:local`. Check that http://localhost:8080 works. (Not verified here: the Docker daemon was not running.)
- [ ] **Deploy to AWS** (see README → Deploy): `cd infra && cdk bootstrap && cdk deploy` (add `-c alertEmail=you@…` and confirm the SNS email subscription). Note the `Url` output.
  - [ ] Smoke test: `curl <Url>/healthz` should show `"opencv":"5.0.0"`. Run the 3 demo scenes and approve an alert using the token from `aws secretsmanager get-secret-value --secret-id <ApproverTokenSecret>`.
  - [ ] Check that a trace JSON appears in the S3 bucket, and that the CloudWatch metric filters and logs show data.
  - [ ] Budget: set an AWS Budget alarm. ALB + 1 Fargate ARM task ≈ low £/day. Leave it running through judging (to 9 Nov), then `cdk destroy`.
- [ ] *(Optional, recommended)* Create the secret `hazardlens/anthropic-api-key`, or use Bedrock (`-c llm=bedrock -c bedrockModelId=…`). Then run `python scripts/evaluate.py --with-llm` locally and add the LLM column to `eval/RESULTS.md` and the report. Do not claim LLM numbers until you have run it.
- [ ] **Push the repo** to GitHub (public, or private with judge access) and put the URL in SUBMISSION.md.
- [ ] **Record the video** (≤ 5:00, must show you, the app working, the architecture and the results), following the script in SUBMISSION.md. Upload it as public or unlisted.
- [ ] **Zip file**: Devpost requires one. Use `git archive -o hazardlens.zip HEAD`.
- [ ] **Fill in the Devpost form** from SUBMISSION.md: story, Built with, "Special Award Consideration" = Agentic Vision Award, Repository URL, Testing instructions, Working web endpoint, video URL, images (screenshot of the UI + `docs/architecture.md` diagram rendered to PNG). Share the approver token privately or leave approvals unprotected for judges.
- [ ] Re-read the claims in README/report/SUBMISSION against `eval/RESULTS.md` after any change. Re-run `make test eval eval-heldout`.

## Nice to have (if time)
- [ ] HTTPS: add a domain + ACM certificate to the ALB (`ApplicationLoadBalancedFargateService` `certificate=` / `redirect_http=True`).
- [ ] Take 5–10 of your own floor photos (no people) and add a "real-world spot check" section with honest results.
- [ ] Render Mermaid diagrams to PNG for the Devpost gallery (e.g. mermaid.live).
- [ ] Restrict ingress with `-c allowedCidr=…` after judging if you want it private.
- [ ] COOL award: not targeted. It would need COOL from AWS Marketplace on Graviton plus a benchmark.
