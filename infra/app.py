#!/usr/bin/env python3
import os

import aws_cdk as cdk

from hazardlens_infra.stack import HazardLensStack

app = cdk.App()
HazardLensStack(
    app, "HazardLens",
    env=cdk.Environment(account=os.getenv("CDK_DEFAULT_ACCOUNT"),
                        region=os.getenv("CDK_DEFAULT_REGION", "eu-west-2")),
    description="HazardLens - agentic trip & slip hazard inspection (OpenCV 5) on ECS Fargate Graviton",
)
cdk.Tags.of(app).add("project", "hazardlens")
cdk.Tags.of(app).add("event", "opencv-ai-competition-2026")
app.synth()
