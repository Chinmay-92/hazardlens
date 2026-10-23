"""AWS infrastructure for HazardLens (AWS CDK v2, Python).

ALB (HTTP) -> ECS Fargate service on ARM64/Graviton running the OpenCV 5 container.
S3 (encrypted, private, 30-day lifecycle) stores JSON run traces - never images.
SNS topic receives alerts *only after* a human approves them in the UI.
Secrets Manager holds the approver token (and optionally the Anthropic API key).
CloudWatch: structured JSON logs, Container Insights, 5xx + unhealthy-host alarms.
"""
from aws_cdk import (CfnOutput, Duration, RemovalPolicy, Stack, aws_cloudwatch as cw,
                     aws_cloudwatch_actions as cw_actions, aws_ec2 as ec2, aws_ecr_assets as ecr_assets,
                     aws_ecs as ecs, aws_ecs_patterns as ecs_patterns, aws_elasticloadbalancingv2 as elbv2, aws_iam as iam, aws_logs as logs,
                     aws_s3 as s3, aws_secretsmanager as sm, aws_sns as sns,
                     aws_sns_subscriptions as subs)
from pathlib import Path

from constructs import Construct

REPO_ROOT = str(Path(__file__).resolve().parents[2])


class HazardLensStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kw) -> None:
        super().__init__(scope, cid, **kw)
        llm = self.node.try_get_context("llm") or "rule"           # rule | anthropic | bedrock
        bedrock_model = self.node.try_get_context("bedrockModelId") or ""
        alert_email = self.node.try_get_context("alertEmail") or ""
        allowed_cidr = self.node.try_get_context("allowedCidr") or "0.0.0.0/0"

        # Public subnets only, no NAT gateway: keeps a hackathon deployment cheap.
        vpc = ec2.Vpc(self, "Vpc", max_azs=2, nat_gateways=0, subnet_configuration=[
            ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24)])

        traces = s3.Bucket(self, "Traces", encryption=s3.BucketEncryption.S3_MANAGED,
                           block_public_access=s3.BlockPublicAccess.BLOCK_ALL, enforce_ssl=True,
                           versioned=False, lifecycle_rules=[s3.LifecycleRule(expiration=Duration.days(30))],
                           removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)

        topic = sns.Topic(self, "Alerts", display_name="HazardLens approved alerts")
        if alert_email:
            topic.add_subscription(subs.EmailSubscription(alert_email))

        approver = sm.Secret(self, "ApproverToken", description="HazardLens approver token",
                             generate_secret_string=sm.SecretStringGenerator(exclude_punctuation=True,
                                                                             password_length=24))

        cluster = ecs.Cluster(self, "Cluster", vpc=vpc, container_insights_v2=ecs.ContainerInsights.ENABLED)
        log_group = logs.LogGroup(self, "Logs", retention=logs.RetentionDays.ONE_MONTH,
                                  removal_policy=RemovalPolicy.DESTROY)

        env = {"TRACE_BUCKET": traces.bucket_name, "ALERT_TOPIC_ARN": topic.topic_arn,
               "LLM_PROVIDER": llm, "LOG_LEVEL": "INFO", "AWS_REGION": self.region}
        secrets = {"APPROVER_TOKEN": ecs.Secret.from_secrets_manager(approver)}
        if llm == "bedrock":
            env["BEDROCK_MODEL_ID"] = bedrock_model
        if llm == "anthropic":
            key = sm.Secret.from_secret_name_v2(self, "AnthropicKey", "hazardlens/anthropic-api-key")
            secrets["ANTHROPIC_API_KEY"] = ecs.Secret.from_secrets_manager(key)

        svc = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "Service", cluster=cluster, cpu=1024, memory_limit_mib=2048, desired_count=1,
            public_load_balancer=True, assign_public_ip=True,
            task_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            runtime_platform=ecs.RuntimePlatform(cpu_architecture=ecs.CpuArchitecture.ARM64,
                                                 operating_system_family=ecs.OperatingSystemFamily.LINUX),
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_asset(REPO_ROOT, platform=ecr_assets.Platform.LINUX_ARM64),
                container_port=8080, environment=env, secrets=secrets,
                log_driver=ecs.LogDrivers.aws_logs(stream_prefix="hazardlens", log_group=log_group)),
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True), open_listener=False)
        svc.load_balancer.connections.allow_from(ec2.Peer.ipv4(allowed_cidr), ec2.Port.tcp(80))
        svc.target_group.configure_health_check(path="/healthz", healthy_http_codes="200",
                                                interval=Duration.seconds(30))

        traces.grant_put(svc.task_definition.task_role)
        topic.grant_publish(svc.task_definition.task_role)
        if llm == "bedrock":
            svc.task_definition.task_role.add_to_principal_policy(iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"], resources=["*"]))

        # Observability: alarms -> same SNS topic (ops), metric filter for guardrail interventions
        ops = sns.Topic(self, "OpsAlarms", display_name="HazardLens ops alarms")
        if alert_email:
            ops.add_subscription(subs.EmailSubscription(alert_email))
        for name, metric in {
            "Target5xx": svc.load_balancer.metrics.http_code_target(
                code=elbv2.HttpCodeTarget.TARGET_5XX_COUNT,
                period=Duration.minutes(5)),
            "UnhealthyHosts": svc.target_group.metrics.unhealthy_host_count(period=Duration.minutes(5)),
        }.items():
            alarm = cw.Alarm(self, name, metric=metric, threshold=1, evaluation_periods=1,
                             treat_missing_data=cw.TreatMissingData.NOT_BREACHING)
            alarm.add_alarm_action(cw_actions.SnsAction(ops))
        logs.MetricFilter(self, "GuardrailInterventions", log_group=log_group,
                          metric_namespace="HazardLens", metric_name="GuardrailInterventions",
                          filter_pattern=logs.FilterPattern.literal('{ $.event = "guardrail" }'),
                          metric_value="1")
        logs.MetricFilter(self, "Escalations", log_group=log_group, metric_namespace="HazardLens",
                          metric_name="Escalations",
                          filter_pattern=logs.FilterPattern.literal('{ $.event = "run" && $.outcome = "escalate" }'),
                          metric_value="1")

        CfnOutput(self, "Url", value=f"http://{svc.load_balancer.load_balancer_dns_name}")
        CfnOutput(self, "TraceBucket", value=traces.bucket_name)
        CfnOutput(self, "AlertTopic", value=topic.topic_arn)
        CfnOutput(self, "ApproverTokenSecret", value=approver.secret_name)
