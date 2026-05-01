import os
from aws_cdk import (
    Stack,
    aws_ecs as ecs,
    aws_ec2 as ec2,
    aws_logs as logs,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_ecr as ecr,
    RemovalPolicy
)

from constructs import Construct

class MentalHealthStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs): 
        super().__init__(scope, construct_id, **kwargs)

        # Create Virtual Private Cloud (VPC) for ECS cluster - isolated network in AWS
        vpc = ec2.Vpc(
            self, "MentalHealthVPC",
            max_azs=2, # Use 2 availability zones (groups of data centers)
            nat_gateways=0, # Use public subnets only to save costs (containers get public IPs)
        )

        # Create ECS cluster - manages which containers run
        cluster = ecs.Cluster(
            self, "MentalHealthCluster",
            vpc=vpc, # Use VPC from above
            cluster_name="mental-health-cluster"
        )

        # Create ECR repository for container images
        ecr_repository = ecr.Repository(
            self, "MentalHealthECR",
            repository_name="mental-health-pipeline",
            removal_policy=RemovalPolicy.DESTROY,  # Delete repo when stack is deleted
            image_scan_on_push=True,  # Scan images for vulnerabilities
            lifecycle_rules=[
                ecr.LifecycleRule(
                    max_image_count=5,  # Keep only 5 most recent images
                    description="Keep only 5 recent images"
                )
            ]
        )

        # Create CloudWatch log group - where container output goes
        log_group = logs.LogGroup(
            self, "MentalHealthLogGroup",
            log_group_name='/ecs/mental-health', # Path in CloudWatch logs
            retention=logs.RetentionDays.ONE_WEEK, # Keep logs for 7 days
            removal_policy=RemovalPolicy.DESTROY # Delete logs when stack is deleted
        )

        # Create IAM role for ECS to pull images and write logs
        execution_role = iam.Role(
            self, "TaskExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"), # ECS can use this role
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy"), # Allows ECS to pull images from ECR and write to CloudWatch
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly") # Allows reading from ECR repositories
            ]
        )

        # Create IAM role for application code (running inside container)
        task_role = iam.Role(
            self, "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"), # ECS tasks can use this role
        )

        # Give container permissions for S3
        task_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW, # Grant permission
            actions=[ # Allowed operations
                "s3:GetObject", # Download S3 files
                "s3:PutObject", # Upload S3 files
                "s3:ListBucket" # List contents of S3 bucket
            ],
            resources=[ # which S3 resources to apply these actions to
                "arn:aws:s3:::mental-health-project-pipeline", # the bucket itself
                "arn:aws:s3:::mental-health-project-pipeline/*" # all objects in the bucket
            ]
        ))
        # Give container permissions for SSM Parameter Store
        task_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "ssm:GetParameter",
                "ssm:GetParameters"
            ],
            resources=[
                f"arn:aws:ssm:{self.region}:{self.account}:parameter/mental-health-pipeline/*"
            ]
        ))

        # Create Fargate task definition
        task_definition = ecs.FargateTaskDefinition(
            self, "MentalhealthTaskDef",
            memory_limit_mib=2048, # 2GB RAM
            cpu=512, # 0.5 vCPU
            execution_role=execution_role, # Role for ECS operations
            task_role=task_role # Role for application
        )

        # Add container to task definition
        container = task_definition.add_container(
            "MentalHealthContainer",
            image=ecs.ContainerImage.from_ecr_repository(ecr_repository, tag="latest"),
            memory_limit_mib=2048, # Container memory limit
            logging=ecs.LogDrivers.aws_logs( # Send container output to CloudWatch
                stream_prefix="mental-health",
                log_group=log_group
            ),
            environment={
                "PYTHONPATH": "/opt/airflow",
                "AIRFLOW__CORE__LOAD_EXAMPLES": "false",
                "AIRFLOW__CORE__DAGS_FOLDER": "/opt/airflow/dags",
                "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN": "sqlite:////tmp/airflow.db",
                "AIRFLOW__CORE__EXECUTOR": "SequentialExecutor",
                # Snowflake
                "SNOWFLAKE_USER": os.getenv("SNOWFLAKE_USER"),
                "SNOWFLAKE_ACCOUNT": os.getenv("SNOWFLAKE_ACCOUNT"),
                "SNOWFLAKE_PRIVATE_KEY_PATH": "/opt/airflow/keys/rsa_key.p8",
                "SNOWFLAKE_WAREHOUSE": os.getenv("SNOWFLAKE_WAREHOUSE"),
                "SNOWFLAKE_DATABASE": os.getenv("SNOWFLAKE_DATABASE"),
                "SNOWFLAKE_ROLE": os.getenv("SNOWFLAKE_ROLE"),
                # Reddit
                "REDDIT_CLIENT_ID": os.getenv("REDDIT_CLIENT_ID"),
                "REDDIT_CLIENT_SECRET": os.getenv("REDDIT_CLIENT_SECRET"),
                "REDDIT_USERNAME": os.getenv("REDDIT_USERNAME"),
                "REDDIT_PASSWORD": os.getenv("REDDIT_PASSWORD"),
                "REDDIT_USER_AGENT": os.getenv("REDDIT_USER_AGENT"),
                # News API
                "NEWS_API_KEY": os.getenv("NEWS_API_KEY"),
                # Anthropic API
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            },
            # Override the entrypoint
            entry_point=["/bin/bash", "-c"],
            # Command
            command=["airflow db init && airflow dags test ingestion_dag $(date +%Y-%m-%d)"]
        )
        
         # Create EventBridge rule for weekly execution (cron job in the cloud)
        rule = events.Rule(
            self, "WeeklyMentalHealthRule",
            schedule=events.Schedule.cron( # Cron schedule
                minute="0",
                hour="8",  # 8 AM UTC
                month="*",
                week_day="1"  # Sunday
            ),
            description="Trigger pipeline weekly"
        )

        # Tell EventBridge what to do when the rule triggers
        rule.add_target(targets.EcsTask(
            cluster=cluster, # ECS cluster
            task_definition=task_definition, # task to run
            launch_type=ecs.LaunchType.FARGATE, # use Fargate
            platform_version=ecs.FargatePlatformVersion.LATEST, # use latest Fargate version
            subnet_selection=ec2.SubnetSelection( # where in VPC to run
                subnet_type=ec2.SubnetType.PUBLIC # Public subnets
            ),
            assign_public_ip=True # public IP for internet access
        ))