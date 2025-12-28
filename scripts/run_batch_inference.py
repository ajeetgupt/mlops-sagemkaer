"""
Batch Inference Script
----------------------
Runs SageMaker Batch Transform jobs for large-scale batch predictions.
"""

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, Optional

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.transformer import Transformer

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchInference:
    """Manages SageMaker Batch Transform jobs."""
    
    def __init__(self, region: Optional[str] = None, role: Optional[str] = None):
        self.session = sagemaker.Session()
        self.region = region or self.session.boto_region_name
        self.role = role or get_execution_role()
        self.sm_client = boto3.client("sagemaker", region_name=self.region)

    def run_batch_transform(
        self,
        model_name: str,
        input_path: str,
        output_path: str,
        config: Dict,
        job_name: Optional[str] = None,
    ) -> Dict:
        """
        Run batch transform job.
        
        Args:
            model_name: SageMaker model name or model package ARN
            input_path: S3 path to input data
            output_path: S3 path for predictions
            config: Batch transform configuration
            job_name: Optional custom job name
            
        Returns:
            Job details
        """
        if not job_name:
            job_name = f"batch-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info(f"Starting batch transform job: {job_name}")
        
        batch_config = config.get("inference", {}).get("batch_transform", {})
        
        # Create transformer
        transformer = Transformer(
            model_name=model_name,
            instance_count=batch_config.get("instance_count", 1),
            instance_type=batch_config.get("instance_type", "ml.m5.xlarge"),
            output_path=output_path,
            base_transform_job_name=job_name,
            sagemaker_session=self.session,
            strategy=batch_config.get("strategy", "SingleRecord"),
            max_payload=batch_config.get("max_payload_mb", 6),
            accept="application/json",
        )
        
        # Start transform job
        transformer.transform(
            data=input_path,
            content_type="text/csv",
            split_type="Line",
            join_source="Input",  # Include input data in output
            wait=False,
        )
        
        logger.info(f"Batch transform job started: {transformer.latest_transform_job.name}")
        
        return {
            "job_name": transformer.latest_transform_job.name,
            "input_path": input_path,
            "output_path": output_path,
            "status": "InProgress",
        }

    def wait_for_completion(self, job_name: str, timeout: int = 3600) -> Dict:
        """Wait for batch transform job to complete."""
        logger.info(f"Waiting for job {job_name} to complete...")
        
        waiter = self.sm_client.get_waiter("transform_job_completed_or_stopped")
        waiter.wait(
            TransformJobName=job_name,
            WaiterConfig={"Delay": 30, "MaxAttempts": timeout // 30},
        )
        
        # Get final status
        response = self.sm_client.describe_transform_job(TransformJobName=job_name)
        
        result = {
            "job_name": job_name,
            "status": response["TransformJobStatus"],
            "input_path": response["TransformInput"]["DataSource"]["S3DataSource"]["S3Uri"],
            "output_path": response["TransformOutput"]["S3OutputPath"],
        }
        
        if response["TransformJobStatus"] == "Failed":
            result["failure_reason"] = response.get("FailureReason", "Unknown")
            logger.error(f"Job failed: {result['failure_reason']}")
        else:
            logger.info(f"Job completed successfully: {job_name}")
        
        return result

    def get_latest_model(self, model_package_group: str) -> str:
        """Get latest approved model from registry."""
        response = self.sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
        
        if not response["ModelPackageSummaryList"]:
            raise ValueError(f"No approved models found in {model_package_group}")
        
        model_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
        logger.info(f"Using model: {model_arn}")
        
        # Create model from package
        model_name = f"batch-model-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        self.sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={"ModelPackageName": model_arn},
            ExecutionRoleArn=self.role,
        )
        
        return model_name

    def schedule_batch_job(
        self,
        schedule_expression: str,
        rule_name: str,
        input_path: str,
        output_path: str,
        model_package_group: str,
    ):
        """Schedule recurring batch transform job using EventBridge."""
        events = boto3.client("events", region_name=self.region)
        
        # Create EventBridge rule
        rule_response = events.put_rule(
            Name=rule_name,
            ScheduleExpression=schedule_expression,
            State="ENABLED",
            Description="Scheduled batch inference job",
        )
        
        # Add Lambda target (would need Lambda function to trigger batch job)
        # For simplicity, this shows the structure
        logger.info(f"EventBridge rule created: {rule_response['RuleArn']}")
        logger.info("Note: You need to create a Lambda function to trigger the batch job")
        
        return rule_response

    def monitor_batch_metrics(self, job_name: str):
        """Publish batch job metrics to CloudWatch."""
        cw = boto3.client("cloudwatch", region_name=self.region)
        
        response = self.sm_client.describe_transform_job(TransformJobName=job_name)
        
        # Calculate metrics
        if "ProcessedRecords" in response:
            processed = response["ProcessedRecords"]
            failed = response.get("FailedRecords", 0)
            
            cw.put_metric_data(
                Namespace="MLOps/BatchInference",
                MetricData=[
                    {"MetricName": "ProcessedRecords", "Value": processed, "Unit": "Count"},
                    {"MetricName": "FailedRecords", "Value": failed, "Unit": "Count"},
                    {"MetricName": "SuccessRate", "Value": (processed - failed) / processed * 100 if processed > 0 else 0, "Unit": "Percent"},
                ],
            )


def main():
    """Main batch inference function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, help="Model name or use latest from registry")
    parser.add_argument("--model-package-group", type=str, default="mlops-models")
    parser.add_argument("--input-path", type=str, required=True, help="S3 input path")
    parser.add_argument("--output-path", type=str, required=True, help="S3 output path")
    parser.add_argument("--config-path", type=str, default="config/deployment_config.yaml")
    parser.add_argument("--wait", action="store_true", help="Wait for completion")
    parser.add_argument("--schedule", type=str, help="Schedule expression (e.g., 'rate(1 day)')")
    args = parser.parse_args()
    
    # Load config
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    
    batch = BatchInference()
    
    # Get model
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = batch.get_latest_model(args.model_package_group)
    
    if args.schedule:
        # Schedule recurring job
        batch.schedule_batch_job(
            schedule_expression=args.schedule,
            rule_name="batch-inference-schedule",
            input_path=args.input_path,
            output_path=args.output_path,
            model_package_group=args.model_package_group,
        )
    else:
        # Run one-time job
        result = batch.run_batch_transform(
            model_name=model_name,
            input_path=args.input_path,
            output_path=args.output_path,
            config=config,
        )
        
        if args.wait:
            result = batch.wait_for_completion(result["job_name"])
            batch.monitor_batch_metrics(result["job_name"])
        
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
