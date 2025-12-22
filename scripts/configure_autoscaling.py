"""
Autoscaling Configuration Script
--------------------------------
Configures autoscaling for SageMaker endpoints.
"""

import argparse
import json
import logging
from typing import Dict, Optional
import boto3
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoscalingConfigurator:
    """Configures autoscaling for SageMaker endpoints."""
    
    def __init__(self, region: Optional[str] = None):
        self.region = region or boto3.Session().region_name
        self.aas_client = boto3.client("application-autoscaling", region_name=self.region)
        self.cw_client = boto3.client("cloudwatch", region_name=self.region)

    def configure_autoscaling(self, endpoint_name: str, variant_name: str, config: Dict) -> Dict:
        """Configure autoscaling for endpoint variant."""
        resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"
        capacity = config["autoscaling"]["capacity"]
        
        # Register scalable target
        self.aas_client.register_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            MinCapacity=capacity["min_instances"],
            MaxCapacity=capacity["max_instances"],
        )
        logger.info(f"Registered scalable target: {resource_id}")
        
        results = {"resource_id": resource_id, "policies": []}
        
        # Target tracking policy
        if config["autoscaling"]["target_tracking"]["enabled"]:
            tt = config["autoscaling"]["target_tracking"]
            self.aas_client.put_scaling_policy(
                PolicyName=tt["policy_name"],
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                PolicyType="TargetTrackingScaling",
                TargetTrackingScalingPolicyConfiguration={
                    "TargetValue": tt["target_value"],
                    "PredefinedMetricSpecification": {
                        "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
                    },
                    "ScaleInCooldown": tt["scale_in_cooldown"],
                    "ScaleOutCooldown": tt["scale_out_cooldown"],
                    "DisableScaleIn": tt.get("disable_scale_in", False),
                },
            )
            results["policies"].append(tt["policy_name"])
            logger.info(f"Created target tracking policy: {tt['policy_name']}")
        
        # Step scaling policies
        if config["autoscaling"]["step_scaling"]["enabled"]:
            for policy in config["autoscaling"]["step_scaling"]["policies"]:
                self._create_step_scaling(resource_id, policy)
                results["policies"].append(policy["policy_name"])
        
        return results

    def _create_step_scaling(self, resource_id: str, policy: Dict):
        """Create step scaling policy with CloudWatch alarm."""
        # Create scaling policy
        adjustment = policy["scaling_adjustment"]
        self.aas_client.put_scaling_policy(
            PolicyName=policy["policy_name"],
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            PolicyType="StepScaling",
            StepScalingPolicyConfiguration={
                "AdjustmentType": policy["adjustment_type"],
                "StepAdjustments": [{
                    "MetricIntervalLowerBound": 0 if adjustment > 0 else None,
                    "MetricIntervalUpperBound": None if adjustment > 0 else 0,
                    "ScalingAdjustment": adjustment,
                }],
                "Cooldown": policy["cooldown"],
            },
        )
        
        # Create CloudWatch alarm
        self.cw_client.put_metric_alarm(
            AlarmName=f"{policy['policy_name']}-alarm",
            MetricName=policy["metric_name"],
            Namespace="AWS/SageMaker",
            Statistic="Average",
            Period=policy["period_seconds"],
            EvaluationPeriods=policy["evaluation_periods"],
            Threshold=policy["alarm_threshold"],
            ComparisonOperator=policy["comparison_operator"],
        )
        logger.info(f"Created step scaling policy: {policy['policy_name']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", type=str, required=True)
    parser.add_argument("--variant-name", type=str, default="AllTraffic")
    parser.add_argument("--config-path", type=str, default="config/autoscaling_config.yaml")
    args = parser.parse_args()
    
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    
    configurator = AutoscalingConfigurator()
    result = configurator.configure_autoscaling(args.endpoint_name, args.variant_name, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
