"""
Endpoint Deployment Script
--------------------------
Handles SageMaker endpoint deployment with blue-green strategy and traffic shifting.
"""

import argparse
import json
import logging
import time
from datetime import datetime
from typing import Dict, Optional

import boto3
import sagemaker
from sagemaker import get_execution_role

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EndpointDeployer:
    """Manages SageMaker endpoint deployments."""
    
    def __init__(
        self,
        region: Optional[str] = None,
        role: Optional[str] = None,
    ):
        self.session = sagemaker.Session()
        self.region = region or self.session.boto_region_name
        self.role = role or get_execution_role()
        self.sm_client = boto3.client("sagemaker", region_name=self.region)

    def deploy_endpoint(
        self,
        model_package_arn: str,
        endpoint_name: str,
        config: Dict,
        environment: str = "production",
    ) -> Dict:
        """
        Deploy model to endpoint with blue-green strategy.
        
        Args:
            model_package_arn: ARN of approved model package
            endpoint_name: Target endpoint name
            config: Deployment configuration
            environment: Target environment (staging/production)
            
        Returns:
            Deployment result
        """
        logger.info(f"Deploying to endpoint: {endpoint_name}")
        
        env_config = config["environments"][environment]
        deployment_config = config["deployment_strategy"]
        
        # Create model from model package
        model_name = f"{endpoint_name}-model-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        model_response = self.sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "ModelPackageName": model_package_arn,
            },
            ExecutionRoleArn=self.role,
            Tags=[
                {"Key": "Environment", "Value": environment},
                {"Key": "DeploymentTime", "Value": datetime.utcnow().isoformat()},
            ],
        )
        logger.info(f"Model created: {model_name}")
        
        # Create endpoint config
        endpoint_config_name = f"{endpoint_name}-config-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        endpoint_config_response = self.sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InstanceType": env_config["instance_type"],
                    "InitialInstanceCount": env_config["instance_count"],
                    "InitialVariantWeight": 1.0,
                }
            ],
            DataCaptureConfig=self._get_data_capture_config(config) if config.get("data_capture", {}).get("enabled") else None,
            Tags=[
                {"Key": "Environment", "Value": environment},
            ],
        )
        logger.info(f"Endpoint config created: {endpoint_config_name}")
        
        # Check if endpoint exists
        try:
            self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            endpoint_exists = True
        except self.sm_client.exceptions.ClientError:
            endpoint_exists = False
        
        if endpoint_exists:
            # Update existing endpoint (blue-green)
            return self._update_endpoint_blue_green(
                endpoint_name=endpoint_name,
                endpoint_config_name=endpoint_config_name,
                deployment_config=deployment_config,
            )
        else:
            # Create new endpoint
            return self._create_endpoint(
                endpoint_name=endpoint_name,
                endpoint_config_name=endpoint_config_name,
            )

    def _create_endpoint(
        self,
        endpoint_name: str,
        endpoint_config_name: str,
    ) -> Dict:
        """Create new endpoint."""
        logger.info(f"Creating new endpoint: {endpoint_name}")
        
        response = self.sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
            Tags=[
                {"Key": "ManagedBy", "Value": "mlops-template"},
            ],
        )
        
        # Wait for endpoint to be in service
        self._wait_for_endpoint(endpoint_name)
        
        return {
            "endpoint_name": endpoint_name,
            "endpoint_config": endpoint_config_name,
            "status": "Created",
            "endpoint_arn": response["EndpointArn"],
        }

    def _update_endpoint_blue_green(
        self,
        endpoint_name: str,
        endpoint_config_name: str,
        deployment_config: Dict,
    ) -> Dict:
        """Update endpoint using blue-green deployment."""
        logger.info(f"Updating endpoint with blue-green deployment: {endpoint_name}")
        
        traffic_config = deployment_config.get("traffic_routing", {})
        traffic_type = traffic_config.get("type", "all_at_once")
        
        if traffic_type == "linear":
            # Gradual traffic shift
            linear_config = traffic_config.get("linear_config", {})
            deployment_preference = {
                "BlueGreenUpdatePolicy": {
                    "TrafficRoutingConfiguration": {
                        "Type": "LINEAR",
                        "LinearStepSize": {
                            "Type": "CAPACITY_PERCENT",
                            "Value": linear_config.get("step_percentage", 20),
                        },
                        "WaitIntervalInSeconds": linear_config.get("wait_interval_seconds", 300),
                    },
                    "TerminationWaitInSeconds": deployment_config.get("health_check", {}).get("termination_wait_seconds", 300),
                    "MaximumExecutionTimeoutInSeconds": 3600,
                },
            }
        elif traffic_type == "canary":
            # Canary deployment
            canary_config = traffic_config.get("canary_config", {})
            deployment_preference = {
                "BlueGreenUpdatePolicy": {
                    "TrafficRoutingConfiguration": {
                        "Type": "CANARY",
                        "CanarySize": {
                            "Type": "CAPACITY_PERCENT",
                            "Value": canary_config.get("initial_percentage", 10),
                        },
                        "WaitIntervalInSeconds": canary_config.get("baking_period_seconds", 600),
                    },
                    "TerminationWaitInSeconds": 300,
                    "MaximumExecutionTimeoutInSeconds": 3600,
                },
            }
        else:
            # All at once
            deployment_preference = {
                "BlueGreenUpdatePolicy": {
                    "TrafficRoutingConfiguration": {
                        "Type": "ALL_AT_ONCE",
                        "WaitIntervalInSeconds": 0,
                    },
                    "TerminationWaitInSeconds": 120,
                    "MaximumExecutionTimeoutInSeconds": 1800,
                },
            }
        
        # Add auto-rollback if configured
        if deployment_config.get("auto_rollback", {}).get("enabled"):
            deployment_preference["AutoRollbackConfiguration"] = {
                "Alarms": [
                    {"AlarmName": alarm}
                    for alarm in deployment_config["auto_rollback"].get("alarm_names", [])
                ],
            }
        
        response = self.sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
            DeploymentConfig=deployment_preference,
            RetainDeploymentConfig=False,
        )
        
        # Wait for update to complete
        self._wait_for_endpoint(endpoint_name)
        
        return {
            "endpoint_name": endpoint_name,
            "endpoint_config": endpoint_config_name,
            "status": "Updated",
            "endpoint_arn": response["EndpointArn"],
            "deployment_type": traffic_type,
        }

    def _get_data_capture_config(self, config: Dict) -> Dict:
        """Get data capture configuration."""
        dc_config = config.get("data_capture", {})
        
        return {
            "EnableCapture": True,
            "InitialSamplingPercentage": dc_config.get("sampling_percentage", 10),
            "DestinationS3Uri": f"s3://{self.session.default_bucket()}/{dc_config.get('destination_path', 'monitoring/data-capture')}",
            "CaptureOptions": [
                {"CaptureMode": mode} for mode in dc_config.get("capture_modes", ["Input", "Output"])
            ],
            "CaptureContentTypeHeader": {
                "JsonContentTypes": ["application/json"],
            },
        }

    def _wait_for_endpoint(self, endpoint_name: str, timeout: int = 1800):
        """Wait for endpoint to be in service."""
        logger.info(f"Waiting for endpoint {endpoint_name} to be in service...")
        
        waiter = self.sm_client.get_waiter("endpoint_in_service")
        waiter.wait(
            EndpointName=endpoint_name,
            WaiterConfig={"Delay": 30, "MaxAttempts": timeout // 30},
        )
        
        logger.info(f"Endpoint {endpoint_name} is now in service")

    def health_check(self, endpoint_name: str) -> Dict:
        """Perform health check on endpoint."""
        runtime = boto3.client("sagemaker-runtime", region_name=self.region)
        
        # Test inference
        test_payload = json.dumps({"instances": [[0.1] * 10]})  # Dummy payload
        
        try:
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=test_payload,
            )
            
            result = json.loads(response["Body"].read().decode())
            
            return {
                "status": "healthy",
                "endpoint_name": endpoint_name,
                "response_time_ms": response["ResponseMetadata"]["HTTPHeaders"].get("x-amzn-requestid"),
                "sample_response": result,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "endpoint_name": endpoint_name,
                "error": str(e),
            }


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-package-arn", type=str, required=True)
    parser.add_argument("--endpoint-name", type=str, required=True)
    parser.add_argument("--environment", type=str, default="production")
    parser.add_argument("--config-path", type=str, default="config/deployment_config.yaml")
    parser.add_argument("--health-check-only", action="store_true")
    args = parser.parse_args()
    
    # Load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    deployer = EndpointDeployer()
    
    if args.health_check_only:
        result = deployer.health_check(args.endpoint_name)
    else:
        result = deployer.deploy_endpoint(
            model_package_arn=args.model_package_arn,
            endpoint_name=args.endpoint_name,
            config=config,
            environment=args.environment,
        )
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
