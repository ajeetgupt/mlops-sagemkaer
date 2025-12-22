"""
Rollback Endpoint Script
------------------------
Handles rollback to previous model versions.
"""

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EndpointRollback:
    """Manages endpoint rollback operations."""
    
    def __init__(self, region: Optional[str] = None):
        self.region = region or boto3.Session().region_name
        self.sm_client = boto3.client("sagemaker", region_name=self.region)

    def get_model_history(self, model_package_group: str, max_results: int = 10) -> List[Dict]:
        """Get history of approved model versions."""
        response = self.sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=max_results,
        )
        
        return [
            {"arn": pkg["ModelPackageArn"], "version": pkg.get("ModelPackageVersion"),
             "creation_time": pkg["CreationTime"].isoformat()}
            for pkg in response.get("ModelPackageSummaryList", [])
        ]

    def get_endpoint_config_history(self, endpoint_name: str) -> List[Dict]:
        """Get history of endpoint configurations."""
        response = self.sm_client.list_endpoint_configs(
            NameContains=endpoint_name, SortBy="CreationTime", SortOrder="Descending"
        )
        return [{"name": c["EndpointConfigName"], "creation_time": c["CreationTime"].isoformat()}
                for c in response.get("EndpointConfigs", [])]

    def rollback_to_previous(self, endpoint_name: str, reason: str = "Manual rollback") -> Dict:
        """Rollback endpoint to previous version."""
        current = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
        current_config = current["EndpointConfigName"]
        
        configs = self.get_endpoint_config_history(endpoint_name)
        target_config = next((c["name"] for c in configs if c["name"] != current_config), None)
        
        if not target_config:
            raise ValueError("No previous configuration found")
        
        logger.info(f"Rolling back {endpoint_name} from {current_config} to {target_config}")
        
        self.sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=target_config,
            DeploymentConfig={
                "BlueGreenUpdatePolicy": {
                    "TrafficRoutingConfiguration": {"Type": "ALL_AT_ONCE", "WaitIntervalInSeconds": 0},
                    "TerminationWaitInSeconds": 60,
                    "MaximumExecutionTimeoutInSeconds": 1800,
                }
            },
        )
        
        waiter = self.sm_client.get_waiter("endpoint_in_service")
        waiter.wait(EndpointName=endpoint_name, WaiterConfig={"Delay": 30, "MaxAttempts": 60})
        
        return {"endpoint": endpoint_name, "previous": current_config, "new": target_config,
                "reason": reason, "status": "completed", "timestamp": datetime.utcnow().isoformat()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", type=str, required=True)
    parser.add_argument("--version", type=str, default="previous")
    parser.add_argument("--reason", type=str, default="Manual rollback")
    parser.add_argument("--list-history", action="store_true")
    parser.add_argument("--model-package-group", type=str, default="mlops-models")
    args = parser.parse_args()
    
    rollback = EndpointRollback()
    
    if args.list_history:
        print("\n=== Model History ===")
        for m in rollback.get_model_history(args.model_package_group):
            print(f"  {m['version']}: {m['arn']}")
        print("\n=== Endpoint Configs ===")
        for c in rollback.get_endpoint_config_history(args.endpoint_name):
            print(f"  {c['name']}")
    else:
        result = rollback.rollback_to_previous(args.endpoint_name, args.reason)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
