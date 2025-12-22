"""
Data Drift Detection Script
---------------------------
Monitors data distribution changes and triggers alerts.
"""

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, Optional
import boto3
import pandas as pd
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift between baseline and current data."""
    
    def __init__(self, region: Optional[str] = None):
        self.region = region or boto3.Session().region_name
        self.s3 = boto3.client("s3", region_name=self.region)
        self.cw = boto3.client("cloudwatch", region_name=self.region)

    def check_drift(self, baseline_path: str, current_path: str, threshold: float = 0.05) -> Dict:
        """Compare baseline and current data distributions."""
        baseline_df = self._load_data(baseline_path)
        current_df = self._load_data(current_path)
        
        results = {"timestamp": datetime.utcnow().isoformat(), "features": {}, "drift_detected": False}
        drift_scores = []
        
        for col in baseline_df.select_dtypes(include=[np.number]).columns:
            if col not in current_df.columns:
                continue
            
            # KS test for numerical columns
            stat, p_value = stats.ks_2samp(baseline_df[col].dropna(), current_df[col].dropna())
            
            drift = p_value < threshold
            results["features"][col] = {"ks_statistic": float(stat), "p_value": float(p_value), "drift": drift}
            drift_scores.append(stat)
            
            if drift:
                results["drift_detected"] = True
                logger.warning(f"Drift detected in {col}: p={p_value:.4f}")
        
        results["overall_drift_score"] = float(np.mean(drift_scores)) if drift_scores else 0.0
        self._publish_metrics(results)
        
        return results

    def _load_data(self, path: str) -> pd.DataFrame:
        """Load data from S3 or local path."""
        if path.startswith("s3://"):
            import io
            bucket, key = path.replace("s3://", "").split("/", 1)
            obj = self.s3.get_object(Bucket=bucket, Key=key)
            return pd.read_csv(io.BytesIO(obj["Body"].read()))
        return pd.read_csv(path)

    def _publish_metrics(self, results: Dict):
        """Publish drift metrics to CloudWatch."""
        self.cw.put_metric_data(
            Namespace="MLOps/Monitoring",
            MetricData=[
                {"MetricName": "DataDriftScore", "Value": results["overall_drift_score"], "Unit": "None"},
                {"MetricName": "DriftDetected", "Value": 1 if results["drift_detected"] else 0, "Unit": "Count"},
            ],
        )
        logger.info(f"Published metrics: drift_score={results['overall_drift_score']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, help="Baseline data path")
    parser.add_argument("--current", type=str, required=True, help="Current data path")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    detector = DriftDetector()
    results = detector.check_drift(args.baseline, args.current, args.threshold)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
    
    print(json.dumps(results, indent=2))
    
    if results["drift_detected"]:
        exit(1)  # Non-zero exit for CI/CD integration


if __name__ == "__main__":
    main()
