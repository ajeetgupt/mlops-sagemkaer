"""
Integration tests for SageMaker pipelines.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestTrainingPipeline:
    """Tests for training pipeline."""

    @patch("sagemaker.Session")
    @patch("sagemaker.get_execution_role")
    def test_pipeline_creation(self, mock_role, mock_session):
        """Test pipeline can be created."""
        mock_role.return_value = "arn:aws:iam::123456789:role/SageMakerRole"
        mock_session.return_value.boto_region_name = "us-east-1"
        mock_session.return_value.default_bucket.return_value = "test-bucket"
        
        # This tests that imports and basic structure work
        # Full pipeline testing requires AWS resources
        from src.pipelines.training_pipeline import load_config
        
        # Test config loading
        config = load_config("config/training_config.yaml")
        assert "hyperparameters" in config
        assert "training" in config
        assert "validation" in config


class TestRetrainingPipeline:
    """Tests for retraining pipeline."""

    @patch("boto3.client")
    def test_trigger_retraining(self, mock_boto):
        """Test retraining trigger."""
        mock_sm = MagicMock()
        mock_sm.start_pipeline_execution.return_value = {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-east-1:123:pipeline/test/execution/123"
        }
        mock_boto.return_value = mock_sm
        
        from src.pipelines.retraining_pipeline import RetrainingPipeline
        
        with patch("sagemaker.Session"):
            pipeline = RetrainingPipeline(pipeline_name="TestPipeline")
            # Mock the sagemaker client
            pipeline.sagemaker_client = mock_sm
            
            result = pipeline.trigger_retraining(trigger_type="test")
            
            assert "PipelineExecutionArn" in result


class TestDriftDetection:
    """Tests for drift detection."""

    def test_drift_detector(self, tmp_path):
        """Test drift detection logic."""
        import pandas as pd
        import numpy as np
        
        # Create baseline and current data with no drift
        baseline = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})
        current = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})
        
        baseline_path = tmp_path / "baseline.csv"
        current_path = tmp_path / "current.csv"
        baseline.to_csv(baseline_path, index=False)
        current.to_csv(current_path, index=False)
        
        from scripts.data_drift_check import DriftDetector
        
        with patch.object(DriftDetector, "_publish_metrics"):
            detector = DriftDetector()
            result = detector.check_drift(str(baseline_path), str(current_path))
        
        # Should not detect drift in similar distributions
        assert "drift_detected" in result
        assert "overall_drift_score" in result

    def test_drift_detected(self, tmp_path):
        """Test drift detection with drifted data."""
        import pandas as pd
        import numpy as np
        
        # Create data with significant drift
        baseline = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})
        current = pd.DataFrame({"feature1": np.random.normal(5, 1, 1000)})  # Shifted mean
        
        baseline_path = tmp_path / "baseline.csv"
        current_path = tmp_path / "current.csv"
        baseline.to_csv(baseline_path, index=False)
        current.to_csv(current_path, index=False)
        
        from scripts.data_drift_check import DriftDetector
        
        with patch.object(DriftDetector, "_publish_metrics"):
            detector = DriftDetector()
            result = detector.check_drift(str(baseline_path), str(current_path))
        
        # Should detect drift
        assert result["drift_detected"] is True
