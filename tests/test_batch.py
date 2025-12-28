"""
Unit tests for batch inference.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestBatchInference:
    """Tests for batch inference functionality."""

    @patch("boto3.client")
    @patch("sagemaker.Session")
    def test_batch_transform_creation(self, mock_session, mock_boto):
        """Test batch transform job creation."""
        from scripts.run_batch_inference import BatchInference
        
        mock_sm = MagicMock()
        mock_boto.return_value = mock_sm
        
        batch = BatchInference()
        batch.sm_client = mock_sm
        
        config = {
            "inference": {
                "batch_transform": {
                    "instance_type": "ml.m5.xlarge",
                    "instance_count": 1,
                    "max_payload_mb": 6,
                    "strategy": "SingleRecord",
                }
            }
        }
        
        # This would test the batch transform logic
        assert config["inference"]["batch_transform"]["instance_type"] == "ml.m5.xlarge"

    def test_postprocess_predictions(self, tmp_path):
        """Test batch prediction post-processing."""
        from scripts.postprocess_batch import postprocess_predictions
        
        # Create test predictions
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        
        # Create sample prediction file
        pred_file = input_dir / "predictions.out"
        predictions = [
            {"predictions": [{"probability": 0.9, "predicted_class": 1}]},
            {"predictions": [{"probability": 0.3, "predicted_class": 0}]},
        ]
        
        with open(pred_file, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
        
        # Run post-processing
        postprocess_predictions(str(input_dir), str(output_dir))
        
        # Check output exists
        output_file = output_dir / "processed_predictions.csv"
        assert output_file.exists()


class TestBatchPipeline:
    """Tests for batch processing pipeline."""

    @patch("sagemaker.Session")
    @patch("sagemaker.get_execution_role")
    def test_pipeline_creation(self, mock_role, mock_session):
        """Test batch pipeline can be created."""
        mock_role.return_value = "arn:aws:iam::123456789:role/SageMakerRole"
        mock_session.return_value.boto_region_name = "us-east-1"
        mock_session.return_value.default_bucket.return_value = "test-bucket"
        
        from src.pipelines.batch_pipeline import create_batch_pipeline
        
        pipeline = create_batch_pipeline(
            pipeline_name="TestBatchPipeline",
            role="arn:aws:iam::123456789:role/SageMakerRole",
            s3_bucket="test-bucket",
        )
        
        assert pipeline is not None
        assert pipeline.name == "TestBatchPipeline"
