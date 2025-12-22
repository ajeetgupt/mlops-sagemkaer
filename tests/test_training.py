"""
Unit tests for training module.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestTrainingScript:
    """Tests for the training script."""

    def test_load_data_csv(self, tmp_path):
        """Test loading CSV data."""
        from src.training.train import load_data
        
        # Create test data
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "target": [0, 1, 0, 1, 0],
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        X, y = load_data(str(tmp_path), "target")
        
        assert len(X) == 5
        assert len(y) == 5
        assert "target" not in X.columns
        assert list(y) == [0, 1, 0, 1, 0]

    def test_evaluate_model(self):
        """Test model evaluation metrics."""
        from src.training.train import evaluate_model
        import xgboost as xgb
        
        # Create mock model and data
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train({"objective": "binary:logistic"}, dtrain, num_boost_round=10)
        
        dval = xgb.DMatrix(X, label=y)
        metrics = evaluate_model(model, dval, y)
        
        assert "auc" in metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["auc"] <= 1
        assert 0 <= metrics["accuracy"] <= 1


class TestPreprocessing:
    """Tests for preprocessing module."""

    def test_handle_missing_values(self):
        """Test missing value handling."""
        from src.training.preprocess import DataPreprocessor
        
        df = pd.DataFrame({
            "feature1": [1, 2, np.nan, 4, 5],
            "feature2": [0.1, np.nan, 0.3, 0.4, 0.5],
            "target": [0, 1, 0, 1, 0],
        })
        
        preprocessor = DataPreprocessor(target_column="target", handle_missing="median")
        result = preprocessor.fit_transform(df)
        
        assert result["feature1"].isnull().sum() == 0
        assert result["feature2"].isnull().sum() == 0

    def test_split_data(self):
        """Test data splitting."""
        from src.training.preprocess import split_data
        
        df = pd.DataFrame({
            "feature1": range(100),
            "target": [0] * 50 + [1] * 50,
        })
        
        train, val, test = split_data(df, train_split=0.8, val_split=0.1, test_split=0.1)
        
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_normalization(self):
        """Test feature normalization."""
        from src.training.preprocess import DataPreprocessor
        
        df = pd.DataFrame({
            "feature1": [10, 20, 30, 40, 50],
            "target": [0, 1, 0, 1, 0],
        })
        
        preprocessor = DataPreprocessor(target_column="target", normalize=True)
        result = preprocessor.fit_transform(df)
        
        # Check standardization (mean ~0, std ~1)
        assert abs(result["feature1"].mean()) < 0.1
        assert abs(result["feature1"].std() - 1) < 0.1


class TestInference:
    """Tests for inference module."""

    def test_input_fn_json(self):
        """Test JSON input parsing."""
        from src.inference.inference import input_fn
        
        payload = json.dumps({"instances": [[0.1, 0.2, 0.3]]})
        result = input_fn(payload, "application/json")
        
        assert result is not None

    def test_output_fn_json(self):
        """Test JSON output formatting."""
        from src.inference.inference import output_fn
        
        predictions = np.array([0.8, 0.3, 0.6])
        result = output_fn(predictions, "application/json")
        
        parsed = json.loads(result)
        assert "predictions" in parsed
        assert len(parsed["predictions"]) == 3
        assert parsed["predictions"][0]["predicted_class"] == 1
        assert parsed["predictions"][1]["predicted_class"] == 0
