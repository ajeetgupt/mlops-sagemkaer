"""
Unit tests for inference module.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestInferenceHandler:
    """Tests for inference handler."""

    def test_input_json_instances(self):
        """Test JSON input with instances format."""
        from src.inference.inference import input_fn
        
        payload = json.dumps({"instances": [[0.1, 0.2], [0.3, 0.4]]})
        result = input_fn(payload, "application/json")
        assert result is not None

    def test_input_json_data(self):
        """Test JSON input with data format."""
        from src.inference.inference import input_fn
        
        payload = json.dumps({"data": [[0.1, 0.2], [0.3, 0.4]]})
        result = input_fn(payload, "application/json")
        assert result is not None

    def test_input_json_list(self):
        """Test JSON input with list format."""
        from src.inference.inference import input_fn
        
        payload = json.dumps([[0.1, 0.2], [0.3, 0.4]])
        result = input_fn(payload, "application/json")
        assert result is not None

    def test_input_csv(self):
        """Test CSV input parsing."""
        from src.inference.inference import input_fn
        
        payload = "0.1,0.2\n0.3,0.4"
        result = input_fn(payload, "text/csv")
        assert result is not None

    def test_output_json(self):
        """Test JSON output formatting."""
        from src.inference.inference import output_fn
        
        predictions = np.array([0.9, 0.1, 0.5])
        result = output_fn(predictions, "application/json")
        
        parsed = json.loads(result)
        assert "predictions" in parsed
        assert len(parsed["predictions"]) == 3
        
        # Check first prediction (0.9 >= 0.5 -> class 1)
        assert parsed["predictions"][0]["probability"] == pytest.approx(0.9)
        assert parsed["predictions"][0]["predicted_class"] == 1
        
        # Check second prediction (0.1 < 0.5 -> class 0)
        assert parsed["predictions"][1]["probability"] == pytest.approx(0.1)
        assert parsed["predictions"][1]["predicted_class"] == 0

    def test_output_csv(self):
        """Test CSV output formatting."""
        from src.inference.inference import output_fn
        
        predictions = np.array([0.9, 0.1])
        result = output_fn(predictions, "text/csv")
        
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert float(lines[0]) == pytest.approx(0.9)


class TestModelLoading:
    """Tests for model loading."""

    @patch("xgboost.Booster")
    def test_model_fn(self, mock_booster):
        """Test model loading function."""
        from src.inference.inference import model_fn
        
        mock_model = MagicMock()
        mock_booster.return_value = mock_model
        
        with patch("os.path.exists", return_value=True):
            result = model_fn("/opt/ml/model")
        
        assert result is not None
