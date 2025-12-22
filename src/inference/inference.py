"""
SageMaker Inference Handler
---------------------------
Custom inference handler for SageMaker real-time endpoints.
Handles model loading, input preprocessing, prediction, and output formatting.
"""

import json
import logging
import os
from io import StringIO
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import xgboost as xgb

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def model_fn(model_dir: str) -> xgb.Booster:
    """
    Load model from the model directory.
    Called once when the endpoint is created.
    
    Args:
        model_dir: Directory containing the model artifacts
        
    Returns:
        Loaded XGBoost model
    """
    logger.info(f"Loading model from {model_dir}")
    
    model_path = os.path.join(model_dir, "xgboost-model")
    
    if not os.path.exists(model_path):
        # Try alternative paths
        for alt_name in ["model.xgb", "model.bin", "model"]:
            alt_path = os.path.join(model_dir, alt_name)
            if os.path.exists(alt_path):
                model_path = alt_path
                break
    
    model = xgb.Booster()
    model.load_model(model_path)
    
    logger.info("Model loaded successfully")
    return model


def input_fn(request_body: str, request_content_type: str) -> xgb.DMatrix:
    """
    Deserialize and preprocess input data.
    
    Args:
        request_body: Raw request body
        request_content_type: Content type of the request
        
    Returns:
        XGBoost DMatrix ready for prediction
    """
    logger.info(f"Processing input with content type: {request_content_type}")
    
    if request_content_type == "application/json":
        # Parse JSON input
        data = json.loads(request_body)
        
        # Handle different JSON formats
        if isinstance(data, dict):
            if "instances" in data:
                # Format: {"instances": [[...], [...], ...]}
                instances = data["instances"]
            elif "data" in data:
                # Format: {"data": [[...], [...], ...]}
                instances = data["data"]
            else:
                # Format: {"feature1": value1, "feature2": value2, ...}
                instances = [list(data.values())]
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                # Format: [{"feature1": v1}, {"feature1": v2}, ...]
                instances = [list(d.values()) for d in data]
            else:
                # Format: [[...], [...], ...] or [...]
                instances = data if isinstance(data[0], list) else [data]
        else:
            raise ValueError(f"Unsupported JSON structure")
        
        df = pd.DataFrame(instances)
        
    elif request_content_type == "text/csv":
        # Parse CSV input
        df = pd.read_csv(StringIO(request_body), header=None)
        
    elif request_content_type == "application/x-npy":
        # Parse numpy array
        import io
        arr = np.load(io.BytesIO(request_body.encode() if isinstance(request_body, str) else request_body))
        df = pd.DataFrame(arr)
        
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")
    
    logger.info(f"Input shape: {df.shape}")
    
    return xgb.DMatrix(df)


def predict_fn(input_data: xgb.DMatrix, model: xgb.Booster) -> np.ndarray:
    """
    Make predictions using the model.
    
    Args:
        input_data: Preprocessed input data
        model: Loaded model
        
    Returns:
        Model predictions
    """
    logger.info("Making predictions")
    
    predictions = model.predict(input_data)
    
    logger.info(f"Generated {len(predictions)} predictions")
    
    return predictions


def output_fn(prediction: np.ndarray, accept: str) -> str:
    """
    Serialize predictions to the desired output format.
    
    Args:
        prediction: Model predictions
        accept: Accepted content type
        
    Returns:
        Serialized predictions
    """
    logger.info(f"Formatting output for content type: {accept}")
    
    if accept == "application/json":
        # Convert to list for JSON serialization
        if isinstance(prediction, np.ndarray):
            predictions_list = prediction.tolist()
        else:
            predictions_list = list(prediction)
        
        # Include probability and binary class
        response = {
            "predictions": [
                {
                    "probability": float(p),
                    "predicted_class": int(p >= 0.5),
                }
                for p in predictions_list
            ]
        }
        
        return json.dumps(response)
    
    elif accept == "text/csv":
        return "\n".join(map(str, prediction.tolist()))
    
    else:
        # Default to JSON
        return json.dumps({"predictions": prediction.tolist()})


# For SageMaker Inference Toolkit compatibility
def handler(data: Dict[str, Any], context: Any) -> List[Dict[str, Any]]:
    """
    Main handler for SageMaker inference requests.
    This is an alternative entry point used by some container configurations.
    """
    # Get the model (should be loaded at startup)
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    model = model_fn(model_dir)
    
    # Process input
    if isinstance(data, list):
        data = data[0]
    
    body = data.get("body", data)
    content_type = data.get("content-type", "application/json")
    
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    
    # Make prediction
    input_data = input_fn(body, content_type)
    predictions = predict_fn(input_data, model)
    
    # Format output
    accept = data.get("accept", "application/json")
    response = output_fn(predictions, accept)
    
    return [{"body": response, "content-type": accept}]
