"""
SageMaker Training Script
-------------------------
XGBoost training script compatible with SageMaker Script Mode.
Handles hyperparameter parsing, data loading, training, and model saving.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse hyperparameters from SageMaker."""
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--min_child_weight", type=float, default=1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--eval_metric", type=str, default="auc")
    parser.add_argument("--early_stopping_rounds", type=int, default=10)

    # Data configuration
    parser.add_argument("--target_column", type=str, default="target")

    # SageMaker specific arguments
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    return parser.parse_args()


def load_data(data_path: str, target_column: str) -> tuple:
    """Load data from the specified path."""
    logger.info(f"Loading data from {data_path}")
    
    # Support both CSV and Parquet
    files = list(Path(data_path).glob("*.csv")) + list(Path(data_path).glob("*.parquet"))
    
    if not files:
        raise ValueError(f"No data files found in {data_path}")
    
    dfs = []
    for file in files:
        if file.suffix == ".csv":
            dfs.append(pd.read_csv(file))
        elif file.suffix == ".parquet":
            dfs.append(pd.read_parquet(file))
    
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Split features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y


def train_model(args, dtrain: xgb.DMatrix, dval: xgb.DMatrix) -> xgb.Booster:
    """Train XGBoost model with given parameters."""
    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "objective": args.objective,
        "eval_metric": args.eval_metric,
        "verbosity": 1,
    }
    
    logger.info(f"Training with parameters: {params}")
    
    evals = [(dtrain, "train"), (dval, "validation")]
    
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=evals,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=10,
    )
    
    return model


def evaluate_model(model: xgb.Booster, dval: xgb.DMatrix, y_val: np.ndarray) -> dict:
    """Evaluate model performance."""
    # Get predictions
    y_pred_proba = model.predict(dval)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        "auc": roc_auc_score(y_val, y_pred_proba),
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
    }
    
    logger.info("Model evaluation metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return metrics


def save_model(model: xgb.Booster, model_dir: str, metrics: dict):
    """Save model and metrics."""
    model_path = os.path.join(model_dir, "xgboost-model")
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save feature importance
    importance = model.get_score(importance_type="gain")
    importance_path = os.path.join(model_dir, "feature_importance.json")
    with open(importance_path, "w") as f:
        json.dump(importance, f, indent=2)
    logger.info(f"Feature importance saved to {importance_path}")


def main():
    """Main training function."""
    logger.info("Starting training job")
    args = parse_args()
    
    # Load training data
    X_train, y_train = load_data(args.train, args.target_column)
    
    # Load validation data
    X_val, y_val = load_data(args.validation, args.target_column)
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train model
    model = train_model(args, dtrain, dval)
    
    # Evaluate model
    metrics = evaluate_model(model, dval, y_val.values)
    
    # Save model and artifacts
    save_model(model, args.model_dir, metrics)
    
    # Log final metrics for SageMaker
    print(f"\n[METRICS] AUC={metrics['auc']:.4f}")
    print(f"[METRICS] Accuracy={metrics['accuracy']:.4f}")
    print(f"[METRICS] F1={metrics['f1']:.4f}")
    
    logger.info("Training job completed successfully")


if __name__ == "__main__":
    main()
