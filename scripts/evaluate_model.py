"""
Model Evaluation Script
-----------------------
Evaluates trained model on test data and generates metrics report.
"""

import argparse
import json
import logging
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_model(model_path: str) -> xgb.Booster:
    """Extract and load model from tar.gz archive."""
    model_dir = os.path.dirname(model_path)
    
    # Check if it's a tar.gz file
    if model_path.endswith(".tar.gz"):
        with tarfile.open(model_path, "r:gz") as tar:
            tar.extractall(path=model_dir)
        model_file = os.path.join(model_dir, "xgboost-model")
    else:
        model_file = model_path
    
    # Try to find model file
    if not os.path.exists(model_file):
        for name in ["xgboost-model", "model.xgb", "model.bin"]:
            candidate = os.path.join(model_dir, name)
            if os.path.exists(candidate):
                model_file = candidate
                break
    
    model = xgb.Booster()
    model.load_model(model_file)
    logger.info(f"Model loaded from {model_file}")
    
    return model


def load_test_data(test_path: str, target_column: str = "target"):
    """Load test data from path."""
    files = list(Path(test_path).glob("*.csv")) + list(Path(test_path).glob("*.parquet"))
    
    if not files:
        raise ValueError(f"No test data found in {test_path}")
    
    dfs = []
    for f in files:
        if f.suffix == ".csv":
            dfs.append(pd.read_csv(f))
        else:
            dfs.append(pd.read_parquet(f))
    
    df = pd.concat(dfs, ignore_index=True)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    logger.info(f"Loaded test data: {len(df)} samples")
    
    return X, y


def evaluate_model(model: xgb.Booster, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate model and compute metrics."""
    dtest = xgb.DMatrix(X)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Basic metrics
    metrics = {
        "auc": float(roc_auc_score(y, y_pred_proba)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    metrics["confusion_matrix"] = {
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1]),
    }
    
    # ROC curve points
    fpr, tpr, roc_thresholds = roc_curve(y, y_pred_proba)
    metrics["roc_curve"] = {
        "fpr": fpr[::10].tolist(),  # Sample every 10th point
        "tpr": tpr[::10].tolist(),
    }
    
    # Precision-Recall curve points
    precision, recall, pr_thresholds = precision_recall_curve(y, y_pred_proba)
    metrics["pr_curve"] = {
        "precision": precision[::10].tolist(),
        "recall": recall[::10].tolist(),
    }
    
    # Sample count
    metrics["sample_count"] = len(y)
    metrics["positive_rate"] = float(y.mean())
    
    logger.info("Evaluation metrics:")
    for key in ["auc", "accuracy", "precision", "recall", "f1"]:
        logger.info(f"  {key}: {metrics[key]:.4f}")
    
    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--test-path", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/evaluation")
    parser.add_argument("--target-column", type=str, default="target")
    args = parser.parse_args()
    
    # Find model file
    model_dir = args.model_path
    if os.path.isdir(model_dir):
        model_files = list(Path(model_dir).glob("*.tar.gz")) + list(Path(model_dir).glob("xgboost-model"))
        if model_files:
            model_path = str(model_files[0])
        else:
            model_path = os.path.join(model_dir, "model.tar.gz")
    else:
        model_path = model_dir
    
    # Load model
    model = extract_model(model_path)
    
    # Load test data
    X_test, y_test = load_test_data(args.test_path, args.target_column)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Prepare output
    evaluation_report = {
        "metrics": metrics,
        "model_path": model_path,
        "test_path": args.test_path,
    }
    
    # Save report
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, "evaluation.json")
    
    with open(output_file, "w") as f:
        json.dump(evaluation_report, f, indent=2)
    
    logger.info(f"Evaluation report saved to {output_file}")
    
    # Print for SageMaker logging
    print(f"\n[EVALUATION] AUC: {metrics['auc']:.4f}")
    print(f"[EVALUATION] Accuracy: {metrics['accuracy']:.4f}")
    print(f"[EVALUATION] F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
