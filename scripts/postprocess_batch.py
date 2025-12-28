"""
Batch Post-processing Script
----------------------------
Post-processes batch inference results.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def postprocess_predictions(input_path: str, output_path: str):
    """
    Post-process batch predictions.
    
    Args:
        input_path: Path to raw predictions
        output_path: Path to save processed results
    """
    logger.info(f"Post-processing predictions from {input_path}")
    
    # Load predictions
    prediction_files = list(Path(input_path).glob("*.out"))
    
    all_results = []
    for file in prediction_files:
        with open(file) as f:
            for line in f:
                result = json.loads(line)
                all_results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Add business logic here
    # Example: Add confidence thresholds, business rules, etc.
    if "predictions" in df.columns:
        df["high_confidence"] = df["predictions"].apply(
            lambda x: x[0]["probability"] > 0.8 if isinstance(x, list) else False
        )
    
    # Save results
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_path) / "processed_predictions.csv"
    df.to_csv(output_file, index=False)
    
    logger.info(f"Processed {len(df)} predictions")
    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    
    postprocess_predictions(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
