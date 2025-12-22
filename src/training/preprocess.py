"""
Data Preprocessing Script
-------------------------
Handles data preprocessing for SageMaker training pipeline.
Includes feature engineering, data validation, and train/val/test split.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing pipeline for ML training."""
    
    def __init__(
        self,
        target_column: str,
        categorical_columns: Optional[List[str]] = None,
        handle_missing: str = "median",
        handle_outliers: str = "clip",
        outlier_std_threshold: float = 3.0,
        normalize: bool = True,
        normalization_method: str = "standard",
    ):
        self.target_column = target_column
        self.categorical_columns = categorical_columns or []
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.outlier_std_threshold = outlier_std_threshold
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        self.scaler = None
        self.feature_stats = {}
        self.encodings = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and transform data."""
        logger.info("Starting preprocessing pipeline")
        
        # Validate data
        self._validate_data(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Handle outliers
        df = self._handle_outliers_values(df)
        
        # Encode categorical variables
        df = self._encode_categorical(df)
        
        # Normalize numerical features
        if self.normalize:
            df = self._normalize_features(df)
        
        logger.info(f"Preprocessing complete. Shape: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        df = self._handle_missing_values(df, fit=False)
        df = self._handle_outliers_values(df, fit=False)
        df = self._encode_categorical(df, fit=False)
        
        if self.normalize and self.scaler is not None:
            df = self._normalize_features(df, fit=False)
        
        return df

    def _validate_data(self, df: pd.DataFrame):
        """Validate input data."""
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
        
        logger.info(f"Data validation passed. Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")

    def _handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df = df.copy()
        
        for col in df.columns:
            if col == self.target_column:
                continue
                
            if df[col].isnull().any():
                if fit:
                    if self.handle_missing == "median":
                        fill_value = df[col].median()
                    elif self.handle_missing == "mean":
                        fill_value = df[col].mean()
                    elif self.handle_missing == "zero":
                        fill_value = 0
                    elif self.handle_missing == "drop":
                        df = df.dropna(subset=[col])
                        continue
                    else:
                        fill_value = df[col].median()
                    
                    self.feature_stats[f"{col}_fill"] = fill_value
                else:
                    fill_value = self.feature_stats.get(f"{col}_fill", 0)
                
                df[col] = df[col].fillna(fill_value)
                logger.info(f"Filled {col} missing values with {fill_value}")
        
        return df

    def _handle_outliers_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle outliers in numerical columns."""
        if self.handle_outliers == "none":
            return df
        
        df = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [c for c in numerical_cols if c != self.target_column]
        
        for col in numerical_cols:
            if fit:
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - self.outlier_std_threshold * std
                upper = mean + self.outlier_std_threshold * std
                self.feature_stats[f"{col}_lower"] = lower
                self.feature_stats[f"{col}_upper"] = upper
            else:
                lower = self.feature_stats.get(f"{col}_lower", df[col].min())
                upper = self.feature_stats.get(f"{col}_upper", df[col].max())
            
            if self.handle_outliers == "clip":
                df[col] = df[col].clip(lower, upper)
            elif self.handle_outliers == "remove":
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        
        return df

    def _encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            
            if fit:
                # Create label encoding
                unique_values = df[col].unique()
                self.encodings[col] = {v: i for i, v in enumerate(unique_values)}
            
            encoding = self.encodings.get(col, {})
            df[col] = df[col].map(encoding).fillna(-1)
        
        return df

    def _normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize numerical features."""
        df = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [c for c in numerical_cols if c != self.target_column]
        
        if not numerical_cols:
            return df
        
        if fit:
            if self.normalization_method == "standard":
                self.scaler = StandardScaler()
            elif self.normalization_method == "minmax":
                self.scaler = MinMaxScaler()
            elif self.normalization_method == "robust":
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            if self.scaler is not None:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df

    def save(self, path: str):
        """Save preprocessor state."""
        import pickle
        state = {
            "feature_stats": self.feature_stats,
            "encodings": self.encodings,
            "scaler": self.scaler,
            "config": {
                "target_column": self.target_column,
                "categorical_columns": self.categorical_columns,
                "handle_missing": self.handle_missing,
                "handle_outliers": self.handle_outliers,
                "outlier_std_threshold": self.outlier_std_threshold,
                "normalize": self.normalize,
                "normalization_method": self.normalization_method,
            }
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        """Load preprocessor from file."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        preprocessor = cls(**state["config"])
        preprocessor.feature_stats = state["feature_stats"]
        preprocessor.encodings = state["encodings"]
        preprocessor.scaler = state["scaler"]
        return preprocessor


def split_data(
    df: pd.DataFrame,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    target_column: str = "target",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets."""
    assert abs(train_split + val_split + test_split - 1.0) < 0.001, "Splits must sum to 1"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_split,
        random_state=random_state,
        stratify=df[target_column] if target_column in df.columns else None,
    )
    
    # Second split: val vs test
    relative_val_size = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        random_state=random_state,
        stratify=temp_df[target_column] if target_column in temp_df.columns else None,
    )
    
    logger.info(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def main():
    """Main preprocessing function for SageMaker Processing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--target-column", type=str, default="target")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    args = parser.parse_args()
    
    # Load data
    input_files = list(Path(args.input_path).glob("*.csv")) + list(Path(args.input_path).glob("*.parquet"))
    dfs = []
    for f in input_files:
        if f.suffix == ".csv":
            dfs.append(pd.read_csv(f))
        else:
            dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    
    logger.info(f"Loaded data with shape: {df.shape}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(target_column=args.target_column)
    
    # Preprocess data
    df_processed = preprocessor.fit_transform(df)
    
    # Split data
    train_df, val_df, test_df = split_data(
        df_processed,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        target_column=args.target_column,
    )
    
    # Save outputs
    os.makedirs(args.output_path, exist_ok=True)
    train_df.to_csv(os.path.join(args.output_path, "train", "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_path, "validation", "validation.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_path, "test", "test.csv"), index=False)
    
    # Save preprocessor
    preprocessor.save(os.path.join(args.output_path, "preprocessor.pkl"))
    
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
