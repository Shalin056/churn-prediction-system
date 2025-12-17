"""
preprocessing.py

Production-grade data preprocessing with no data leakage.
All transformations happen INSIDE pipelines, fitted only on training data.

Author: [Shalin Bhavsar]
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import (
    RAW_DATA_FILE,
    TARGET_COLUMN,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    NUMERICAL_IMPUTATION_STRATEGY,
    CATEGORICAL_IMPUTATION_STRATEGY,
    SCALING_METHOD,
    ENCODING_METHOD
)


def load_data(filepath: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Parameters
    ----------
    filepath : Path
        Path to the CSV file
        
    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✅ Loaded data: {df.shape}")
    
    # Convert TotalCharges to numeric (handle any string values)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate that dataset has required columns and correct types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate
        
    Raises
    ------
    ValueError
        If validation fails
    """
    # Check target column exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found")
    
    # Check feature columns exist
    missing_features = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    print("✅ Data validation passed")


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    CRITICAL: This must happen BEFORE any preprocessing to avoid data leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    test_size : float
        Proportion of data for test set
    random_state : int
        Random seed for reproducibility
    stratify : bool
        Whether to stratify split by target variable
        
    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    # Separate features and target
    X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )
    
    print(f"✅ Data split complete:")
    print(f"   Train: {X_train.shape}")
    print(f"   Test:  {X_test.shape}")
    print(f"   Train churn rate: {y_train.mean():.2%}")
    print(f"   Test churn rate:  {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


def build_preprocessor(
    scaling_method: str = SCALING_METHOD,
    encoding_method: str = ENCODING_METHOD
) -> ColumnTransformer:
    """
    Build preprocessing pipeline for numerical and categorical features.
    
    CRITICAL: All transformations are fitted ONLY on training data.
    The pipeline learns statistics (mean, median, categories) from training data
    and applies those same transformations to validation/test data.
    
    Parameters
    ----------
    scaling_method : str
        Scaling method: "standard", "minmax", or "robust"
    encoding_method : str
        Encoding method: "onehot" or "label"
        
    Returns
    -------
    ColumnTransformer
        Complete preprocessing pipeline
    """
    
    # Choose scaler
    if scaling_method == "standard":
        scaler = StandardScaler()
    elif scaling_method == "minmax":
        scaler = MinMaxScaler()
    elif scaling_method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")
    
    # Numerical pipeline: impute → scale
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=NUMERICAL_IMPUTATION_STRATEGY)),
        ("scaler", scaler)
    ])
    
    # Choose encoder
    if encoding_method == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif encoding_method == "label":
        encoder = LabelEncoder()
    else:
        raise ValueError(f"Unknown encoding method: {encoding_method}")
    
    # Categorical pipeline: impute → encode
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=CATEGORICAL_IMPUTATION_STRATEGY)),
        ("encoder", encoder)
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"  # Drop any other columns
    )
    
    print("✅ Preprocessor built:")
    print(f"   - Numerical features: {len(NUMERICAL_FEATURES)}")
    print(f"   - Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"   - Scaling: {scaling_method}")
    print(f"   - Encoding: {encoding_method}")
    
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Extract feature names after preprocessing.
    
    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted preprocessor
        
    Returns
    -------
    list
        List of feature names
    """
    try:
        # Get numerical feature names
        num_features = list(
            preprocessor.named_transformers_['num']
            .named_steps['scaler']
            .get_feature_names_out()
        )
        
        # Get categorical feature names
        cat_features = list(
            preprocessor.named_transformers_['cat']
            .named_steps['encoder']
            .get_feature_names_out()
        )
        
        return num_features + cat_features
    
    except Exception as e:
        print(f"⚠️ Could not extract feature names: {e}")
        return []


def preprocess_new_data(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    include_target: bool = False
) -> pd.DataFrame:
    """
    Preprocess new data using a fitted preprocessor.
    Used for making predictions on new customers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw input data
    preprocessor : ColumnTransformer
        Fitted preprocessor
    include_target : bool
        Whether to include target column in output
        
    Returns
    -------
    pd.DataFrame
        Preprocessed data
    """
    # Drop ID column if present
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])
    
    # Separate target if present
    if TARGET_COLUMN in df.columns and not include_target:
        df = df.drop(columns=[TARGET_COLUMN])
    
    # Transform data
    X_transformed = preprocessor.transform(df)
    
    # Convert to DataFrame
    feature_names = get_feature_names(preprocessor)
    if feature_names:
        return pd.DataFrame(X_transformed, columns=feature_names, index=df.index)
    else:
        return pd.DataFrame(X_transformed, index=df.index)


def check_data_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """
    Check for potential data leakage between train and test sets.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    """
    # Check for overlapping indices
    overlap = set(X_train.index) & set(X_test.index)
    if overlap:
        print(f"⚠️ WARNING: {len(overlap)} overlapping indices found!")
    else:
        print("✅ No overlapping indices - good!")
    
    # Check feature distributions
    for col in X_train.select_dtypes(include=[np.number]).columns:
        train_mean = X_train[col].mean()
        test_mean = X_test[col].mean()
        diff = abs(train_mean - test_mean) / train_mean if train_mean != 0 else 0
        
        if diff > 0.5:  # More than 50% difference
            print(f"⚠️ WARNING: Large difference in {col}: {diff:.1%}")


def print_preprocessing_summary(X_train, X_test, y_train, y_test):
    """
    Print summary of preprocessing results.
    
    Parameters
    ----------
    X_train, X_test : array-like
        Preprocessed features
    y_train, y_test : array-like
        Target variables
    """
    print("\n" + "="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)
    print(f"Training set:   {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"Test set:       {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
    print(f"\nClass distribution (training):")
    print(f"  No Churn (0): {(y_train == 0).sum():,} ({(y_train == 0).mean():.1%})")
    print(f"  Churn (1):    {(y_train == 1).sum():,} ({(y_train == 1).mean():.1%})")
    print(f"\nClass imbalance ratio: {(y_train == 0).sum() / (y_train == 1).sum():.2f}:1")
    print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PREPROCESSING MODULE - STANDALONE TEST")
    print("="*70)
    
    # Load data
    df = load_data()
    
    # Validate data
    validate_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Check for data leakage
    check_data_leakage(X_train, X_test)
    
    # Build preprocessor
    preprocessor = build_preprocessor()
    
    # Fit and transform
    print("\nFitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    print("Transforming test data...")
    X_test_processed = preprocessor.transform(X_test)
    
    # Print summary
    print_preprocessing_summary(X_train_processed, X_test_processed, y_train, y_test)
    
    # Get feature names
    feature_names = get_feature_names(preprocessor)
    print(f"Total features after preprocessing: {len(feature_names)}")
    print(f"First 10 features: {feature_names[:10]}")
    
    print("\n✅ Preprocessing module test complete!")