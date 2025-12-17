"""
modeling.py

Production-grade model training, evaluation, and management.
Trains multiple models and tracks performance metrics.

Author: [Shalin Bhavsar]
Date: 2025
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available")

from src.config import (
    MODELS,
    MODELS_DIR,
    MODEL_REGISTRY_FILE,
    BEST_MODEL_FILE,
    DEFAULT_THRESHOLD,
    RANDOM_STATE
)


def create_model_pipeline(
    preprocessor,
    model_type: str = "random_forest"
) -> Pipeline:
    """
    Create a complete ML pipeline with preprocessor + model.
    
    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted or unfitted preprocessor
    model_type : str
        Type of model: "logistic_regression", "random_forest", "xgboost", "lightgbm"
        
    Returns
    -------
    Pipeline
        Complete ML pipeline
    """
    
    if model_type not in MODELS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Get model configuration
    model_config = MODELS[model_type]
    params = model_config["params"]
    
    # Create model instance
    if model_type == "logistic_regression":
        model = LogisticRegression(**params)
    
    elif model_type == "random_forest":
        model = RandomForestClassifier(**params)
    
    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed")
        model = XGBClassifier(**params)
    
    elif model_type == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed")
        model = LGBMClassifier(**params)
    
    else:
        raise ValueError(f"Model type not implemented: {model_type}")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    return pipeline


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    verbose: bool = True
) -> Pipeline:
    """
    Train a model pipeline.
    
    Parameters
    ----------
    pipeline : Pipeline
        ML pipeline to train
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    verbose : bool
        Whether to print training info
        
    Returns
    -------
    Pipeline
        Trained pipeline
    """
    if verbose:
        print(f"Training {pipeline.named_steps['model'].__class__.__name__}...")
    
    pipeline.fit(X_train, y_train)
    
    if verbose:
        print("✅ Training complete")
    
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model performance on test data.
    
    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    threshold : float
        Classification threshold
    verbose : bool
        Whether to print results
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    # Get predictions
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "threshold": threshold
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print("MODEL EVALUATION")
        print('='*70)
        print(f"Accuracy:   {metrics['accuracy']:.4f}")
        print(f"Precision:  {metrics['precision']:.4f}")
        print(f"Recall:     {metrics['recall']:.4f}")
        print(f"F1-Score:   {metrics['f1']:.4f}")
        print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
        print(f"Threshold:  {metrics['threshold']:.2f}")
        print('='*70)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"TN: {cm[0,0]:>6,}  |  FP: {cm[0,1]:>6,}")
        print(f"FN: {cm[1,0]:>6,}  |  TP: {cm[1,1]:>6,}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    return metrics


def optimize_threshold(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = "f1",
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    n_thresholds: int = 81
) -> Tuple[float, float]:
    """
    Find optimal classification threshold by maximizing a metric.
    
    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation labels
    metric : str
        Metric to optimize: "f1", "precision", "recall"
    threshold_range : tuple
        (min, max) threshold values to test
    n_thresholds : int
        Number of thresholds to test
        
    Returns
    -------
    tuple
        (best_threshold, best_metric_value)
    """
    # Get predicted probabilities
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    
    # Test different thresholds
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == "f1":
            score = f1_score(y_val, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_val, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_val, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    # Find best threshold
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    print(f"\n🎯 Optimal threshold: {best_threshold:.3f}")
    print(f"   {metric.upper()}: {best_score:.4f}")
    print(f"   Improvement from 0.5: {((best_score / scores[40]) - 1) * 100:+.1f}%")
    
    return best_threshold, best_score


def save_model(
    pipeline: Pipeline,
    model_name: str,
    metrics: Dict[str, float],
    metadata: Dict[str, Any] = None
) -> Path:
    """
    Save trained model and update model registry.
    
    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline
    model_name : str
        Name for the model file
    metrics : dict
        Performance metrics
    metadata : dict, optional
        Additional metadata
        
    Returns
    -------
    Path
        Path to saved model file
    """
    # Create models directory if needed
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(pipeline, model_path)
    print(f"💾 Model saved: {model_path}")
    
    # Update model registry
    registry = load_model_registry()
    
    registry[model_name] = {
        "model_path": str(model_path),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    
    save_model_registry(registry)
    
    return model_path


def load_model(model_name: str = None, model_path: Path = None) -> Pipeline:
    """
    Load a saved model.
    
    Parameters
    ----------
    model_name : str, optional
        Name of model in registry
    model_path : Path, optional
        Direct path to model file
        
    Returns
    -------
    Pipeline
        Loaded model pipeline
    """
    if model_path is None:
        if model_name is None:
            raise ValueError("Must provide either model_name or model_path")
        
        model_path = MODELS_DIR / f"{model_name}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    pipeline = joblib.load(model_path)
    print(f"📦 Model loaded: {model_path}")
    
    return pipeline


def load_model_registry() -> Dict:
    """Load model registry from JSON file."""
    if MODEL_REGISTRY_FILE.exists():
        with open(MODEL_REGISTRY_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_model_registry(registry: Dict) -> None:
    """Save model registry to JSON file."""
    with open(MODEL_REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)


def get_best_model() -> Tuple[str, Dict]:
    """
    Get the best performing model from registry.
    
    Returns
    -------
    tuple
        (model_name, model_info)
    """
    registry = load_model_registry()
    
    if not registry:
        raise ValueError("No models in registry")
    
    # Find model with highest ROC-AUC
    best_name = max(
        registry.keys(),
        key=lambda k: registry[k]["metrics"].get("roc_auc", 0)
    )
    
    return best_name, registry[best_name]


def compare_models(registry: Dict = None) -> pd.DataFrame:
    """
    Compare all models in registry.
    
    Parameters
    ----------
    registry : dict, optional
        Model registry (loads from file if not provided)
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    if registry is None:
        registry = load_model_registry()
    
    if not registry:
        print("No models in registry")
        return pd.DataFrame()
    
    # Extract metrics for each model
    comparison = []
    for name, info in registry.items():
        metrics = info["metrics"]
        metrics["model_name"] = name
        metrics["timestamp"] = info["timestamp"]
        comparison.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(comparison)
    df = df.sort_values("roc_auc", ascending=False)
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    from src.preprocessing import load_data, split_data, build_preprocessor
    
    print("="*70)
    print("MODELING MODULE - STANDALONE TEST")
    print("="*70)
    
    # Load and split data
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Build preprocessor
    preprocessor = build_preprocessor()
    
    # Train multiple models
    model_types = ["logistic_regression", "random_forest"]
    
    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Training {model_type}")
        print('='*70)
        
        # Create and train pipeline
        pipeline = create_model_pipeline(preprocessor, model_type)
        pipeline = train_model(pipeline, X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(pipeline, X_test, y_test)
        
        # Save model
        save_model(pipeline, model_type, metrics)
    
    # Compare all models
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print('='*70)
    comparison = compare_models()
    print(comparison[["model_name", "accuracy", "f1", "roc_auc"]].to_string(index=False))
    
    # Get best model
    best_name, best_info = get_best_model()
    print(f"\n🏆 Best model: {best_name}")
    print(f"   ROC-AUC: {best_info['metrics']['roc_auc']:.4f}")
    
    print("\n✅ Modeling module test complete!")