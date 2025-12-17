"""
config.py

Central configuration management for the churn prediction system.
All paths, parameters, and settings in one place.

Author: [Shalin Bhavsar]
Date: 2025
"""

from pathlib import Path
from typing import List

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================

RAW_DATA_FILE = RAW_DATA_DIR / "synthetic_churn_data.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_churn_data.csv"

# ============================================================================
# MODEL FILES
# ============================================================================

MODEL_REGISTRY_FILE = MODELS_DIR / "model_registry.json"
BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Target column
TARGET_COLUMN = "Churn"

# ID column (to be dropped)
ID_COLUMN = "CustomerID"

# Numerical features
NUMERICAL_FEATURES: List[str] = [
    "Age",
    "Tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SupportTickets",
    "UsageScore",
]

# Categorical features
CATEGORICAL_FEATURES: List[str] = [
    "Gender",
    "Contract",
    "PaymentMethod",
]

# All feature columns
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================

# Train-test split parameters
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15  # For train-val-test split
RANDOM_STATE = 42
STRATIFY = True

# Imputation strategies
NUMERICAL_IMPUTATION_STRATEGY = "median"
CATEGORICAL_IMPUTATION_STRATEGY = "most_frequent"

# Scaling method
SCALING_METHOD = "standard"  # Options: "standard", "minmax", "robust"

# Encoding method
ENCODING_METHOD = "onehot"  # Options: "onehot", "label", "target"

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Model names
MODELS = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "params": {
            "max_iter": 1000,
            "class_weight": "balanced",
            "C": 0.1,
            "random_state": RANDOM_STATE
        }
    },
    "random_forest": {
        "name": "Random Forest",
        "params": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }
    },
    "xgboost": {
        "name": "XGBoost",
        "params": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 3,
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
            "use_label_encoder": False
        }
    },
    "lightgbm": {
        "name": "LightGBM",
        "params": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1
        }
    }
}

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

# Random search parameters
N_ITER_RANDOM_SEARCH = 20
CV_FOLDS = 3
SCORING_METRIC = "roc_auc"

# Hyperparameter grids
RANDOM_FOREST_PARAM_GRID = {
    "model__n_estimators": [100, 200, 300, 400],
    "model__max_depth": [5, 10, 15, 20, None],
    "model__min_samples_split": [10, 20, 50, 100],
    "model__min_samples_leaf": [5, 10, 20, 50],
    "model__max_features": ["sqrt", "log2", None]
}

XGBOOST_PARAM_GRID = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7, 9],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.7, 0.8, 0.9],
    "model__colsample_bytree": [0.7, 0.8, 0.9],
    "model__scale_pos_weight": [1, 2, 3, 4]
}

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

# Default classification threshold
DEFAULT_THRESHOLD = 0.5

# Threshold optimization range
THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 0.9
THRESHOLD_STEPS = 81

# Metrics to track
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# ============================================================================
# BUSINESS PARAMETERS
# ============================================================================

# Cost-benefit analysis
COST_FALSE_POSITIVE = 50    # Cost of incorrectly targeting non-churner
COST_FALSE_NEGATIVE = 500   # Cost of missing a churner
REVENUE_PER_CUSTOMER = 1500 # Annual revenue per customer
RETENTION_CAMPAIGN_COST = 200  # Cost to run retention campaign

# ============================================================================
# API PARAMETERS
# ============================================================================

API_TITLE = "Customer Churn Prediction API"
API_DESCRIPTION = "Production ML API for real-time churn prediction with explainability"
API_VERSION = "1.0.0"
API_HOST = "0.0.0.0"
API_PORT = 8000

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_path(model_name: str) -> Path:
    """Get path for a specific model file."""
    return MODELS_DIR / f"{model_name}.pkl"

def get_feature_names() -> List[str]:
    """Get all feature column names."""
    return FEATURE_COLUMNS.copy()

def print_config():
    """Print current configuration."""
    print("="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Raw Data: {RAW_DATA_FILE}")
    print(f"Models Dir: {MODELS_DIR}")
    print(f"\nFeatures:")
    print(f"  - Numerical: {len(NUMERICAL_FEATURES)}")
    print(f"  - Categorical: {len(CATEGORICAL_FEATURES)}")
    print(f"  - Total: {len(FEATURE_COLUMNS)}")
    print(f"\nModels Available: {', '.join(MODELS.keys())}")
    print(f"Random State: {RANDOM_STATE}")
    print("="*70)


if __name__ == "__main__":
    # Print configuration when run directly
    print_config()