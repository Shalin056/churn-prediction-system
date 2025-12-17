"""
feature_engineering.py

Advanced feature engineering to improve model performance.
Creates derived features, interactions, and aggregations.

Author: [Shalin Bhavsar]
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for creating advanced churn prediction features.
    Implements sklearn's transformer interface for use in pipelines.
    """
    
    def __init__(self, create_interactions: bool = True):
        """
        Parameters
        ----------
        create_interactions : bool
            Whether to create interaction features
        """
        self.create_interactions = create_interactions
        self.fitted_ = False
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for this transformer).
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : pd.Series, optional
            Target variable (not used)
            
        Returns
        -------
        self
        """
        self.fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data by adding engineered features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
            
        Returns
        -------
        pd.DataFrame
            Features with engineered columns added
        """
        if not self.fitted_:
            raise RuntimeError("Transformer must be fitted before transform")
        
        # Make a copy to avoid modifying original
        X_new = X.copy()
        
        # Add all engineered features
        X_new = self._add_ratio_features(X_new)
        X_new = self._add_tenure_features(X_new)
        X_new = self._add_engagement_features(X_new)
        X_new = self._add_risk_indicators(X_new)
        
        if self.create_interactions:
            X_new = self._add_interaction_features(X_new)
        
        return X_new
    
    def _add_ratio_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio and derived financial features.
        """
        # Monthly charges relative to tenure
        X['ChargesPerMonth'] = X['TotalCharges'] / (X['Tenure'] + 1)
        
        # Charges relative to usage
        X['ChargesPerUsagePoint'] = X['MonthlyCharges'] / (X['UsageScore'] + 1)
        
        # Support tickets per month
        X['TicketsPerMonth'] = X['SupportTickets'] / (X['Tenure'] + 1)
        
        return X
    
    def _add_tenure_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure-based features.
        """
        # Tenure categories
        X['IsNewCustomer'] = (X['Tenure'] <= 6).astype(int)
        X['IsShortTenure'] = (X['Tenure'] <= 12).astype(int)
        X['IsMediumTenure'] = ((X['Tenure'] > 12) & (X['Tenure'] <= 24)).astype(int)
        X['IsLongTenure'] = (X['Tenure'] > 24).astype(int)
        
        # Tenure in years
        X['TenureYears'] = X['Tenure'] / 12
        
        return X
    
    def _add_engagement_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create engagement and usage features.
        """
        # Usage categories
        X['IsLowUsage'] = (X['UsageScore'] < 30).astype(int)
        X['IsHighUsage'] = (X['UsageScore'] > 70).astype(int)
        
        # Engagement score (composite)
        X['EngagementScore'] = (
            X['UsageScore'] * 0.6 +
            (100 - X['SupportTickets'] * 10).clip(0, 100) * 0.4
        )
        
        return X
    
    def _add_risk_indicators(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk indicator features.
        """
        # High monthly charges
        X['IsHighCharges'] = (X['MonthlyCharges'] > 100).astype(int)
        
        # Many support tickets
        X['HasManyTickets'] = (X['SupportTickets'] >= 3).astype(int)
        
        # New expensive customer (high risk combination)
        X['IsNewExpensive'] = (
            (X['Tenure'] <= 12) & (X['MonthlyCharges'] > 80)
        ).astype(int)
        
        # Low engagement with high charges (risky)
        X['IsDisengagedExpensive'] = (
            (X['UsageScore'] < 40) & (X['MonthlyCharges'] > 80)
        ).astype(int)
        
        return X
    
    def _add_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        """
        # Tenure × Monthly Charges
        X['Tenure_MonthlyCharges'] = X['Tenure'] * X['MonthlyCharges']
        
        # Usage × Monthly Charges
        X['Usage_MonthlyCharges'] = X['UsageScore'] * X['MonthlyCharges']
        
        # Tenure × Usage
        X['Tenure_Usage'] = X['Tenure'] * X['UsageScore']
        
        # Support Tickets × Monthly Charges
        X['Tickets_MonthlyCharges'] = X['SupportTickets'] * X['MonthlyCharges']
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """
        Get names of output features.
        
        Returns
        -------
        list
            List of feature names
        """
        # This would need to be properly implemented for sklearn compatibility
        return None


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add all engineered features to a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with base features
        
    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features added
    """
    engineer = ChurnFeatureEngineer(create_interactions=True)
    engineer.fit(df)
    return engineer.transform(df)


def get_feature_importance_groups() -> dict:
    """
    Group features by type for analysis.
    
    Returns
    -------
    dict
        Dictionary mapping group names to feature lists
    """
    return {
        "base_numerical": [
            "Age", "Tenure", "MonthlyCharges", "TotalCharges",
            "SupportTickets", "UsageScore"
        ],
        "base_categorical": [
            "Gender", "Contract", "PaymentMethod"
        ],
        "ratio_features": [
            "ChargesPerMonth", "ChargesPerUsagePoint", "TicketsPerMonth"
        ],
        "tenure_features": [
            "IsNewCustomer", "IsShortTenure", "IsMediumTenure",
            "IsLongTenure", "TenureYears"
        ],
        "engagement_features": [
            "IsLowUsage", "IsHighUsage", "EngagementScore"
        ],
        "risk_indicators": [
            "IsHighCharges", "HasManyTickets", "IsNewExpensive",
            "IsDisengagedExpensive"
        ],
        "interactions": [
            "Tenure_MonthlyCharges", "Usage_MonthlyCharges",
            "Tenure_Usage", "Tickets_MonthlyCharges"
        ]
    }


def analyze_feature_groups(
    X: pd.DataFrame,
    feature_importances: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Analyze feature importance by feature groups.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe
    feature_importances : np.ndarray
        Feature importance values from model
    feature_names : list
        List of feature names
        
    Returns
    -------
    pd.DataFrame
        Group-level importance summary
    """
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })
    
    # Get feature groups
    groups = get_feature_importance_groups()
    
    # Calculate group importance
    group_importance = []
    for group_name, group_features in groups.items():
        # Find matching features (handles encoded categorical features)
        matching = [f for f in feature_names if any(gf in f for gf in group_features)]
        group_imp = importance_df[importance_df['feature'].isin(matching)]['importance'].sum()
        
        group_importance.append({
            'group': group_name,
            'total_importance': group_imp,
            'n_features': len(matching)
        })
    
    result = pd.DataFrame(group_importance)
    result = result.sort_values('total_importance', ascending=False)
    result['importance_pct'] = (result['total_importance'] / result['total_importance'].sum() * 100)
    
    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    from src.preprocessing import load_data
    
    print("="*70)
    print("FEATURE ENGINEERING MODULE - STANDALONE TEST")
    print("="*70)
    
    # Load data
    df = load_data()
    
    print(f"\nOriginal features: {df.shape[1]}")
    print(f"Original shape: {df.shape}")
    
    # Add engineered features
    print("\nAdding engineered features...")
    df_engineered = add_all_features(df)
    
    print(f"\nAfter feature engineering:")
    print(f"Total features: {df_engineered.shape[1]}")
    print(f"New features added: {df_engineered.shape[1] - df.shape[1]}")
    print(f"Shape: {df_engineered.shape}")
    
    # Show new features
    original_cols = set(df.columns)
    new_cols = [col for col in df_engineered.columns if col not in original_cols]
    
    print(f"\nNew features created ({len(new_cols)}):")
    for col in sorted(new_cols):
        print(f"  - {col}")
    
    # Show sample statistics
    print("\nSample statistics for new features:")
    print(df_engineered[new_cols[:5]].describe())
    
    # Show feature groups
    groups = get_feature_importance_groups()
    print(f"\nFeature groups defined:")
    for group_name, features in groups.items():
        print(f"  {group_name}: {len(features)} features")
    
    print("\n✅ Feature engineering module test complete!")