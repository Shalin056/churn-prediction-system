"""
explainability.py

Model explainability using SHAP (SHapley Additive exPlanations).
Provides global and local interpretability for churn predictions.

Author: [Shalin Bhavsar]
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class ChurnExplainer:
    """
    Wrapper for SHAP explainability focused on churn prediction.
    """
    
    def __init__(self, model, X_background: pd.DataFrame = None):
        """
        Initialize explainer with a trained model.
        
        Parameters
        ----------
        model : sklearn Pipeline
            Trained model pipeline
        X_background : pd.DataFrame, optional
            Background dataset for SHAP (uses subset if None)
        """
        self.model = model
        self.explainer = None
        self.X_background = X_background
        self.feature_names = None
        
    def initialize_explainer(self, X_sample: pd.DataFrame = None, max_samples: int = 100):
        """
        Initialize SHAP explainer.
        
        Parameters
        ----------
        X_sample : pd.DataFrame, optional
            Sample data to use as background
        max_samples : int
            Maximum number of background samples
        """
        # Get preprocessed features for background
        if X_sample is not None:
            X_bg = X_sample.sample(min(max_samples, len(X_sample)), random_state=42)
        elif self.X_background is not None:
            X_bg = self.X_background.sample(min(max_samples, len(self.X_background)), random_state=42)
        else:
            raise ValueError("Need either X_sample or X_background")
        
        # Transform through preprocessor
        X_bg_transformed = self.model.named_steps['preprocessor'].transform(X_bg)
        
        # Get feature names
        self.feature_names = self._get_feature_names()
        
        # Initialize SHAP explainer for tree models
        try:
            self.explainer = shap.TreeExplainer(
                self.model.named_steps['model'],
                X_bg_transformed
            )
            print("✅ SHAP TreeExplainer initialized")
        except Exception as e:
            print(f"⚠️ TreeExplainer failed, using KernelExplainer: {e}")
            # Fallback to KernelExplainer (slower but works for any model)
            self.explainer = shap.KernelExplainer(
                lambda x: self.model.named_steps['model'].predict_proba(x)[:, 1],
                X_bg_transformed
            )
            print("✅ SHAP KernelExplainer initialized")
    
    def _get_feature_names(self) -> List[str]:
        """Extract feature names from preprocessor."""
        try:
            preprocessor = self.model.named_steps['preprocessor']
            
            num_features = list(
                preprocessor.named_transformers_['num']
                .named_steps['scaler']
                .get_feature_names_out()
            )
            
            cat_features = list(
                preprocessor.named_transformers_['cat']
                .named_steps['encoder']
                .get_feature_names_out()
            )
            
            return num_features + cat_features
        
        except Exception as e:
            print(f"⚠️ Could not extract feature names: {e}")
            return [f"Feature_{i}" for i in range(100)]  # Placeholder
    
    def explain_instance(
        self,
        X_instance: pd.DataFrame,
        top_n: int = 10
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Explain a single prediction.
        
        Parameters
        ----------
        X_instance : pd.DataFrame
            Single instance to explain (one row)
        top_n : int
            Number of top features to return
            
        Returns
        -------
        tuple
            (shap_values, top_features_list)
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Call initialize_explainer first.")
        
        # Transform instance
        X_transformed = self.model.named_steps['preprocessor'].transform(X_instance)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_transformed)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get positive class
        
        # Get top contributing features
        shap_df = pd.DataFrame({
            'feature': self.feature_names[:len(shap_values[0])],
            'shap_value': shap_values[0],
            'abs_shap': np.abs(shap_values[0])
        })
        
        top_features = shap_df.nlargest(top_n, 'abs_shap')[['feature', 'shap_value']].to_dict('records')
        
        return shap_values, top_features
    
    def get_top_risk_factors(
        self,
        X_instance: pd.DataFrame,
        top_n: int = 3
    ) -> List[Dict[str, any]]:
        """
        Get top risk factors for a customer.
        
        Parameters
        ----------
        X_instance : pd.DataFrame
            Customer data
        top_n : int
            Number of top factors
            
        Returns
        -------
        list
            List of risk factors with impact scores
        """
        shap_values, top_features = self.explain_instance(X_instance, top_n)
        
        risk_factors = []
        for feat in top_features:
            impact = "increases" if feat['shap_value'] > 0 else "decreases"
            risk_factors.append({
                'feature': feat['feature'],
                'impact': impact,
                'magnitude': abs(feat['shap_value']),
                'shap_value': feat['shap_value']
            })
        
        return risk_factors
    
    def plot_waterfall(
        self,
        X_instance: pd.DataFrame,
        save_path: Optional[Path] = None
    ):
        """
        Create waterfall plot for a single prediction.
        
        Parameters
        ----------
        X_instance : pd.DataFrame
            Instance to explain
        save_path : Path, optional
            Path to save plot
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized")
        
        X_transformed = self.model.named_steps['preprocessor'].transform(X_instance)
        shap_values = self.explainer.shap_values(X_transformed)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create SHAP Explanation object
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=self.explainer.expected_value if not isinstance(self.explainer.expected_value, list) else self.explainer.expected_value[1],
            data=X_transformed[0],
            feature_names=self.feature_names[:len(shap_values[0])]
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"💾 Waterfall plot saved to {save_path}")
        
        plt.show()
    
    def plot_global_importance(
        self,
        X_sample: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[Path] = None
    ):
        """
        Create global feature importance plot.
        
        Parameters
        ----------
        X_sample : pd.DataFrame
            Sample of data to explain
        top_n : int
            Number of top features to show
        save_path : Path, optional
            Path to save plot
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized")
        
        X_transformed = self.model.named_steps['preprocessor'].transform(X_sample)
        shap_values = self.explainer.shap_values(X_transformed)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=self.feature_names[:shap_values.shape[1]],
            max_display=top_n,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"💾 Summary plot saved to {save_path}")
        
        plt.show()
    
    def get_feature_importance_df(
        self,
        X_sample: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate mean absolute SHAP values as feature importance.
        
        Parameters
        ----------
        X_sample : pd.DataFrame
            Sample of data
            
        Returns
        -------
        pd.DataFrame
            Feature importance scores
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized")
        
        X_transformed = self.model.named_steps['preprocessor'].transform(X_sample)
        shap_values = self.explainer.shap_values(X_transformed)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP value for each feature
        importance = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df


def generate_explanation_text(risk_factors: List[Dict]) -> str:
    """
    Generate human-readable explanation from risk factors.
    
    Parameters
    ----------
    risk_factors : list
        List of risk factor dictionaries
        
    Returns
    -------
    str
        Human-readable explanation
    """
    if not risk_factors:
        return "No significant risk factors identified."
    
    explanation = "Top reasons for churn prediction:\n"
    
    for i, factor in enumerate(risk_factors, 1):
        feature = factor['feature'].replace('_', ' ').title()
        impact = factor['impact']
        
        explanation += f"{i}. {feature} {impact} churn risk\n"
    
    return explanation


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    from src.preprocessing import load_data, split_data, build_preprocessor
    from src.modeling import create_model_pipeline, train_model
    
    print("="*70)
    print("EXPLAINABILITY MODULE - STANDALONE TEST")
    print("="*70)
    
    # Load and prepare data
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train a simple model
    print("\nTraining model...")
    preprocessor = build_preprocessor()
    pipeline = create_model_pipeline(preprocessor, "random_forest")
    pipeline.fit(X_train, y_train)
    
    # Initialize explainer
    print("\nInitializing explainer...")
    explainer = ChurnExplainer(pipeline)
    explainer.initialize_explainer(X_train, max_samples=100)
    
    # Explain a single prediction
    print("\nExplaining a single instance...")
    test_instance = X_test.iloc[[0]]
    risk_factors = explainer.get_top_risk_factors(test_instance, top_n=5)
    
    print("\nTop risk factors:")
    for factor in risk_factors:
        print(f"  {factor['feature']}: {factor['impact']} "
              f"(magnitude: {factor['magnitude']:.4f})")
    
    # Generate explanation text
    explanation = generate_explanation_text(risk_factors)
    print(f"\n{explanation}")
    
    # Get global feature importance
    print("\nCalculating global feature importance...")
    importance_df = explainer.get_feature_importance_df(X_test.head(100))
    print("\nTop 10 most important features:")
    print(importance_df.head(10).to_string(index=False))
    
    print("\n✅ Explainability module test complete!")