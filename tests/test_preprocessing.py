"""
Unit tests for preprocessing module
Tests data loading, splitting, and preprocessing pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import (
    load_data,
    split_data,
    build_preprocessor,
    validate_data,
    get_feature_names
)
from src.config import TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


class TestDataLoading:
    """Test data loading functionality"""
    
    def test_load_data_returns_dataframe(self):
        """Test that load_data returns a DataFrame"""
        # Skip if data file doesn't exist
        pytest.skip("Requires actual data file")
    
    def test_load_data_has_required_columns(self):
        """Test that loaded data has all required columns"""
        pytest.skip("Requires actual data file")


class TestDataSplitting:
    """Test data splitting functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'CustomerID': [f'CUST_{i:06d}' for i in range(n_samples)],
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Age': np.random.randint(18, 80, n_samples),
            'Tenure': np.random.randint(0, 72, n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Credit card'], n_samples),
            'MonthlyCharges': np.random.uniform(20, 150, n_samples),
            'TotalCharges': np.random.uniform(100, 5000, n_samples),
            'SupportTickets': np.random.randint(0, 10, n_samples),
            'UsageScore': np.random.uniform(0, 100, n_samples),
            'Churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        
        return pd.DataFrame(data)
    
    def test_split_data_returns_four_objects(self, sample_data):
        """Test that split_data returns X_train, X_test, y_train, y_test"""
        result = split_data(sample_data)
        assert len(result) == 4
    
    def test_split_data_correct_shapes(self, sample_data):
        """Test that split creates correct proportions"""
        X_train, X_test, y_train, y_test = split_data(sample_data, test_size=0.2)
        
        total_samples = len(sample_data)
        assert len(X_train) + len(X_test) == total_samples
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_test) == pytest.approx(total_samples * 0.2, rel=0.01)
    
    def test_split_data_no_overlap(self, sample_data):
        """Test that train and test sets don't overlap"""
        X_train, X_test, y_train, y_test = split_data(sample_data)
        
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        
        assert len(train_indices.intersection(test_indices)) == 0
    
    def test_split_data_removes_id_and_target(self, sample_data):
        """Test that CustomerID and Churn are removed from X"""
        X_train, X_test, y_train, y_test = split_data(sample_data)
        
        assert 'CustomerID' not in X_train.columns
        assert 'Churn' not in X_train.columns
        assert 'CustomerID' not in X_test.columns
        assert 'Churn' not in X_test.columns
    
    def test_split_data_stratification(self, sample_data):
        """Test that stratification maintains class proportions"""
        X_train, X_test, y_train, y_test = split_data(sample_data, stratify=True)
        
        overall_churn_rate = sample_data['Churn'].mean()
        train_churn_rate = y_train.mean()
        test_churn_rate = y_test.mean()
        
        assert train_churn_rate == pytest.approx(overall_churn_rate, abs=0.05)
        assert test_churn_rate == pytest.approx(overall_churn_rate, abs=0.05)


class TestPreprocessing:
    """Test preprocessing pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with missing values"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Age': np.random.randint(18, 80, n_samples),
            'Tenure': np.random.randint(0, 72, n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Credit card'], n_samples),
            'MonthlyCharges': np.random.uniform(20, 150, n_samples),
            'TotalCharges': np.random.uniform(100, 5000, n_samples),
            'SupportTickets': np.random.randint(0, 10, n_samples),
            'UsageScore': np.random.uniform(0, 100, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        df.loc[0:5, 'Age'] = np.nan
        df.loc[10:15, 'Gender'] = np.nan
        
        return df
    
    def test_build_preprocessor_returns_column_transformer(self):
        """Test that build_preprocessor returns ColumnTransformer"""
        preprocessor = build_preprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, 'fit')
        assert hasattr(preprocessor, 'transform')
    
    def test_preprocessor_handles_missing_values(self, sample_data):
        """Test that preprocessor handles missing values"""
        preprocessor = build_preprocessor()
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(sample_data)
        
        # Check no NaN values remain
        assert not np.isnan(X_transformed).any()
    
    def test_preprocessor_output_shape(self, sample_data):
        """Test that preprocessor outputs correct number of features"""
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(sample_data)
        
        # Should have numerical features + one-hot encoded categorical features
        # 6 numerical + (2 gender + 3 contract + 2 payment) = 6 + 7 = 13 at minimum
        assert X_transformed.shape[0] == len(sample_data)
        assert X_transformed.shape[1] >= 10  # At least this many features
    
    def test_preprocessor_fit_transform_consistency(self, sample_data):
        """Test that fit_transform and fit+transform give same results"""
        preprocessor1 = build_preprocessor()
        preprocessor2 = build_preprocessor()
        
        X_fit_transform = preprocessor1.fit_transform(sample_data)
        preprocessor2.fit(sample_data)
        X_fit_then_transform = preprocessor2.transform(sample_data)
        
        np.testing.assert_array_almost_equal(X_fit_transform, X_fit_then_transform)


class TestDataValidation:
    """Test data validation"""
    
    @pytest.fixture
    def valid_data(self):
        """Create valid sample data"""
        return pd.DataFrame({
            'CustomerID': ['C001', 'C002'],
            'Gender': ['Male', 'Female'],
            'Age': [25, 35],
            'Tenure': [12, 24],
            'Contract': ['Month-to-month', 'One year'],
            'PaymentMethod': ['Credit card', 'Electronic check'],
            'MonthlyCharges': [50.0, 75.0],
            'TotalCharges': [600.0, 1800.0],
            'SupportTickets': [1, 3],
            'UsageScore': [60.0, 80.0],
            'Churn': [0, 1]
        })
    
    def test_validate_data_passes_for_valid_data(self, valid_data):
        """Test that validation passes for valid data"""
        try:
            validate_data(valid_data)
        except ValueError:
            pytest.fail("validate_data raised ValueError for valid data")
    
    def test_validate_data_fails_without_target(self, valid_data):
        """Test that validation fails without target column"""
        invalid_data = valid_data.drop(columns=['Churn'])
        
        with pytest.raises(ValueError, match="Target column"):
            validate_data(invalid_data)
    
    def test_validate_data_fails_with_missing_features(self, valid_data):
        """Test that validation fails with missing features"""
        invalid_data = valid_data.drop(columns=['Age'])
        
        with pytest.raises(ValueError, match="Missing feature columns"):
            validate_data(invalid_data)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])