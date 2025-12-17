"""
gcp_integration.py

Integration with Google Cloud Platform services:
- BigQuery for prediction logging
- Cloud Storage for model versioning
- Cloud Monitoring for metrics

Author: [Shalin Bhavsar]
Date: 2025
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
import json
import uuid

try:
    from google.cloud import bigquery
    from google.cloud import storage
    from google.cloud import monitoring_v3
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    print("⚠️ Google Cloud SDK not installed. GCP features disabled.")


class BigQueryLogger:
    """Log predictions to BigQuery for analytics"""
    
    def __init__(self, project_id: str, dataset_id: str = "churn_predictions"):
        """
        Initialize BigQuery logger
        
        Parameters
        ----------
        project_id : str
            GCP project ID
        dataset_id : str
            BigQuery dataset ID
        """
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("google-cloud-bigquery not installed")
        
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = "predictions"
        
        self.client = bigquery.Client(project=project_id)
        self.table_ref = f"{project_id}.{dataset_id}.{self.table_id}"
    
    def log_prediction(
        self,
        customer_id: Optional[str],
        churn_probability: float,
        churn_prediction: int,
        risk_level: str,
        model_version: str = "v1.0"
    ) -> str:
        """
        Log a single prediction to BigQuery
        
        Parameters
        ----------
        customer_id : str, optional
            Customer identifier
        churn_probability : float
            Predicted churn probability
        churn_prediction : int
            Binary prediction (0 or 1)
        risk_level : str
            Risk level (Low/Medium/High/Critical)
        model_version : str
            Model version identifier
            
        Returns
        -------
        str
            Prediction ID
        """
        prediction_id = str(uuid.uuid4())
        
        row = {
            "prediction_id": prediction_id,
            "customer_id": customer_id,
            "churn_probability": float(churn_probability),
            "churn_prediction": int(churn_prediction),
            "risk_level": risk_level,
            "prediction_timestamp": datetime.utcnow().isoformat(),
            "model_version": model_version
        }
        
        errors = self.client.insert_rows_json(self.table_ref, [row])
        
        if errors:
            raise Exception(f"BigQuery insert failed: {errors}")
        
        return prediction_id
    
    def log_batch_predictions(self, predictions: List[Dict]) -> List[str]:
        """
        Log multiple predictions to BigQuery
        
        Parameters
        ----------
        predictions : list
            List of prediction dictionaries
            
        Returns
        -------
        list
            List of prediction IDs
        """
        rows = []
        prediction_ids = []
        
        for pred in predictions:
            prediction_id = str(uuid.uuid4())
            prediction_ids.append(prediction_id)
            
            row = {
                "prediction_id": prediction_id,
                "customer_id": pred.get("customer_id"),
                "churn_probability": float(pred["churn_probability"]),
                "churn_prediction": int(pred["churn_prediction"]),
                "risk_level": pred["risk_level"],
                "prediction_timestamp": datetime.utcnow().isoformat(),
                "model_version": pred.get("model_version", "v1.0")
            }
            rows.append(row)
        
        errors = self.client.insert_rows_json(self.table_ref, rows)
        
        if errors:
            raise Exception(f"BigQuery batch insert failed: {errors}")
        
        return prediction_ids
    
    def get_prediction_stats(self, days: int = 7) -> Dict:
        """
        Get prediction statistics for last N days
        
        Parameters
        ----------
        days : int
            Number of days to analyze
            
        Returns
        -------
        dict
            Statistics summary
        """
        query = f"""
        SELECT
            COUNT(*) as total_predictions,
            AVG(churn_probability) as avg_churn_prob,
            SUM(CASE WHEN churn_prediction = 1 THEN 1 ELSE 0 END) as predicted_churners,
            COUNT(DISTINCT customer_id) as unique_customers,
            COUNTIF(risk_level = 'Critical') as critical_risk_count,
            COUNTIF(risk_level = 'High') as high_risk_count,
            COUNTIF(risk_level = 'Medium') as medium_risk_count,
            COUNTIF(risk_level = 'Low') as low_risk_count
        FROM `{self.table_ref}`
        WHERE prediction_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
        """
        
        query_job = self.client.query(query)
        results = list(query_job.result())
        
        if results:
            return dict(results[0].items())
        return {}


class ModelStorageGCS:
    """Store and version models in Google Cloud Storage"""
    
    def __init__(self, project_id: str, bucket_name: str):
        """
        Initialize GCS model storage
        
        Parameters
        ----------
        project_id : str
            GCP project ID
        bucket_name : str
            GCS bucket name
        """
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("google-cloud-storage not installed")
        
        self.project_id = project_id
        self.bucket_name = bucket_name
        
        self.client = storage.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)
    
    def upload_model(
        self,
        model_path: str,
        model_name: str,
        version: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Upload model to GCS
        
        Parameters
        ----------
        model_path : str
            Local path to model file
        model_name : str
            Model name
        version : str
            Model version
        metadata : dict, optional
            Model metadata
            
        Returns
        -------
        str
            GCS path to uploaded model
        """
        blob_name = f"models/{model_name}/{version}/model.pkl"
        blob = self.bucket.blob(blob_name)
        
        # Add metadata
        if metadata:
            blob.metadata = metadata
        
        blob.upload_from_filename(model_path)
        
        gcs_path = f"gs://{self.bucket_name}/{blob_name}"
        print(f"✅ Model uploaded to: {gcs_path}")
        
        return gcs_path
    
    def download_model(self, model_name: str, version: str, local_path: str):
        """
        Download model from GCS
        
        Parameters
        ----------
        model_name : str
            Model name
        version : str
            Model version
        local_path : str
            Local path to save model
        """
        blob_name = f"models/{model_name}/{version}/model.pkl"
        blob = self.bucket.blob(blob_name)
        
        blob.download_to_filename(local_path)
        print(f"✅ Model downloaded to: {local_path}")
    
    def list_model_versions(self, model_name: str) -> List[str]:
        """
        List all versions of a model
        
        Parameters
        ----------
        model_name : str
            Model name
            
        Returns
        -------
        list
            List of version strings
        """
        prefix = f"models/{model_name}/"
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
        
        versions = set()
        for blob in blobs:
            parts = blob.name.split('/')
            if len(parts) >= 3:
                versions.add(parts[2])
        
        return sorted(list(versions))


class CloudMonitoringMetrics:
    """Send custom metrics to Cloud Monitoring"""
    
    def __init__(self, project_id: str):
        """
        Initialize Cloud Monitoring client
        
        Parameters
        ----------
        project_id : str
            GCP project ID
        """
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("google-cloud-monitoring not installed")
        
        self.project_id = project_id
        self.project_name = f"projects/{project_id}"
        self.client = monitoring_v3.MetricServiceClient()
    
    def send_prediction_metric(
        self,
        churn_probability: float,
        risk_level: str
    ):
        """
        Send prediction metric to Cloud Monitoring
        
        Parameters
        ----------
        churn_probability : float
            Predicted churn probability
        risk_level : str
            Risk level
        """
        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/churn/prediction_probability"
        series.resource.type = "global"
        
        # Add labels
        series.metric.labels["risk_level"] = risk_level
        
        # Create data point
        now = datetime.utcnow()
        seconds = int(now.timestamp())
        nanos = int((now.timestamp() - seconds) * 10**9)
        
        interval = monitoring_v3.TimeInterval(
            {"end_time": {"seconds": seconds, "nanos": nanos}}
        )
        
        point = monitoring_v3.Point(
            {"interval": interval, "value": {"double_value": churn_probability}}
        )
        
        series.points = [point]
        
        # Write time series
        self.client.create_time_series(
            name=self.project_name,
            time_series=[series]
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_gcp_services(project_id: str) -> Dict:
    """
    Initialize all GCP services
    
    Parameters
    ----------
    project_id : str
        GCP project ID
        
    Returns
    -------
    dict
        Dictionary of initialized services
    """
    services = {}
    
    if GOOGLE_CLOUD_AVAILABLE:
        try:
            services['bigquery'] = BigQueryLogger(project_id)
            print("✅ BigQuery logger initialized")
        except Exception as e:
            print(f"⚠️ BigQuery initialization failed: {e}")
        
        try:
            bucket_name = f"{project_id}-churn-models"
            services['storage'] = ModelStorageGCS(project_id, bucket_name)
            print("✅ GCS model storage initialized")
        except Exception as e:
            print(f"⚠️ GCS initialization failed: {e}")
        
        try:
            services['monitoring'] = CloudMonitoringMetrics(project_id)
            print("✅ Cloud Monitoring initialized")
        except Exception as e:
            print(f"⚠️ Cloud Monitoring initialization failed: {e}")
    else:
        print("⚠️ Google Cloud SDK not available. Install with:")
        print("   pip install google-cloud-bigquery google-cloud-storage google-cloud-monitoring")
    
    return services


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Example usage
    PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-project-id")
    
    print("="*70)
    print("GCP INTEGRATION TEST")
    print("="*70)
    
    services = initialize_gcp_services(PROJECT_ID)
    
    if 'bigquery' in services:
        print("\n📊 Testing BigQuery logging...")
        try:
            pred_id = services['bigquery'].log_prediction(
                customer_id="TEST_001",
                churn_probability=0.75,
                churn_prediction=1,
                risk_level="High",
                model_version="v1.0"
            )
            print(f"✅ Logged prediction: {pred_id}")
        except Exception as e:
            print(f"❌ BigQuery test failed: {e}")
    
    print("\n✅ GCP integration test complete!")