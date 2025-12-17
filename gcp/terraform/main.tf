# Terraform configuration for GCP deployment
# Creates Cloud Run service, BigQuery dataset, and monitoring

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Configure GCP Provider
provider "google" {
  project = var.project_id
  region  = var.region
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "churn-prediction-api"
}

# Enable required APIs
resource "google_project_service" "cloud_run" {
  service = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloud_build" {
  service = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "bigquery" {
  service = "bigquery.googleapis.com"
  disable_on_destroy = false
}

# Cloud Run Service
resource "google_cloud_run_service" "api" {
  name     = var.service_name
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/${var.service_name}:latest"
        
        resources {
          limits = {
            memory = "2Gi"
            cpu    = "2"
          }
        }
        
        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }
        
        env {
          name  = "ENVIRONMENT"
          value = "production"
        }
      }
      
      container_concurrency = 80
      timeout_seconds       = 300
    }
    
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = "10"
        "autoscaling.knative.dev/minScale" = "1"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [google_project_service.cloud_run]
}

# Allow unauthenticated access
resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_service.api.name
  location = google_cloud_run_service.api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# BigQuery Dataset for storing predictions
resource "google_bigquery_dataset" "churn_predictions" {
  dataset_id    = "churn_predictions"
  friendly_name = "Customer Churn Predictions"
  description   = "Stores churn predictions and model performance metrics"
  location      = "US"

  labels = {
    environment = "production"
    project     = "churn-prediction"
  }

  depends_on = [google_project_service.bigquery]
}

# BigQuery Table for predictions
resource "google_bigquery_table" "predictions" {
  dataset_id = google_bigquery_dataset.churn_predictions.dataset_id
  table_id   = "predictions"

  schema = jsonencode([
    {
      name = "prediction_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "customer_id"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "churn_probability"
      type = "FLOAT64"
      mode = "REQUIRED"
    },
    {
      name = "churn_prediction"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "risk_level"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "prediction_timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "model_version"
      type = "STRING"
      mode = "REQUIRED"
    }
  ])

  time_partitioning {
    type  = "DAY"
    field = "prediction_timestamp"
  }
}

# Cloud Storage bucket for model artifacts
resource "google_storage_bucket" "models" {
  name          = "${var.project_id}-churn-models"
  location      = "US"
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = "production"
    project     = "churn-prediction"
  }
}

# Outputs
output "api_url" {
  description = "Cloud Run service URL"
  value       = google_cloud_run_service.api.status[0].url
}

output "bigquery_dataset" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.churn_predictions.dataset_id
}

output "model_bucket" {
  description = "GCS bucket for models"
  value       = google_storage_bucket.models.name
}