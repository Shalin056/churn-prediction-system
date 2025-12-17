#!/bin/bash

# Google Cloud Platform Deployment Script
# Deploys Customer Churn Prediction API to Cloud Run

set -e

echo "=========================================="
echo "GCP Deployment - Churn Prediction System"
echo "=========================================="

# Configuration
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
SERVICE_NAME="churn-prediction-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "✅ gcloud CLI found"

# Set project
echo ""
echo "Setting GCP project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo ""
echo "Enabling required GCP APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build Docker image
echo ""
echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME:latest .

# Tag for Container Registry
docker tag $IMAGE_NAME:latest $IMAGE_NAME:v1

# Configure Docker for GCR
echo ""
echo "Configuring Docker authentication..."
gcloud auth configure-docker

# Push to Google Container Registry
echo ""
echo "📤 Pushing image to Google Container Registry..."
docker push $IMAGE_NAME:latest
docker push $IMAGE_NAME:v1

# Deploy to Cloud Run
echo ""
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --set-env-vars "PROJECT_ID=$PROJECT_ID" \
  --set-env-vars "ENVIRONMENT=production"

# Get service URL
echo ""
echo "✅ Deployment complete!"
echo ""
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --format 'value(status.url)')

echo "=========================================="
echo "🎉 API deployed successfully!"
echo "=========================================="
echo ""
echo "API URL:        $SERVICE_URL"
echo "Health Check:   $SERVICE_URL/health"
echo "API Docs:       $SERVICE_URL/docs"
echo ""
echo "Test with:"
echo "curl $SERVICE_URL/health"
echo ""
echo "To view logs:"
echo "gcloud run logs read $SERVICE_NAME --region $REGION --limit 50"
echo ""
echo "To update deployment:"
echo "./deploy_gcp.sh"
echo "=========================================="