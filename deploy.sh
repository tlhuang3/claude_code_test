#!/bin/bash
set -e

# Configuration - replace these with your actual values
ACR_NAME="yourcompanyacr"
RESOURCE_GROUP="your-resource-group"
KEY_VAULT_URL="https://your-keyvault.vault.azure.net/"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-35-turbo"  # Your deployed model name

# Login to Azure
echo "Logging in to Azure..."
az login

# Build and push Docker image to Azure Container Registry
echo "Building and pushing Docker image..."
az acr build --registry $ACR_NAME --image restaurant-review-app:latest .

# Replace variables in the deployment template
echo "Creating deployment configuration..."
sed -e "s/\${ACR_NAME}/$ACR_NAME/g" \
    -e "s|\${KEY_VAULT_URL}|$KEY_VAULT_URL|g" \
    -e "s|\${AZURE_OPENAI_DEPLOYMENT_NAME}|$AZURE_OPENAI_DEPLOYMENT_NAME|g" \
    azure-deploy.yaml > azure-deploy-filled.yaml

# Deploy the container
echo "Deploying container..."
az container create --resource-group $RESOURCE_GROUP --file azure-deploy-filled.yaml

# Clean up the temporary file
rm azure-deploy-filled.yaml

echo "Deployment complete! Your app should be available at:"
echo "http://$(az container show --resource-group $RESOURCE_GROUP --name restaurant-review-app --query ipAddress.ip --output tsv):8000"
echo "You can test the health check endpoint at: http://$(az container show --resource-group $RESOURCE_GROUP --name restaurant-review-app --query ipAddress.ip --output tsv):8000/health"