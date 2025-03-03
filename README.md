# Restaurant Review Analyzer

This application analyzes restaurant reviews from Excel files using Azure OpenAI, generating summaries and satisfaction scores (1-5).

## Features

- **Upload Excel files** with restaurant review data
- **Select the column** containing reviews to analyze
- **AI-powered analysis** using Azure OpenAI
- **Automatic scoring** on a scale from 1-5
- **Download results** as Excel files with added summary and score columns
- **Production-ready** FastAPI backend that can be deployed to Azure
- **React frontend** for easy interaction

## Architecture

The application consists of two components:

1. **Backend API (FastAPI)**: 
   - Handles file uploads and processing
   - Integrates with Azure OpenAI for review analysis
   - Manages background processing tasks
   - Secure credential management with Azure Key Vault

2. **Frontend (React)**:
   - User-friendly interface
   - Responsive design
   - Progress tracking
   - File upload/download

## Prerequisites

- Azure subscription
- Azure OpenAI resource
- Azure Container Registry
- Azure Key Vault (for secure API key storage)

## Setup Instructions

### Backend Setup

1. **Azure OpenAI Setup**:
   - Create an Azure OpenAI resource
   - Deploy a model (e.g., `gpt-35-turbo`)
   - Note the endpoint and API key

2. **Azure Key Vault Setup**:
   - Create a Key Vault
   - Add secrets:
     - `AZURE-OPENAI-KEY`: Your Azure OpenAI API key
     - `AZURE-OPENAI-ENDPOINT`: Your Azure OpenAI endpoint

3. **Configure Environment Variables**:
   For local development, create a `.env` file with:
   ```
   KEY_VAULT_URL=https://your-keyvault.vault.azure.net/
   AZURE_OPENAI_DEPLOYMENT_NAME=your-model-deployment-name
   ```

4. **Deploy to Azure**:
   - Update the variables in `deploy.sh` with your values
   - Make the script executable: `chmod +x deploy.sh`
   - Run the deployment script: `./deploy.sh`

### Frontend Setup

1. **Configure API Endpoint**:
   - Update the `API_BASE_URL` in `frontend/index.html` to point to your deployed API

2. **Deploy to Azure Static Web Apps or App Service**:
   - For Static Web Apps, follow the Azure Static Web Apps deployment guide
   - For App Service, deploy using the Azure CLI or Visual Studio Code

## Local Development

### Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app:app --reload
```

### Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the server
npm start
```

## Usage

1. Open the web application
2. Upload an Excel file containing restaurant reviews
3. Select the column containing the reviews
4. Click "Analyze Reviews"
5. Wait for the analysis to complete
6. Download the results file

## Integration with Databricks

This application can integrate with Databricks for more advanced data processing:

1. **Data Preparation**: Use Databricks to clean and prepare review data
2. **Batch Processing**: Process large volumes of reviews using Databricks clusters
3. **Advanced Analytics**: Perform additional analytics on review data and scores

To integrate:

1. Use Databricks to read and preprocess data
2. Call the API from Databricks notebooks
3. Store results in your data lake or data warehouse

## Security Considerations

- All credentials are stored in Azure Key Vault
- The application uses Azure Managed Identity for authentication
- API access can be restricted using Azure API Management

## Maintenance and Monitoring

- The application logs all operations and errors
- Azure Application Insights can be added for monitoring
- Health check endpoint available at `/health`

## License

[MIT License](LICENSE)