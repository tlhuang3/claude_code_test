"""
Restaurant Review Analysis Application

This application processes restaurant reviews from Excel files,
uses Azure OpenAI to analyze sentiments, and outputs results with
summaries and scores.
"""

import os
import logging
from typing import Tuple, Optional, List, Dict, Any
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import uuid
from datetime import datetime
import io
import shutil
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Restaurant Review Analyzer",
    description="Analyzes restaurant reviews using Azure OpenAI to generate summaries and satisfaction scores",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a directory for temporary files
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# Models
class AnalysisTask(BaseModel):
    """Model for tracking analysis tasks"""
    task_id: str = Field(..., description="Unique identifier for the task")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Status of the task (pending, processing, completed, failed)")
    created_at: datetime = Field(..., description="When the task was created")
    completed_at: Optional[datetime] = Field(None, description="When the task was completed")
    result_path: Optional[str] = Field(None, description="Path to the result file")
    error: Optional[str] = Field(None, description="Error message if task failed")

class ColumnSelectionRequest(BaseModel):
    """Model for selecting the review column"""
    task_id: str = Field(..., description="Task ID from file upload")
    review_column: str = Field(..., description="Column name containing reviews")

# In-memory storage for tasks (in production, use a database)
tasks_db: Dict[str, AnalysisTask] = {}

# Get Azure OpenAI client
def get_openai_client() -> AzureOpenAI:
    """
    Initialize Azure OpenAI client with credentials from Key Vault
    """
    try:
        # For local development with environment variables
        if "AZURE_OPENAI_KEY" in os.environ and "AZURE_OPENAI_ENDPOINT" in os.environ:
            return AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_KEY"],
                api_version="2023-05-15",
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
            )
        
        # For production with Azure managed identity
        credential = DefaultAzureCredential()
        key_vault_url = os.environ.get("KEY_VAULT_URL")
        
        if not key_vault_url:
            raise ValueError("KEY_VAULT_URL environment variable not set")
            
        secret_client = SecretClient(vault_url=key_vault_url, credential=credential)
        api_key = secret_client.get_secret("AZURE-OPENAI-KEY").value
        endpoint = secret_client.get_secret("AZURE-OPENAI-ENDPOINT").value
        
        return AzureOpenAI(
            api_key=api_key,
            api_version="2023-05-15",
            azure_endpoint=endpoint
        )
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not initialize Azure OpenAI client")

def analyze_review(client: AzureOpenAI, review: str) -> Tuple[str, Optional[int]]:
    """
    Analyze a restaurant review using Azure OpenAI to generate a summary and score.
    
    Args:
        client: Azure OpenAI client
        review: The review text to analyze
        
    Returns:
        Tuple of (summary, score)
    """
    if not review or pd.isna(review):
        return "", None
    
    prompt = f"""
    Review: "{review}"
    
    Task 1: Summarize this restaurant review in one or two sentences.
    
    Task 2: Rate this review on a scale from 1 to 5, where:
    1 = Extremely dissatisfied
    2 = Dissatisfied
    3 = Neutral
    4 = Satisfied
    5 = Extremely satisfied
    
    Your response should be in this format:
    Summary: [your summary]
    Score: [numeric score 1-5]
    """
    
    try:
        # Use the model deployed in your Azure OpenAI resource
        response = client.chat.completions.create(
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo"),
            messages=[
                {"role": "system", "content": "You analyze restaurant reviews to extract a concise summary and satisfaction score from 1-5."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0
        )
        
        response_text = response.choices[0].message.content
        
        # Extract summary and score from response
        summary_line = next((line for line in response_text.split('\n') if line.startswith("Summary:")), "")
        score_line = next((line for line in response_text.split('\n') if line.startswith("Score:")), "")
        
        summary = summary_line.replace("Summary:", "").strip() if summary_line else ""
        score = score_line.replace("Score:", "").strip() if score_line else None
        
        try:
            score = int(score) if score else None
            if score and (score < 1 or score > 5):
                logger.warning(f"Score out of range (1-5): {score}")
                score = max(1, min(5, score))  # Clamp between 1-5
        except ValueError:
            logger.warning(f"Could not parse score from '{score_line}'")
            score = None
            
        return summary, score
    except Exception as e:
        logger.error(f"Error analyzing review: {str(e)}")
        return "", None

async def process_excel_file(task_id: str, file_path: str, review_column: str) -> None:
    """
    Background task to process the Excel file with reviews
    
    Args:
        task_id: The task ID
        file_path: Path to the uploaded Excel file
        review_column: Name of the column containing reviews
    """
    task = tasks_db[task_id]
    task.status = "processing"
    
    try:
        # Read Excel file
        df = pd.read_excel(file_path)
        
        if review_column not in df.columns:
            raise ValueError(f"Column '{review_column}' not found in the Excel file")
        
        # Initialize OpenAI client
        client = get_openai_client()
        
        # Process reviews
        results = []
        total_reviews = len(df)
        
        for i, review in enumerate(df[review_column]):
            logger.info(f"Processing review {i+1}/{total_reviews}")
            summary, score = analyze_review(client, review)
            results.append({"Summary": summary, "Score": score})
            
        # Add results to dataframe
        results_df = pd.DataFrame(results)
        df = pd.concat([df, results_df], axis=1)
        
        # Save to Excel
        result_filename = f"analyzed_{task_id}.xlsx"
        result_path = os.path.join(TEMP_DIR, result_filename)
        df.to_excel(result_path, index=False)
        
        # Update task status
        task.status = "completed"
        task.completed_at = datetime.now()
        task.result_path = result_path
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        task.status = "failed"
        task.error = str(e)
    finally:
        # Clean up the original file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Failed to clean up file {file_path}: {str(e)}")

@app.post("/upload/", response_model=dict)
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload an Excel file with restaurant reviews
    
    Returns:
        task_id and columns of the Excel file
    """
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are supported")
    
    try:
        # Generate a task ID
        task_id = str(uuid.uuid4())
        
        # Save the uploaded file
        temp_file_path = os.path.join(TEMP_DIR, f"upload_{task_id}{os.path.splitext(file.filename)[1]}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read the Excel file to get columns
        df = pd.read_excel(temp_file_path)
        columns = df.columns.tolist()
        
        # Create a task entry
        tasks_db[task_id] = AnalysisTask(
            task_id=task_id,
            filename=file.filename,
            status="pending",
            created_at=datetime.now()
        )
        
        return {
            "task_id": task_id,
            "filename": file.filename,
            "columns": columns
        }
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/")
async def analyze_reviews(request: ColumnSelectionRequest, background_tasks: BackgroundTasks):
    """
    Start the analysis of reviews in the selected column
    """
    if request.task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks_db[request.task_id]
    
    if task.status != "pending":
        raise HTTPException(status_code=400, detail=f"Task is already in {task.status} state")
    
    # Get the file path
    file_path = os.path.join(TEMP_DIR, f"upload_{request.task_id}{os.path.splitext(task.filename)[1]}")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found")
    
    # Start the background task
    background_tasks.add_task(
        process_excel_file,
        task_id=request.task_id,
        file_path=file_path,
        review_column=request.review_column
    )
    
    return {"message": "Analysis started", "task_id": request.task_id}

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status of a task
    """
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks_db[task_id]
    return {
        "task_id": task.task_id,
        "filename": task.filename,
        "status": task.status,
        "created_at": task.created_at,
        "completed_at": task.completed_at,
        "error": task.error
    }

@app.get("/download/{task_id}")
async def download_results(task_id: str):
    """
    Download the results of a completed task
    """
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks_db[task_id]
    
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Task is not completed yet")
    
    if not task.result_path or not os.path.exists(task.result_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        task.result_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"analyzed_{task.filename}"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    """Runs when the application starts"""
    logger.info("Starting Restaurant Review Analyzer API")

@app.on_event("shutdown")
async def shutdown_event():
    """Runs when the application shuts down"""
    logger.info("Shutting down Restaurant Review Analyzer API")
    # Clean up temporary files
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    # For local development
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)