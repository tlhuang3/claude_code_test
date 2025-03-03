FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create temp directory for file storage
RUN mkdir -p temp_files && chmod 777 temp_files

# Expose the port used by the FastAPI application
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]