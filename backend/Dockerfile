# Sets the base image as Python 3.10
FROM python:3.10

# Sets the working directory inside the container to /app
WORKDIR /app

# Copies requirements.txt and installs necessary packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies all remaining files from the local backend directory to /app in the container
COPY . .

# Runs the FastAPI app with Uvicorn when the container starts using 'app' in main.py
# --host 0.0.0.0 makes it accessible from outside the container (for Render)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]