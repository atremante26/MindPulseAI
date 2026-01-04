# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /opt/airflow

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libre2-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install latest Airflow with constraints
RUN pip install --no-cache-dir apache-airflow==2.10.2 \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.2/constraints-3.11.txt"

# Copy pipeline code
COPY pipeline/ ./pipeline/
COPY airflow/dags/ ./dags/
COPY gx/ ./gx/

# Create necessary directories for Airflow
RUN mkdir -p logs plugins temp

# Set environment variables for Airflow
ENV AIRFLOW_HOME=/opt/airflow
ENV PYTHONPATH=/opt/airflow
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
ENV AIRFLOW__CORE__LOAD_EXAMPLES=false
ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////tmp/airflow.db
ENV AIRFLOW__CORE__EXECUTOR=SequentialExecutor

# Set the default command to run your ingestion DAG
CMD ["sh", "-c", "airflow db init && airflow dags test ingestion_dag $(date +%Y-%m-%d)"]