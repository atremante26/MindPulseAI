# MindPulseAI

> A full-stack mental health analytics platform that tracks how mental health is discussed across Reddit and news media — surfacing trends, forecasting shifts in public discourse, and helping you find the right resources for your needs.

🌐 **Live Site**: [mindpulseai.io](https://mindpulseai.io)  
📡 **API Docs**: [Render](https://mental-health-project-bct5.onrender.com/docs)  
👤 **Built by**: [Andrew Tremante](https://andrewtremante.com)

---

## Overview

MindPulseAI ingests real-time data from Reddit and news APIs, runs it through a machine learning pipeline, and serves insights through a React frontend. The platform updates weekly via an automated Airflow pipeline on AWS ECS Fargate and stores processed data in Snowflake.

### What It Does

- **Tracks** mental health discourse volume and sentiment across Reddit and news media.
- **Forecasts** 30-day trends using Facebook Prophet time-series models.
- **Explains** forecast datapoints using LLM-powered insights (Claude 3.5 Haiku).
- **Recommends** personalized mental health resources via a content-based recommender system.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React, Vite, GitHub Pages |
| **Backend API** | FastAPI, Python, Render |
| **Pipeline** | Apache Airflow, AWS ECS Fargate, EventBridge |
| **Storage** | AWS S3 (data lake), Snowflake (warehouse) |
| **ML / AI** | Prophet, DistilBERT, Claude 3.5 Haiku, Cosine Similarity |
| **Data Validation** | Great Expectations |
| **Infrastructure** | AWS CDK, Docker, ECR |

---

## Architecture

```
Reddit API ──┐
             ├──► ECS Fargate (Airflow) ──► S3 ──► Snowflake ──► FastAPI ──► React
News API ────┘         │
                       └──► Retrain Models (Prophet, Recommender, LLM Insights)
```

**Pipeline runs weekly** via EventBridge cron (`0 8 ? * 1 *`). Each run:
1. Ingests Reddit posts and news articles
2. Scores sentiment using DistilBERT transformer
3. Validates data quality with Great Expectations
4. Loads to Snowflake
5. Retrains all 4 ML models
6. Saves updated forecasts and insights to S3

---

## The Four Models

### 1. Prophet Forecasting
Facebook Prophet trains on historical volume and sentiment data from Reddit and News API, generating 30-day forecasts with confidence intervals for both sources.

### 2. LLM Insights
Claude 3.5 Haiku interprets clicked forecast datapoints, surfacing statistical significance, mental health domain interpretation, and actionable recommendations grounded in WHO data.

### 3. Workplace Mental Health Clustering
HDBSCAN clustering of 914 Open Sourcing Mental Illness (OSMI) survey respondents, segmenting workers into profiles based on demographics, treatment history, and work environment. Gower distance handles mixed categorical and numeric features.

### 4. Recommender System
Weighted cosine similarity across concern, cost, age, and resource type feature groups. Crisis resources receive an explicit boost for ethical prioritization, with match scores clamped to 0–100%.

---

## Project Structure

```
MindPulseAI/
├── pipeline/                  # Data ingestion and ETL
│   ├── ingestion/             # Reddit, News ingestors + sentiment analyzer
│   └── snowflake/             # Snowflake loading logic
├── airflow/
│   └── dags/                  # Airflow DAG definitions
├── analysis/
│   ├── forecasting/           # Prophet training and utilities
│   ├── insights/              # LLM insight generation
│   ├── recommendations/       # Recommender system
|   ├── clustering/            # Clustering OSMI survey
│   └── config/                # Model hyperparameters
├── backend/                   # FastAPI REST API
├── frontend/                  # React + Vite
├── gx/                        # Great Expectations suites
├── infrastructure/            # AWS CDK stack
├── data/                      # Static datasets and mental health resources
└── Dockerfile                 # Production ECS container
```

---

## Infrastructure

- **Compute**: AWS ECS Fargate (0.5 vCPU, 2GB RAM) — runs ~16 seconds/week
- **Scheduling**: EventBridge weekly cron
- **Storage**: S3 for raw/processed data, Snowflake for analytics
- **Monitoring**: CloudWatch logs
- **CI/CD**: GitHub Actions → ECR → ECS

---

## Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

---

## Author

**Andrew Tremante** - Data Scientist & ML Engineer  
[Website](https://andrewtremante.com) · [LinkedIn](https://www.linkedin.com/in/andrew-tremante-71253a238/) · [GitHub](https://github.com/atremante26) · [Google Scholar](https://scholar.google.com/citations?user=EXRKa94AAAAJ&hl=en)