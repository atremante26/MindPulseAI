from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pipeline.ingestion.ingest_reddit import RedditIngestor
from pipeline.ingestion.ingest_news import NewsIngestor
from pipeline.snowflake.load_snowflake import (
    load_reddit_to_snowflake,
    load_news_to_snowflake
)

default_args = {
    'owner': 'andrew',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def run_reddit():
    RedditIngestor().run("reddit", "reddit_suite", save_s3=True, save_local=False)

def ingest_news():
    NewsIngestor().run("news", "news_suite", save_s3=True, save_local=False)

def retrain_forecasting():
    from analysis.forecasting import (
        prepare_reddit_data, 
        prepare_news_data, 
        train_models as train_forecasting, 
        save_models,
        save_to_s3
    )
    
    reddit_data = prepare_reddit_data()
    news_data = prepare_news_data()
    models, forecasts, metrics = train_forecasting(reddit_data, news_data)
    _ = save_models(models, forecasts, metrics, reddit_data, news_data)
    save_to_s3(forecasts, metrics, reddit_data, news_data)

def retrain_insights():
    from analysis.insights import train as train_insights
    _, _, _ = train_insights()

with DAG(
    dag_id="ingestion_dag",
    default_args=default_args,
    start_date=datetime(2025, 7, 21),
    schedule_interval="@weekly",
    catchup=False,
    tags=["mental_health"],
) as dag:

    ingest_reddit_task = PythonOperator(
        task_id='ingest_reddit',
        python_callable=run_reddit
    )

    ingest_news_task = PythonOperator(
        task_id='ingest_news',
        python_callable=ingest_news
    )

    load_reddit_task = PythonOperator(
        task_id='load_reddit_to_snowflake',
        python_callable=load_reddit_to_snowflake 
    )

    load_news_task = PythonOperator(
        task_id='load_news_to_snowflake',
        python_callable=load_news_to_snowflake
    )

    retrain_forecasting_task = PythonOperator(
        task_id="retrain_forecasting",
        python_callable=retrain_forecasting
    )

    retrain_insights_task = PythonOperator(
        task_id="retrain_insights",
        python_callable=retrain_insights
    )

    ingest_reddit_task >> load_reddit_task
    ingest_news_task >> load_news_task
    [load_reddit_task, load_news_task] >> retrain_forecasting_task >> retrain_insights_task