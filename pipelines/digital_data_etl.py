import mlflow

from steps.etl import crawl_links, get_or_create_user


def digital_data_etl(user_full_name: str, links: list[str]) -> str:
    """Digital Data ETL pipeline using MLflow for tracking."""
    
    with mlflow.start_run(run_name="digital_data_etl") as run:
        # Step 1: Get or create user
        user = get_or_create_user(user_full_name)
        
        # Step 2: Crawl links
        crawled_links = crawl_links(user=user, links=links)
        
        # Log final metrics
        mlflow.log_metric("total_links_processed", len(links))
        mlflow.log_metric("total_crawled_links", len(crawled_links))
        
        return run.info.run_id
