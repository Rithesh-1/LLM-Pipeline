import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import yaml
from loguru import logger
from qdrant_client.http import exceptions

from llm_pipeline_system.application.dataset import generation
from llm_pipeline_system.domain.cleaned_documents import (
    CleanedArticleDocument,
    CleanedDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
)
from llm_pipeline_system.domain.dataset import DatasetType, InstructTrainTestSplit, PreferenceTrainTestSplit
from llm_pipeline_system.domain.prompt import GenerateDatasetSamplesPrompt
from llm_pipeline_system.domain.types import DataCategory
from llm_pipeline_system.settings import settings

# ETL Pipeline Functions
# ... (previous ETL functions remain here) ...

# Feature Engineering Pipeline Functions
# ... (previous Feature Engineering functions remain here) ...

# Dataset Generation Pipeline Functions

def _query_feature_store() -> list:
    logger.info("Querying feature store.")
    def __fetch(cleaned_document_type: type[CleanedDocument], limit: int = 1) -> list[CleanedDocument]:
        try:
            cleaned_documents, next_offset = cleaned_document_type.bulk_find(limit=limit)
        except exceptions.UnexpectedResponse:
            return []
        while next_offset:
            documents, next_offset = cleaned_document_type.bulk_find(limit=limit, offset=next_offset)
            cleaned_documents.extend(documents)
        return cleaned_documents

    def __fetch_articles() -> list[CleanedDocument]:
        return __fetch(CleanedArticleDocument)

    def __fetch_posts() -> list[CleanedDocument]:
        return __fetch(CleanedPostDocument)

    def __fetch_repositories() -> list[CleanedDocument]:
        return __fetch(CleanedRepositoryDocument)

    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(__fetch_articles): "articles",
            executor.submit(__fetch_posts): "posts",
            executor.submit(__fetch_repositories): "repositories",
        }
        results = {}
        for future in as_completed(future_to_query):
            query_name = future_to_query[future]
            try:
                results[query_name] = future.result()
            except Exception:
                logger.exception(f"'{query_name}' request failed.")
                results[query_name] = []
    
    cleaned_documents = [doc for query_result in results.values() for doc in query_result]
    return cleaned_documents

def _create_prompts(documents: list, dataset_type: DatasetType, llm_provider: generation.LLMProvider) -> dict[DataCategory, list[GenerateDatasetSamplesPrompt]]:
    dataset_generator = generation.get_dataset_generator(dataset_type)
    grouped_prompts = dataset_generator.get_prompts(documents, llm_provider)
    
    prompt_categories = list(grouped_prompts.keys())
    prompt_num_samples = {category: len(prompts) for category, prompts in grouped_prompts.items()}
    mlflow.log_dict({"data_categories": prompt_categories, "data_categories_num_prompts": prompt_num_samples}, "prompts_metadata.json")

    return grouped_prompts

def _generate_instruct_dataset(prompts: dict[DataCategory, list[GenerateDatasetSamplesPrompt]], test_split_size: float, mock: bool, llm_provider: generation.LLMProvider) -> InstructTrainTestSplit:
    dataset_generator = generation.get_dataset_generator(DatasetType.INSTRUCTION)
    datasets = dataset_generator.generate(prompts, test_size=test_split_size, mock=mock, llm_provider=llm_provider)
    
    instruct_dataset_categories = list(datasets.train.keys())
    train_num_samples = {category: instruct_dataset.num_samples for category, instruct_dataset in datasets.train.items()}
    test_num_samples = {category: instruct_dataset.num_samples for category, instruct_dataset in datasets.test.items()}
    metadata = {
        "data_categories": instruct_dataset_categories,
        "test_split_size": datasets.test_split_size,
        "train_num_samples_per_category": train_num_samples,
        "test_num_samples_per_category": test_num_samples,
    }
    mlflow.log_dict(metadata, "instruct_dataset_metadata.json")

    return datasets

def _generate_preference_dataset(
    prompts: dict[DataCategory, list[GenerateDatasetSamplesPrompt]], 
    test_split_size: float, 
    mock: bool, 
    llm_provider: generation.LLMProvider
) -> PreferenceTrainTestSplit:
    dataset_generator = generation.get_dataset_generator(DatasetType.PREFERENCE)
    datasets = dataset_generator.generate(prompts, test_size=test_split_size, mock=mock, llm_provider=llm_provider)

    preference_dataset_categories = list(datasets.train.keys())
    train_num_samples = {category: preference_dataset.num_samples for category, preference_dataset in datasets.train.items()}
    test_num_samples = {category: preference_dataset.num_samples for category, preference_dataset in datasets.test.items()}
    metadata = {
        "data_categories": preference_dataset_categories,
        "test_split_size": datasets.test_split_size,
        "train_num_samples_per_category": train_num_samples,
        "test_num_samples_per_category": test_num_samples,
    }
    mlflow.log_dict(metadata, "preference_dataset_metadata.json")

    return datasets

def _push_to_huggingface(dataset: InstructTrainTestSplit | PreferenceTrainTestSplit, dataset_id: str):
    assert dataset_id is not None, "Dataset id must be provided for pushing to Huggingface"
    assert settings.HUGGINGFACE_ACCESS_TOKEN is not None, "Huggingface access token must be provided for pushing to Huggingface"
    logger.info(f"Pushing dataset {dataset_id} to Hugging Face.")
    huggingface_dataset = dataset.to_huggingface(flatten=True)
    huggingface_dataset.push_to_hub(dataset_id, token=settings.HUGGINGFACE_ACCESS_TOKEN)

def run_generate_instruct_datasets(args):
    print("Running Generate Instruct Datasets pipeline...")
    with mlflow.start_run(run_name="generate_instruct_datasets") as run:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
        mlflow.log_params(config)

        cleaned_documents = _query_feature_store()
        prompts = _create_prompts(cleaned_documents, DatasetType.INSTRUCTION, llm_provider)
        
        llm_provider_str = config.get("parameters", {}).get("llm_provider", "openai")
        llm_provider = generation.LLMProvider(llm_provider_str)

        dataset = _generate_instruct_dataset(
            prompts, 
            config.get("parameters", {}).get("test_split_size", 0.1), 
            config.get("parameters", {}).get("mock", False),
            llm_provider
        )

        if config.get("parameters", {}).get("push_to_huggingface", False):
            _push_to_huggingface(dataset, config.get("parameters", {}).get("dataset_id"))

def run_generate_preference_datasets(args):
    print("Running Generate Preference Datasets pipeline...")
    with mlflow.start_run(run_name="generate_preference_datasets") as run:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
        mlflow.log_params(config)

        cleaned_documents = _query_feature_store()
        prompts = _create_prompts(cleaned_documents, DatasetType.PREFERENCE, llm_provider)
        
        llm_provider_str = config.get("parameters", {}).get("llm_provider", "openai")
        llm_provider = generation.LLMProvider(llm_provider_str)
        
        dataset = _generate_preference_dataset(
            prompts, 
            config.get("parameters", {}).get("test_split_size", 0.1), 
            config.get("parameters", {}).get("mock", False),
            llm_provider
        )

        if config.get("parameters", {}).get("push_to_huggingface", False):
            _push_to_huggingface(dataset, config.get("parameters", {}).get("dataset_id"))

def run_digital_data_etl(args):
    """Run the digital data ETL pipeline with MLflow tracking."""
    print("Running Digital Data ETL pipeline...")
    
    with mlflow.start_run(run_name="digital_data_etl") as run:
        # Load configuration
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Log configuration parameters
        mlflow.log_params(config)
        
        # Import the pipeline functions
        from steps.etl import crawl_links, get_or_create_user
        
        # Execute pipeline steps
        user_full_name = config.get("parameters", {}).get("user_full_name", "User")
        links = config.get("parameters", {}).get("links", [])
        
        logger.info(f"Starting ETL pipeline for user: {user_full_name}")
        
        # Step 1: Get or create user
        user = get_or_create_user(user_full_name)
        
        # Step 2: Crawl links
        crawled_links = crawl_links(user=user, links=links)
        
        # Log final results
        mlflow.log_metric("total_crawled_links", len(crawled_links))
        
        logger.info("Digital Data ETL pipeline completed successfully")
        
        return run.info.run_id


def run_feature_engineering(args):
    """Run the feature engineering pipeline with MLflow tracking."""
    print("Running Feature Engineering pipeline...")
    
    with mlflow.start_run(run_name="feature_engineering") as run:
        # Load configuration
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Log configuration parameters
        mlflow.log_params(config)
        
        # Import the pipeline functions
        from steps import feature_engineering as fe_steps
        
        # Execute pipeline steps
        author_full_names = config.get("parameters", {}).get("author_full_names", ["User"])
        
        logger.info(f"Starting Feature Engineering pipeline for authors: {author_full_names}")
        
        # Step 1: Query data warehouse
        raw_documents = fe_steps.query_data_warehouse(author_full_names)
        
        # Step 2: Clean documents
        cleaned_documents = fe_steps.clean_documents(raw_documents)
        
        # Step 3: Load cleaned documents to vector DB
        load_result_1 = fe_steps.load_to_vector_db(cleaned_documents)
        
        # Step 4: Chunk and embed documents
        embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
        
        # Step 5: Load embedded documents to vector DB
        load_result_2 = fe_steps.load_to_vector_db(embedded_documents)
        
        # Log final results
        mlflow.log_metric("total_authors_processed", len(author_full_names))
        mlflow.log_metric("raw_documents_count", len(raw_documents) if raw_documents else 0)
        mlflow.log_metric("cleaned_documents_count", len(cleaned_documents) if cleaned_documents else 0)
        
        logger.info("Feature Engineering pipeline completed successfully")
        
        return run.info.run_id


def main():
    """Main function to run different pipelines based on arguments."""
    parser = argparse.ArgumentParser(description="Run MLflow pipelines")
    parser.add_argument("--pipeline", type=str, required=True, 
                       choices=["digital_data_etl", "feature_engineering", "generate_instruct_datasets", "generate_preference_datasets"],
                       help="Pipeline to run")
    parser.add_argument("--config-path", type=str, required=True,
                       help="Path to the configuration file")
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI to the running server
    mlflow.set_tracking_uri("http://localhost:5000")
    logger.info("MLflow tracking URI set to: http://localhost:5000")
    
    if args.pipeline == "digital_data_etl":
        run_digital_data_etl(args)
    elif args.pipeline == "feature_engineering":
        run_feature_engineering(args)
    elif args.pipeline == "generate_instruct_datasets":
        run_generate_instruct_datasets(args)
    elif args.pipeline == "generate_preference_datasets":
        run_generate_preference_datasets(args)
    else:
        raise ValueError(f"Unknown pipeline: {args.pipeline}")


if __name__ == "__main__":
    main()
