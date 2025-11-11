import mlflow
from datetime import datetime as dt
from pathlib import Path

import click
import yaml
from loguru import logger

from llm_pipeline_system import settings
from pipelines import (
    digital_data_etl,
    end_to_end_data,
    evaluating,
    export_artifact_to_json,
    feature_engineering,
    generate_datasets,
    training,
)
from llm_pipeline_system.application.dataset import generation # Added this import


@click.command(
    help="""
LLM Pipeline System project CLI v0.0.1. 

Main entry point for the pipeline execution. 
This entrypoint is where everything comes together.

Run the LLM Pipeline System project pipelines with various options.
Uses ClearML for unified experiment tracking and monitoring across all pipelines.

Run a pipeline with the required parameters. This executes
all steps in the pipeline in the correct order.

Examples:

  \b
  # Run the pipeline with default options
  python run.py
               
  \b
  # Run the pipeline without cache
  python run.py --no-cache
  
  \b
  # Run only the ETL pipeline
  python run.py --only-etl

"""
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--run-end-to-end-data",
    is_flag=True,
    default=False,
    help="Whether to run all the data pipelines in one go.",
)
@click.option(
    "--run-etl",
    is_flag=True,
    default=False,
    help="Whether to run the ETL pipeline.",
)
@click.option(
    "--run-export-artifact-to_json",
    is_flag=True,
    default=False,
    help="Whether to run the Artifact -> JSON pipeline",
)
@click.option(
    "--etl-config-filename",
    default="digital_data_etl.yaml",
    help="Filename of the ETL config file.",
)
@click.option(
    "--run-feature-engineering",
    is_flag=True,
    default=False,
    help="Whether to run the FE pipeline.",
)
@click.option(
    "--run-generate-instruct-datasets",
    is_flag=True,
    default=False,
    help="Whether to run the instruct dataset generation pipeline.",
)
@click.option(
    "--run-generate-preference-datasets",
    is_flag=True,
    default=False,
    help="Whether to run the preference dataset generation pipeline.",
)
@click.option(
    "--run-training",
    is_flag=True,
    default=False,
    help="Whether to run the training pipeline.",
)
@click.option(
    "--run-evaluation",
    is_flag=True,
    default=False,
    help="Whether to run the evaluation pipeline.",
)
@click.option(
    "--export-settings",
    is_flag=True,
    default=False,
    help="Whether to export your settings (deprecated - now uses .env file).",
)
def main(
    no_cache: bool = False,
    run_end_to_end_data: bool = False,
    run_etl: bool = False,
    etl_config_filename: str = "digital_data_etl.yaml",
    run_export_artifact_to_json: bool = False,
    run_feature_engineering: bool = False,
    run_generate_instruct_datasets: bool = False,
    run_generate_preference_datasets: bool = False,
    run_training: bool = False,
    run_evaluation: bool = False,
    export_settings: bool = False,
) -> None:
    mlflow.set_tracking_uri("http://localhost:5000")
    assert (
        run_end_to_end_data
        or run_etl
        or run_export_artifact_to_json
        or run_feature_engineering
        or run_generate_instruct_datasets
        or run_generate_preference_datasets
        or run_training
        or run_evaluation
        or export_settings
    ), "Please specify an action to run."

    if export_settings:
        logger.info("Settings export functionality removed. Use .env file for configuration.")
        settings.export()

    pipeline_args = {
        "enable_cache": not no_cache,
    }
    root_dir = Path(__file__).resolve().parent.parent

    if run_end_to_end_data:
        config_path = root_dir / "configs" / "end_to_end_data.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
        
        # Read config file and extract parameters
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        parameters = config.get('parameters', {})
        author_links = parameters.get('author_links', [])
        test_split_size = parameters.get('test_split_size', 0.1)
        push_to_huggingface = parameters.get('push_to_huggingface', False)
        dataset_id = parameters.get('dataset_id')
        mock = parameters.get('mock', False)
        
        assert author_links, f"author_links not found in config: {config_path}"
        
        logger.info(f"Running end-to-end data pipeline for {len(author_links)} authors, mock={mock}")
        end_to_end_data(
            author_links=author_links,
            test_split_size=test_split_size,
            push_to_huggingface=push_to_huggingface,
            dataset_id=dataset_id,
            mock=mock
        )

    if run_etl:
        config_path = root_dir / "configs" / etl_config_filename
        assert config_path.exists(), f"Config file not found: {config_path}"
        
        # Read config file and extract parameters
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        parameters = config.get('parameters', {})
        user_full_name = parameters.get('user_full_name')
        links = parameters.get('links', [])
        
        assert user_full_name, f"user_full_name not found in config: {config_path}"
        assert links, f"links not found in config: {config_path}"
        
        logger.info(f"Running ETL for user: {user_full_name} with {len(links)} links")
        digital_data_etl(user_full_name=user_full_name, links=links)

    if run_export_artifact_to_json:
        run_args_etl = {}
        pipeline_args["config_path"] = root_dir / "configs" / "export_artifact_to_json.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"export_artifact_to_json_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        export_artifact_to_json(**run_args_etl)

    if run_feature_engineering:
        config_path = root_dir / "configs" / "feature_engineering.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
        
        # Read config file and extract parameters
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        parameters = config.get('parameters', {})
        author_full_names = parameters.get('author_full_names', [])
        
        assert author_full_names, f"author_full_names not found in config: {config_path}"
        
        logger.info(f"Running feature engineering for authors: {author_full_names}")
        feature_engineering(author_full_names=author_full_names)

    if run_generate_instruct_datasets:
        config_path = root_dir / "configs" / "generate_instruct_datasets.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
        
        # Read config file and extract parameters
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        parameters = config.get('parameters', {})
        
        # Import DatasetType here to avoid circular imports
        from llm_pipeline_system.domain.dataset import DatasetType
        
        dataset_type_str = parameters.get('dataset_type', 'instruction')
        dataset_type = DatasetType.INSTRUCTION if dataset_type_str == 'instruction' else DatasetType.PREFERENCE
        test_split_size = parameters.get('test_split_size', 0.1)
        push_to_huggingface = parameters.get('push_to_huggingface', False)
        dataset_id = parameters.get('dataset_id')
        mock = parameters.get('mock', False)
        
        parameters['llm_provider'] = 'gemini' # Force Gemini
        parameters['GEMINI_MODEL_ID'] = 'gemini-pro' # Force gemini-pro
        
        llm_provider_str = parameters.get('llm_provider', 'openai') # Extract llm_provider
        llm_provider = generation.LLMProvider(llm_provider_str) # Convert to Enum
        
        logger.info(f"Running generate datasets: type={dataset_type_str}, mock={mock}")
        generate_datasets(
            dataset_type=dataset_type,
            test_split_size=test_split_size,
            push_to_huggingface=push_to_huggingface,
            dataset_id=dataset_id,
            mock=mock,
            llm_provider=llm_provider # Pass llm_provider
        )

    if run_generate_preference_datasets:
        config_path = root_dir / "configs" / "generate_preference_datasets.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
        
        # Read config file and extract parameters
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        parameters = config.get('parameters', {})
        
        # Import DatasetType here to avoid circular imports
        from llm_pipeline_system.domain.dataset import DatasetType
        
        dataset_type_str = parameters.get('dataset_type', 'preference')
        dataset_type = DatasetType.PREFERENCE if dataset_type_str == 'preference' else DatasetType.INSTRUCTION
        test_split_size = parameters.get('test_split_size', 0.05)
        push_to_huggingface = parameters.get('push_to_huggingface', False)
        dataset_id = parameters.get('dataset_id')
        mock = parameters.get('mock', False)
        
        parameters['llm_provider'] = 'gemini' # Force Gemini
        parameters['GEMINI_MODEL_ID'] = 'gemini-pro' # Force gemini-pro
        
        llm_provider_str = parameters.get('llm_provider', 'openai') # Extract llm_provider
        llm_provider = generation.LLMProvider(llm_provider_str) # Convert to Enum
        
        logger.info(f"Running generate datasets: type={dataset_type_str}, mock={mock}")
        generate_datasets(
            dataset_type=dataset_type,
            test_split_size=test_split_size,
            push_to_huggingface=push_to_huggingface,
            dataset_id=dataset_id,
            mock=mock,
            llm_provider=llm_provider # Pass llm_provider
        )

    if run_training:
        run_args_cd = {}
        pipeline_args["config_path"] = root_dir / "configs" / "training.yaml"
        pipeline_args["run_name"] = f"training_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        training(**run_args_cd)

    if run_evaluation:
        run_args_cd = {}
        pipeline_args["config_path"] = root_dir / "configs" / "evaluating.yaml"
        pipeline_args["run_name"] = f"evaluation_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        evaluating(**run_args_cd)


if __name__ == "__main__":
    main()
