from pathlib import Path

from loguru import logger


def export_artifact_to_json(artifact_names: list[str], output_dir: Path = Path("output")) -> None:
    """
    Export artifacts to JSON files. 
    Note: This function has been simplified after ZenML removal.
    Artifacts are now expected to be available through other means.
    """
    logger.warning("export_artifact_to_json function has been simplified after ZenML removal.")
    logger.info("Please use the MLflow-based run_mlflow.py pipeline for artifact management.")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for artifact_name in artifact_names:
        logger.info(f"Artifact export for '{artifact_name}' requires manual implementation without ZenML.")
