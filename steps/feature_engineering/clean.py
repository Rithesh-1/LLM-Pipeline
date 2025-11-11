import mlflow
from loguru import logger

from llm_pipeline_system.application.preprocessing import CleaningDispatcher
from llm_pipeline_system.domain.cleaned_documents import CleanedDocument


def clean_documents(
    documents: list,
) -> list:
    cleaned_documents = []
    for document in documents:
        cleaned_document = CleaningDispatcher.dispatch(document)
        cleaned_documents.append(cleaned_document)

    # Log metadata to MLflow
    try:
        metadata = _get_metadata(cleaned_documents)
        mlflow.log_dict(metadata, "cleaned_documents_metadata.json")
        mlflow.log_metric("total_documents_cleaned", len(cleaned_documents))
    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")

    return cleaned_documents


def _get_metadata(cleaned_documents: list[CleanedDocument]) -> dict:
    metadata = {"num_documents": len(cleaned_documents)}
    for document in cleaned_documents:
        category = document.get_category()
        if category not in metadata:
            metadata[category] = {}
        if "authors" not in metadata[category]:
            metadata[category]["authors"] = list()

        metadata[category]["num_documents"] = metadata[category].get("num_documents", 0) + 1
        metadata[category]["authors"].append(document.author_full_name)

    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))

    return metadata
