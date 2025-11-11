import mlflow
from loguru import logger

from llm_pipeline_system.application import utils
from llm_pipeline_system.domain.base import VectorBaseDocument


def load_to_vector_db(
    documents: list,
) -> bool:
    logger.info(f"Loading {len(documents)} documents into the vector database.")

    grouped_documents = VectorBaseDocument.group_by_class(documents)
    for document_class, documents in grouped_documents.items():
        logger.info(f"Loading documents into {document_class.get_collection_name()}")
        for documents_batch in utils.misc.batch(documents, size=4):
            try:
                document_class.bulk_insert(documents_batch)
            except Exception:
                logger.error(f"Failed to insert documents into {document_class.get_collection_name()}")

                return False

    # Log to MLflow
    try:
        mlflow.log_metric("documents_loaded_to_vector_db", len(documents))
    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")

    return True
