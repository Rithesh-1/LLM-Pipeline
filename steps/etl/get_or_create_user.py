import mlflow
from loguru import logger

from llm_pipeline_system.application import utils
from llm_pipeline_system.domain.documents import UserDocument


def get_or_create_user(user_full_name: str) -> UserDocument:
    logger.info(f"Getting or creating user: {user_full_name}")

    first_name, last_name = utils.split_user_full_name(user_full_name)

    user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

    # Log metadata to MLflow
    try:
        metadata = _get_metadata(user_full_name, user)
        mlflow.log_dict(metadata, "user_metadata.json")
        mlflow.log_param("user_full_name", user_full_name)
        mlflow.log_param("user_id", str(user.id))
    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")
        # Continue execution even if MLflow logging fails

    return user


def _get_metadata(user_full_name: str, user: UserDocument) -> dict:
    return {
        "query": {
            "user_full_name": user_full_name,
        },
        "retrieved": {
            "user_id": str(user.id),
            "first_name": user.first_name,
            "last_name": user.last_name,
        },
    }
