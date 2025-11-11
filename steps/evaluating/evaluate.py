from loguru import logger

from llm_pipeline_system.model.evaluation.local_evaluation import run_evaluation_on_local
from llm_pipeline_system.settings import settings


def evaluate(
    is_dummy: bool = False,
) -> None:
    """
    Run model evaluation using local inference (free alternative to SageMaker).
    """
    logger.info("Running evaluation using local inference (SageMaker alternative)")
    
    if settings.USE_LOCAL_INFERENCE:
        run_evaluation_on_local(is_dummy=is_dummy)
    else:
        # Fallback to SageMaker if explicitly configured (not recommended)
        logger.warning("SageMaker evaluation is deprecated. Consider using local evaluation.")
        from llm_pipeline_system.model.evaluation.sagemaker import run_evaluation_on_sagemaker
        run_evaluation_on_sagemaker(is_dummy=is_dummy)
