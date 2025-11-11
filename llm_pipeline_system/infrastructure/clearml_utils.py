import os
from typing import Optional

from clearml import Task
from loguru import logger

from llm_pipeline_system.settings import settings


def configure_clearml() -> Optional[Task]:
    """
    Configure ClearML for monitoring and experiment tracking.
    Returns the initialized Task if successful, None otherwise.
    """
    if not os.path.exists(os.path.join(os.path.expanduser("~"), "clearml.conf")):
        logger.warning("ClearML configuration file not found. Running in offline mode.")
        Task.set_offline(offline_mode=True)

    try:
        # Initialize ClearML Task for monitoring
        task = Task.init(
            project_name=settings.CLEARML_PROJECT or "LLM-Pipeline-System",
            task_name="RAG-Inference-Monitoring",
            auto_connect_frameworks=True,
            auto_connect_arg_parser=True,
        )
        
        # Set task type for inference monitoring
        task.set_system_tags(["inference", "rag", "monitoring"])
        
        logger.info("ClearML configured successfully for RAG monitoring.")
        return task
        
    except Exception as e:
        logger.warning(f"Failed to initialize ClearML: {e}. Continuing without monitoring.")
        return None


def get_or_create_clearml_task(task_name: str = "RAG-Operation") -> Optional[Task]:
    """
    Get current ClearML task or create a new one for specific operations.
    """
    if not os.path.exists(os.path.join(os.path.expanduser("~"), "clearml.conf")):
        logger.warning("ClearML configuration file not found. Running in offline mode.")
        Task.set_offline(offline_mode=True)

    try:
        # Try to get current task
        current_task = Task.current_task()
        if current_task:
            return current_task
            
        # Create new task if none exists
        task = Task.init(
            project_name=settings.CLEARML_PROJECT or "LLM-Pipeline-System",
            task_name=task_name,
            auto_connect_frameworks=False,
        )
        return task
        
    except Exception as e:
        logger.warning(f"Failed to get/create ClearML task: {e}")
        return None
