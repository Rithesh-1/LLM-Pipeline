from clearml import Task
from clearml.backend_api.session.defs import MissingConfigError
from loguru import logger

from llm_pipeline_system.model.finetuning.local_finetuning import run_local_finetuning


def train(
    finetuning_type: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    learning_rate: float,
    dataset_huggingface_workspace: str = "mlabonne",
    is_dummy: bool = False,
) -> None:
    try:
        task = Task.init(project_name="LLM-Pipeline-System", task_name="Training")
    except MissingConfigError:
        logger.warning("ClearML keys not found. Running in offline mode.")
        Task.set_offline(offline_mode=True)
        task = Task.init(project_name="LLM-Pipeline-System", task_name="Training")

    run_local_finetuning(
        finetuning_type=finetuning_type,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        dataset_huggingface_workspace=dataset_huggingface_workspace,
        is_dummy=is_dummy,
    )
