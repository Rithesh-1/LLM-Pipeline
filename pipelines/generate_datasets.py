from llm_pipeline_system.domain.dataset import DatasetType
from steps import generate_datasets as cd_steps
from llm_pipeline_system.application.dataset import generation # Added this import


def generate_datasets(
    dataset_type: DatasetType = DatasetType.INSTRUCTION,
    test_split_size: float = 0.1,
    push_to_huggingface: bool = False,
    dataset_id: str | None = None,
    mock: bool = False,
    llm_provider: generation.LLMProvider = generation.LLMProvider.OPENAI, # Added llm_provider
    wait_for: str | list[str] | None = None,
) -> None:
    cleaned_documents = cd_steps.query_feature_store()
    prompts = cd_steps.create_prompts(documents=cleaned_documents, dataset_type=dataset_type, llm_provider=llm_provider)
    if dataset_type == DatasetType.INSTRUCTION:
        dataset = cd_steps.generate_intruction_dataset(prompts=prompts, test_split_size=test_split_size, mock=mock, llm_provider=llm_provider) # Pass llm_provider
    elif dataset_type == DatasetType.PREFERENCE:
        dataset = cd_steps.generate_preference_dataset(prompts=prompts, test_split_size=test_split_size, mock=mock, llm_provider=llm_provider) # Pass llm_provider
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    if push_to_huggingface:
        cd_steps.push_to_huggingface(dataset=dataset, dataset_id=dataset_id)
