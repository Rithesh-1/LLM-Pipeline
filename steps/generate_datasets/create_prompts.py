from typing_extensions import Annotated

from llm_pipeline_system.application.dataset import generation
from llm_pipeline_system.domain.dataset import DatasetType
from llm_pipeline_system.domain.prompt import GenerateDatasetSamplesPrompt
from llm_pipeline_system.domain.types import DataCategory


def create_prompts(
    documents: Annotated[list, "queried_cleaned_documents"],
    dataset_type: Annotated[DatasetType, "dataset_type"],
    llm_provider: Annotated[generation.LLMProvider, "llm_provider"] = generation.LLMProvider.OPENAI, # Added llm_provider
) -> Annotated[dict[DataCategory, list[GenerateDatasetSamplesPrompt]], "prompts"]:
    dataset_generator = generation.get_dataset_generator(dataset_type)
    grouped_prompts = dataset_generator.get_prompts(documents, llm_provider) # Pass llm_provider

    return grouped_prompts


def _get_metadata(grouped_prompts: dict[DataCategory, list[GenerateDatasetSamplesPrompt]]) -> dict:
    prompt_categories = list(grouped_prompts.keys())
    prompt_num_samples = {category: len(prompts) for category, prompts in grouped_prompts.items()}

    return {"data_categories": prompt_categories, "data_categories_num_prompts": prompt_num_samples}
