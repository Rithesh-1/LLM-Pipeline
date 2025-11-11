from typing import Any

from typing_extensions import Annotated

from llm_pipeline_system.application.dataset import generation
from llm_pipeline_system.domain.dataset import DatasetType, PreferenceTrainTestSplit
from llm_pipeline_system.domain.prompt import GenerateDatasetSamplesPrompt
from llm_pipeline_system.domain.types import DataCategory


def generate_preference_dataset(
    prompts: Annotated[dict[DataCategory, list[GenerateDatasetSamplesPrompt]], "prompts"],
    test_split_size: Annotated[float, "test_split_size"],
    mock: Annotated[bool, "mock_generation"] = False,
) -> PreferenceTrainTestSplit:
    dataset_generator = generation.get_dataset_generator(DatasetType.PREFERENCE)
    datasets = dataset_generator.generate(prompts, test_size=test_split_size, mock=mock)

    return datasets


def _get_metadata_preference_dataset(datasets: PreferenceTrainTestSplit) -> dict[str, Any]:
    instruct_dataset_categories = list(datasets.train.keys())
    train_num_samples = {
        category: instruct_dataset.num_samples for category, instruct_dataset in datasets.train.items()
    }
    test_num_samples = {category: instruct_dataset.num_samples for category, instruct_dataset in datasets.test.items()}

    return {
        "data_categories": instruct_dataset_categories,
        "test_split_size": datasets.test_split_size,
        "train_num_samples_per_category": train_num_samples,
        "test_num_samples_per_category": test_num_samples,
    }
