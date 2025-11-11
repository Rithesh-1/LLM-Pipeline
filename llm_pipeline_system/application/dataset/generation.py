from abc import ABC, abstractmethod
from enum import Enum

import tiktoken
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from loguru import logger

from llm_pipeline_system import domain
from llm_pipeline_system.application import utils
from llm_pipeline_system.domain.cleaned_documents import CleanedDocument
from llm_pipeline_system.domain.dataset import DatasetType, TrainTestSplit
from llm_pipeline_system.domain.prompt import GenerateDatasetSamplesPrompt, Prompt
from llm_pipeline_system.domain.types import DataCategory
from llm_pipeline_system.settings import settings

from . import constants
from . import utils as generation_utils
from .output_parsers import ListPydanticOutputParser


class LLMProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


class DatasetGenerator(ABC):
    # tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL_ID) # Removed class-level tokenizer
    dataset_type: DatasetType | None = None

    system_prompt_template = """You are a helpful assistant who generates {dataset_format} based on the given context. \
Provide your response in JSON format.
"""
    prompt_template_str: str | None = None

    @classmethod
    def _get_tokenizer(cls, llm: BaseChatModel):
        if isinstance(llm, ChatOpenAI):
            return tiktoken.encoding_for_model(settings.OPENAI_MODEL_ID)
        elif isinstance(llm, ChatGoogleGenerativeAI):
            try:
                import google.generativeai as genai

                # This uses the underlying model's count_tokens method
                return lambda text: llm.get_num_tokens(text)
            except (ImportError, AttributeError) as e:
                logger.warning(
                    f"Could not use Google SDK for token counting due to: {e}. Falling back to character count."
                )
                return len
        else:
            logger.warning(
                f"Unsupported LLM provider '{type(llm).__name__}' for tokenization. Falling back to character count."
            )
            return len

    @classmethod
    def get_system_prompt(cls) -> Prompt:
        assert cls.dataset_type is not None, "Dataset type must be set before calling get_system_prompt()"

        dataset_format = (
            "instruction-answer pairs" if cls.dataset_type == DatasetType.INSTRUCTION else "instruction-answer triples"
        )
        input_variables = {
            "dataset_format": dataset_format,
        }
        system_prompt = cls.system_prompt_template.format(**input_variables)

        return Prompt(
            template=cls.system_prompt_template,
            input_variables=input_variables,
            content=system_prompt,
        )

    @classmethod
    def get_prompts(
        cls, documents: list[CleanedDocument], llm: BaseChatModel
    ) -> dict[DataCategory, list[GenerateDatasetSamplesPrompt]]:
        documents = generation_utils.extract_substrings(documents)

        grouped_prompts = {}
        grouped_cleaned_documents = CleanedDocument.group_by_category(documents)
        for category, category_documents in grouped_cleaned_documents.items():
            category_prompts = [cls.get_prompt(document, llm) for document in category_documents]
            grouped_prompts[category] = category_prompts

        return grouped_prompts

    @classmethod
    def get_prompt(cls, document: CleanedDocument, llm: BaseChatModel) -> GenerateDatasetSamplesPrompt:
        assert cls.prompt_template_str is not None, "Prompt template must be set before calling get_prompt()"

        data_category = document.get_category()

        prompt_template = PromptTemplate.from_template(
            template=cls.prompt_template_str,
            template_format="jinja2",
        )
        input_variables = {
            "extract": document.content,
        }
        prompt = prompt_template.format(**input_variables)

        tokenizer = cls._get_tokenizer(llm)
        num_tokens = tokenizer(prompt)

        # Apply token window based on provider
        # This logic is tricky without a real tokenizer to decode.
        # A better approach is to let the API handle truncation if needed,
        # or use a more sophisticated truncation strategy.
        # For now, we will just count the tokens.
        # A simple character-based truncation could be added if necessary.
        max_tokens = getattr(settings, f"{llm.provider.upper()}_MAX_TOKEN_WINDOW", None)
        if max_tokens and num_tokens > max_tokens:
            logger.warning(f"Prompt for document {document.id} exceeds max token window. Truncation is not yet implemented robustly.")

        prompt = GenerateDatasetSamplesPrompt(
            template=prompt_template.template,
            input_variables=input_variables,
            content=prompt,
            num_tokens=num_tokens,
            data_category=data_category,
            document=document,
        )

        return prompt

    @classmethod
    def generate(
        cls,
        documents: list[CleanedDocument],
        test_size: float = 0.2,
        mock: bool = False,
        llm_provider: LLMProvider = LLMProvider.OPENAI,
    ) -> TrainTestSplit:
        assert cls.dataset_type is not None, "Dataset type must be set before calling generate()"
        def _to_langchain(
            prompt: GenerateDatasetSamplesPrompt,
        ) -> list[BaseMessage]:
            messages = [
                SystemMessage(content=cls.get_system_prompt().content),
                HumanMessage(content=prompt.content),
            ]

            return messages
        if mock:
            llm = FakeListLLM(responses=[constants.get_mocked_response(cls.dataset_type)])
        else:
            if llm_provider == LLMProvider.OPENAI:
                assert settings.OPENAI_API_KEY is not None, "OpenAI API key must be set to generate datasets"

                llm = ChatOpenAI(
                    model=settings.OPENAI_MODEL_ID,
                    api_key=settings.OPENAI_API_KEY,
                    max_tokens=2000 if cls.dataset_type == DatasetType.PREFERENCE else 1200,
                    temperature=0.7,
                )
            elif llm_provider == LLMProvider.GEMINI:
                assert settings.GOOGLE_API_KEY is not None, "Google API key must be set to generate datasets"

                llm = ChatGoogleGenerativeAI(
                    model="models/gemini-pro",  # Corrected model name
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_output_tokens=2000 if cls.dataset_type == DatasetType.PREFERENCE else 1200,
                    temperature=0.7,
                    convert_system_message_to_human=True,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        parser = ListPydanticOutputParser(pydantic_object=cls._get_dataset_sample_type())

        # Get prompts now that we have the llm object
        prompts = cls.get_prompts(documents, llm)

        chain = llm | parser
        
        datasets = {}
        for category, category_prompts in prompts.items():
            langchain_category_prompts = [_to_langchain(prompt) for prompt in category_prompts]
            batches = utils.misc.batch(langchain_category_prompts, size=24)

            flattened_instruct_dataset_samples = []
            for batch in batches:
                try:
                    batched_dataset_samples = chain.batch(batch, stop=None)

                    for instruct_dataset_sample_batch in batched_dataset_samples:
                        flattened_instruct_dataset_samples.extend(instruct_dataset_sample_batch)
                except OutputParserException:
                    logger.exception(f"Failed to parse the output JSON for a batch for category {category}")

            dataset = domain.dataset.build_dataset(
                dataset_type=cls.dataset_type, category=category, samples=flattened_instruct_dataset_samples
            )
            datasets[category] = dataset
            logger.info(f"Generated {len(dataset.samples)} samples for category '{category}'.")

        processed_datasets = cls.post_process_datasets(datasets, test_size=test_size)

        return processed_datasets

    @classmethod
    def _get_dataset_sample_type(
        cls,
    ) -> type[domain.dataset.InstructDatasetSample] | type[domain.dataset.PreferenceDatasetSample]:
        return (
            domain.dataset.InstructDatasetSample
            if cls.dataset_type == DatasetType.INSTRUCTION
            else domain.dataset.PreferenceDatasetSample
        )

    @classmethod
    @abstractmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, domain.dataset.InstructDataset], test_size: float
    ) -> TrainTestSplit:
        pass


class InstructionDatasetGenerator(DatasetGenerator):
    dataset_type = DatasetType.INSTRUCTION

    prompt_template_str = """Based on the following extract, generate five instruction-answer pairs. Each instruction \
must ask to write about a specific topic contained in the context. Each answer \
must provide a relevant paragraph based on the information found in the \
context. Only use concepts from the context to generate the instructions. \
Instructions must never explicitly mention a context, a system, a course, or an extract. \
Instructions must be self-contained and general. \
Answers must imitate the writing style of the context. \
    
Example instruction: Explain the concept of an LLM Twin. \
Example answer: An LLM Twin is essentially an AI character that mimics your writing style, personality, and voice. \
It's designed to write just like you by incorporating these elements into a language model. \
The idea is to create a digital replica of your writing habits using advanced AI techniques. \

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {"instruction": "...", "answer": "..."},
    ...
]

Extract:
{{extract}}
"""

    @classmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, domain.dataset.InstructDataset], test_size: float
    ) -> TrainTestSplit:
        train_test_split = generation_utils.create_instruct_train_test_split(
            datasets, test_size=test_size, random_state=42
        )

        return train_test_split


class PreferenceDatasetGenerator(DatasetGenerator):
    dataset_type = DatasetType.PREFERENCE

    prompt_template_str = """Based on the following extract, generate five instruction-answer triples. Each triple should consist of:
1. An instruction asking about a specific topic in the context.
2. A generated answer that attempts to answer the instruction based on the context, named as 'rejected'.
3. An extracted answer that is a relevant excerpt directly from the given context, named as 'chosen'.

Instructions must be self-contained and general, without explicitly mentioning a context, system, course, or extract.

Important:
- Ensure that the extracted answer, the chosen one, is a verbatim copy from the context, including all punctuation and apostrophes.
- Do not add any ellipsis (...) or [...]  to indicate skipped text in the extracted answer.
- If the relevant text is not continuous, use two separate sentences from the context instead of skipping text.

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {
        "instruction": "...",
        "rejected": "...",
        "chosen": "..."
    },
    ...
]

Extract:
{{extract}}
"""

    @classmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, domain.dataset.PreferenceDataset], test_size: float
    ) -> TrainTestSplit:
        datasets = generation_utils.filter_short_answers(datasets)
        datasets = generation_utils.filter_answer_format(datasets)

        remaining_samples = sum([dataset.num_samples for dataset in datasets.values()])
        logger.info(
            f"Filtered out short answers and answers with incorrect format. Remaining samples: {remaining_samples}"
        )

        train_test_split = generation_utils.create_preference_train_test_split(
            datasets, test_size=test_size, random_state=42
        )

        return train_test_split


def get_dataset_generator(dataset_type: DatasetType) -> type[DatasetGenerator]:
    if dataset_type == DatasetType.INSTRUCTION:
        return InstructionDatasetGenerator
    elif dataset_type == DatasetType.PREFERENCE:
        return PreferenceDatasetGenerator
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
