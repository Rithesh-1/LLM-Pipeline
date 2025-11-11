from loguru import logger
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    # --- Required settings even when working locally. ---

    # OpenAI API
    OPENAI_MODEL_ID: str

    OPENAI_API_KEY: str | None = None

    # Google API
    GOOGLE_API_KEY: str | None = None
    GEMINI_MODEL_ID: str = "gemini-pro"
    GEMINI_MAX_TOKEN_WINDOW: int = 32768 # A reasonable default for Gemini 1.5 Flash

    # LLM Provider
    LLM_PROVIDER: str = "openai"

    # Huggingface API
    HUGGINGFACE_ACCESS_TOKEN: str | None = None



    # --- Required settings when deploying the code. ---
    # --- Otherwise, default values values work fine. ---

    # MongoDB database
    DATABASE_HOST: str = "mongodb://llm_pipeline_system:llm_pipeline_system@127.0.0.1:27017"
    DATABASE_NAME: str = "llm_pipeline_system"

    # Qdrant vector database
    USE_QDRANT_CLOUD: bool = False
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    QDRANT_CLOUD_URL: str = "str"
    QDRANT_APIKEY: str | None = None

    # --- Optional settings used to tweak the code. ---

    # API Rate Limiting
    API_RATE_LIMIT_PER_MINUTE: int = 15
    API_TOKEN_LIMIT_PER_MINUTE: int = 1000000

    # Model Configuration (Free alternatives to AWS SageMaker)
    HF_MODEL_ID: str = "TwinLlama-3.1-8B-DPO"
    
    # Local Inference Settings
    USE_LOCAL_INFERENCE: bool = True  # Use local inference instead of cloud
    LOCAL_INFERENCE_DEVICE: str = "auto"  # "auto", "cpu", "cuda"
    LOCAL_INFERENCE_PORT: int = 8000
    LOAD_IN_8BIT: bool = False  # Memory optimization
    LOAD_IN_4BIT: bool = False  # More aggressive memory optimization
    
    # Hugging Face Spaces Settings (Free cloud alternative)
    USE_HF_SPACES: bool = False  # Use HF Spaces for deployment
    HF_SPACE_NAME: str = "llm-twin-inference"
    HF_SPACE_HARDWARE: str = "cpu-basic"  # Free tier
    HF_SPACE_SDK: str = "gradio"
    
    # Inference Parameters
    MAX_INPUT_LENGTH: int = 2048
    MAX_TOTAL_TOKENS: int = 4096
    MAX_BATCH_TOTAL_TOKENS: int = 4096
    TEMPERATURE_INFERENCE: float = 0.01
    TOP_P_INFERENCE: float = 0.9
    MAX_NEW_TOKENS_INFERENCE: int = 150
    


    # RAG
    TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    RAG_MODEL_DEVICE: str = "cpu"

    # LinkedIn Credentials
    LINKEDIN_USERNAME: str | None = None
    LINKEDIN_PASSWORD: str | None = None

    # ClearML Configuration (for experiment tracking and monitoring)
    CLEARML_PROJECT: str | None = None

    @property
    def OPENAI_MAX_TOKEN_WINDOW(self) -> int:
        official_max_token_window = {
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }.get(self.OPENAI_MODEL_ID, 128000)

        max_token_window = int(official_max_token_window * 0.90)

        return max_token_window

    @classmethod
    def load_settings(cls) -> "Settings":
        """
        Loads the settings from the .env file and default values.

        Returns:
            Settings: The initialized settings object.
        """

        logger.info("Loading settings from the .env file.")
        settings = Settings()

        return settings

    def export(self) -> None:
        """
        Exports the settings to a file (MLflow compatible).
        """
        
        logger.info("Settings export functionality removed. Use .env file for configuration.")


settings = Settings.load_settings()
