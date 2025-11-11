import os

from google import genai
from loguru import logger


def get_gemini_client() -> genai.GenerativeModel:
    """
    Initializes and returns a Gemini client.

    Returns:
        An initialized Gemini client.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    genai.configure(api_key=api_key)
    model_id = os.environ.get("GEMINI_MODEL_ID", "gemini-1.5-flash")

    return genai.GenerativeModel(model_id)


def generate_content(model: genai.GenerativeModel, prompt: str) -> str:
    """
    Generates content using the provided Gemini model and prompt.

    Args:
        model: The Gemini model client.
        prompt: The text prompt to send to the model.

    Returns:
        The generated text content.
    """
    logger.info("Generating content with Gemini...")
    try:
        response = model.generate_content(prompt)
        if response.parts:
            return response.text
        else:
            logger.warning("Received an empty response from Gemini.")
            return ""
    except Exception as e:
        logger.error(f"An error occurred while generating content with Gemini: {e}")
        return ""