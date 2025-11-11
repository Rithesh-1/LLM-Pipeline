import os
from typing import Dict, Union

from loguru import logger


def get_api_limits() -> Dict[str, Union[int, None]]:
    """
    Retrieves API rate and token limits from environment variables.

    This function reads the following environment variables:
    - API_RATE_LIMIT_PER_MINUTE: The number of API calls allowed per minute.
    - API_TOKEN_LIMIT_PER_MINUTE: The number of tokens that can be processed per minute.

    If the environment variables are not set or contain an invalid integer, a warning
    is logged, and the corresponding limit is set to None (unlimited).

    Returns:
        A dictionary containing the configured limits, e.g.,
        {'rate_limit': 60, 'token_limit': 40000}.
    """
    rate_limit_str = os.getenv("API_RATE_LIMIT_PER_MINUTE")
    token_limit_str = os.getenv("API_TOKEN_LIMIT_PER_MINUTE")

    rate_limit = None
    if rate_limit_str and rate_limit_str.isdigit():
        rate_limit = int(rate_limit_str)
    elif rate_limit_str:
        logger.warning(
            f"Invalid value for API_RATE_LIMIT_PER_MINUTE: '{rate_limit_str}'. Limit will be disabled."
        )

    token_limit = None
    if token_limit_str and token_limit_str.isdigit():
        token_limit = int(token_limit_str)
    elif token_limit_str:
        logger.warning(
            f"Invalid value for API_TOKEN_LIMIT_PER_MINUTE: '{token_limit_str}'. Limit will be disabled."
        )

    return {"rate_limit": rate_limit, "token_limit": token_limit}