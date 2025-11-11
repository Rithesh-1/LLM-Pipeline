"""
Hugging Face infrastructure module.
Free alternative to AWS services.
"""

from .spaces_deployment import HuggingFaceInference, HuggingFaceSpacesDeployment

__all__ = ["HuggingFaceSpacesDeployment", "HuggingFaceInference"]
