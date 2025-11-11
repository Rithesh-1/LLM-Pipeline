"""
Deployment infrastructure module.
Free alternatives to AWS deployment services.
"""

from .local_deployment import LocalDeploymentManager, deploy_model

__all__ = ["LocalDeploymentManager", "deploy_model"]
