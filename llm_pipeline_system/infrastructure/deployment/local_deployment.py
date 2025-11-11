"""
Local deployment script as a free alternative to AWS SageMaker.
Provides model deployment using local inference and Hugging Face Spaces.
"""

import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from llm_engineering.infrastructure.huggingface import HuggingFaceSpacesDeployment
from llm_engineering.model.inference.local_inference import LocalInferenceServer
from llm_engineering.settings import settings


class LocalDeploymentManager:
    """
    Deployment manager for local and Hugging Face Spaces deployment.
    Free alternative to AWS SageMaker deployment.
    """

    def __init__(self):
        self.local_server = None
        self.hf_deployment = None
        
        if settings.HUGGINGFACE_ACCESS_TOKEN:
            self.hf_deployment = HuggingFaceSpacesDeployment()

    def deploy_locally(
        self,
        model_id: Optional[str] = None,
        port: Optional[int] = None,
        background: bool = True,
    ) -> Dict[str, any]:
        """
        Deploy model locally using FastAPI server.
        
        Args:
            model_id: Hugging Face model ID
            port: Port to run the server on
            background: Whether to run in background
            
        Returns:
            Deployment information
        """
        if model_id is None:
            model_id = settings.HF_MODEL_ID
        
        if port is None:
            port = settings.LOCAL_INFERENCE_PORT
        
        logger.info(f"Deploying model locally: {model_id}")
        logger.info(f"Server will be available at: http://localhost:{port}")
        
        try:
            self.local_server = LocalInferenceServer(model_id=model_id, port=port)
            
            if background:
                # Run server in background using subprocess
                script_content = f'''
import sys
sys.path.append("{Path(__file__).parent.parent.parent}")

from llm_engineering.model.inference.local_inference import LocalInferenceServer

server = LocalInferenceServer("{model_id}", {port})
server.start_server()
'''
                
                script_file = Path("temp_server.py")
                script_file.write_text(script_content)
                
                process = subprocess.Popen(
                    ["python", str(script_file)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                
                # Wait a bit for server to start
                time.sleep(5)
                
                # Clean up temp file
                script_file.unlink(missing_ok=True)
                
                return {
                    "status": "deployed",
                    "deployment_type": "local",
                    "model_id": model_id,
                    "endpoint_url": f"http://localhost:{port}",
                    "health_check_url": f"http://localhost:{port}/health",
                    "process_id": process.pid,
                }
            else:
                # Run server in foreground
                self.local_server.start_server()
                return {
                    "status": "running",
                    "deployment_type": "local",
                    "model_id": model_id,
                    "endpoint_url": f"http://localhost:{port}",
                }
                
        except Exception as e:
            logger.error(f"Local deployment failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "deployment_type": "local",
            }

    def deploy_to_huggingface_spaces(
        self,
        space_name: Optional[str] = None,
        model_id: Optional[str] = None,
        hardware: str = "cpu-basic",
        private: bool = False,
    ) -> Dict[str, any]:
        """
        Deploy model to Hugging Face Spaces.
        
        Args:
            space_name: Name for the space
            model_id: Hugging Face model ID
            hardware: Hardware type (cpu-basic is free)
            private: Whether space should be private
            
        Returns:
            Deployment information
        """
        if not self.hf_deployment:
            return {
                "status": "failed",
                "error": "HUGGINGFACE_ACCESS_TOKEN is required for Spaces deployment",
                "deployment_type": "huggingface_spaces",
            }
        
        if space_name is None:
            space_name = settings.HF_SPACE_NAME
        
        if model_id is None:
            model_id = settings.HF_MODEL_ID
        
        logger.info(f"Deploying model to Hugging Face Spaces: {space_name}")
        logger.info(f"Model: {model_id}")
        logger.info(f"Hardware: {hardware}")
        
        try:
            # Create and deploy to space
            space_url = self.hf_deployment.create_space(
                space_name=space_name,
                model_id=model_id,
                hardware=hardware,
                private=private,
            )
            
            # Deploy the model
            deployed_url = self.hf_deployment.deploy_model_to_space(
                space_name=space_name,
                model_id=model_id,
            )
            
            return {
                "status": "deployed",
                "deployment_type": "huggingface_spaces",
                "model_id": model_id,
                "space_name": space_name,
                "space_url": deployed_url,
                "hardware": hardware,
            }
            
        except Exception as e:
            logger.error(f"Hugging Face Spaces deployment failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "deployment_type": "huggingface_spaces",
            }

    def get_deployment_status(self, deployment_info: Dict[str, any]) -> Dict[str, any]:
        """Get status of a deployment."""
        
        deployment_type = deployment_info.get("deployment_type")
        
        if deployment_type == "local":
            return self._get_local_status(deployment_info)
        elif deployment_type == "huggingface_spaces":
            return self._get_hf_spaces_status(deployment_info)
        else:
            return {"status": "unknown", "error": "Unknown deployment type"}

    def _get_local_status(self, deployment_info: Dict[str, any]) -> Dict[str, any]:
        """Get status of local deployment."""
        try:
            import requests
            
            health_url = deployment_info.get("health_check_url")
            if not health_url:
                return {"status": "unknown", "error": "No health check URL"}
            
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return {"status": "healthy", "response": response.json()}
            else:
                return {"status": "unhealthy", "status_code": response.status_code}
                
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    def _get_hf_spaces_status(self, deployment_info: Dict[str, any]) -> Dict[str, any]:
        """Get status of Hugging Face Spaces deployment."""
        if not self.hf_deployment:
            return {"status": "error", "error": "HF deployment not available"}
        
        space_name = deployment_info.get("space_name")
        if not space_name:
            return {"status": "error", "error": "No space name provided"}
        
        return self.hf_deployment.get_space_status(space_name)

    def delete_deployment(self, deployment_info: Dict[str, any]) -> bool:
        """Delete a deployment."""
        
        deployment_type = deployment_info.get("deployment_type")
        
        if deployment_type == "local":
            return self._delete_local_deployment(deployment_info)
        elif deployment_type == "huggingface_spaces":
            return self._delete_hf_spaces_deployment(deployment_info)
        else:
            logger.error(f"Unknown deployment type: {deployment_type}")
            return False

    def _delete_local_deployment(self, deployment_info: Dict[str, any]) -> bool:
        """Delete local deployment."""
        try:
            process_id = deployment_info.get("process_id")
            if process_id:
                import psutil
                process = psutil.Process(process_id)
                process.terminate()
                logger.info(f"Terminated local server process: {process_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete local deployment: {e}")
            return False

    def _delete_hf_spaces_deployment(self, deployment_info: Dict[str, any]) -> bool:
        """Delete Hugging Face Spaces deployment."""
        if not self.hf_deployment:
            return False
        
        space_name = deployment_info.get("space_name")
        if not space_name:
            return False
        
        return self.hf_deployment.delete_space(space_name)


def deploy_model(
    deployment_type: str = "local",
    model_id: Optional[str] = None,
    **kwargs
) -> Dict[str, any]:
    """
    Convenience function to deploy a model.
    
    Args:
        deployment_type: "local" or "huggingface_spaces"
        model_id: Hugging Face model ID
        **kwargs: Additional deployment parameters
        
    Returns:
        Deployment information
    """
    manager = LocalDeploymentManager()
    
    if deployment_type == "local":
        return manager.deploy_locally(model_id=model_id, **kwargs)
    elif deployment_type == "huggingface_spaces":
        return manager.deploy_to_huggingface_spaces(model_id=model_id, **kwargs)
    else:
        return {
            "status": "failed",
            "error": f"Unknown deployment type: {deployment_type}",
        }


if __name__ == "__main__":
    # Example usage
    logger.info("Starting model deployment")
    
    # Deploy locally
    local_deployment = deploy_model("local")
    logger.info(f"Local deployment: {local_deployment}")
    
    # Optionally deploy to Hugging Face Spaces
    if settings.HUGGINGFACE_ACCESS_TOKEN:
        hf_deployment = deploy_model("huggingface_spaces")
        logger.info(f"HF Spaces deployment: {hf_deployment}")
    else:
        logger.info("Skipping HF Spaces deployment (no token provided)")
