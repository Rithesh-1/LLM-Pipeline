"""
Hugging Face Spaces deployment as a free alternative to AWS SageMaker.
Provides model deployment and inference capabilities using HF Spaces.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from huggingface_hub import HfApi, create_repo
from loguru import logger

from llm_engineering.settings import settings


class HuggingFaceSpacesDeployment:
    """
    Deployment service for Hugging Face Spaces as an alternative to AWS SageMaker.
    """

    def __init__(self):
        self.api = HfApi()
        self.token = settings.HUGGINGFACE_ACCESS_TOKEN
        if not self.token:
            raise ValueError("HUGGINGFACE_ACCESS_TOKEN is required for Spaces deployment")

    def create_space(
        self,
        space_name: str,
        model_id: str,
        hardware: str = "cpu-basic",  # Free tier
        sdk: str = "gradio",
        private: bool = False,
    ) -> str:
        """
        Create a new Hugging Face Space for model deployment.
        
        Args:
            space_name: Name of the space
            model_id: Hugging Face model ID to deploy
            hardware: Hardware type (cpu-basic is free)
            sdk: SDK to use (gradio, streamlit, etc.)
            private: Whether the space should be private
            
        Returns:
            Space URL
        """
        try:
            # Create the space
            space_url = create_repo(
                repo_id=f"{self.api.whoami(token=self.token)['name']}/{space_name}",
                repo_type="space",
                token=self.token,
                private=private,
                space_sdk=sdk,
                space_hardware=hardware,
            )
            
            logger.info(f"Created Hugging Face Space: {space_url}")
            return space_url
            
        except Exception as e:
            logger.error(f"Failed to create Hugging Face Space: {e}")
            raise

    def deploy_model_to_space(
        self,
        space_name: str,
        model_id: str,
        app_file_content: Optional[str] = None,
    ) -> str:
        """
        Deploy a model to Hugging Face Spaces.
        
        Args:
            space_name: Name of the space
            model_id: Hugging Face model ID
            app_file_content: Custom app.py content, if None uses default
            
        Returns:
            Deployed space URL
        """
        try:
            username = self.api.whoami(token=self.token)['name']
            repo_id = f"{username}/{space_name}"
            
            # Create default app.py if not provided
            if app_file_content is None:
                app_file_content = self._generate_default_app(model_id)
            
            # Create requirements.txt
            requirements_content = self._generate_requirements()
            
            # Upload files to the space
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write app.py
                app_file = temp_path / "app.py"
                app_file.write_text(app_file_content)
                
                # Write requirements.txt
                req_file = temp_path / "requirements.txt"
                req_file.write_text(requirements_content)
                
                # Upload files
                self.api.upload_folder(
                    folder_path=temp_dir,
                    repo_id=repo_id,
                    repo_type="space",
                    token=self.token,
                )
            
            space_url = f"https://huggingface.co/spaces/{repo_id}"
            logger.info(f"Model deployed to Hugging Face Space: {space_url}")
            return space_url
            
        except Exception as e:
            logger.error(f"Failed to deploy model to Hugging Face Space: {e}")
            raise

    def _generate_default_app(self, model_id: str) -> str:
        """Generate default Gradio app for model inference."""
        return f'''
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load model and tokenizer
model_id = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

def generate_text(prompt, max_length=150, temperature=0.7, top_p=0.9):
    """Generate text using the model."""
    try:
        # Generate response
        outputs = pipe(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = outputs[0]["generated_text"]
        # Remove the original prompt from the response
        response = generated_text[len(prompt):].strip()
        return response
        
    except Exception as e:
        return f"Error generating text: {{str(e)}}"

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            label="Input Prompt",
            placeholder="Enter your prompt here...",
            lines=3
        ),
        gr.Slider(
            minimum=50,
            maximum=500,
            value=150,
            label="Max Length"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=0.7,
            label="Temperature"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            label="Top P"
        ),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=5),
    title="LLM Text Generation",
    description="Generate text using the deployed language model.",
)

if __name__ == "__main__":
    iface.launch()
'''

    def _generate_requirements(self) -> str:
        """Generate requirements.txt for the Space."""
        return """
torch
transformers
gradio
accelerate
"""

    def get_space_status(self, space_name: str) -> Dict[str, Any]:
        """Get the status of a deployed space."""
        try:
            username = self.api.whoami(token=self.token)['name']
            repo_id = f"{username}/{space_name}"
            
            space_info = self.api.space_info(repo_id=repo_id, token=self.token)
            return {
                "status": "running" if space_info.runtime and space_info.runtime.stage == "RUNNING" else "stopped",
                "url": f"https://huggingface.co/spaces/{repo_id}",
                "hardware": space_info.runtime.hardware if space_info.runtime else "unknown",
                "sdk": space_info.sdk,
            }
            
        except Exception as e:
            logger.error(f"Failed to get space status: {e}")
            return {"status": "error", "error": str(e)}

    def delete_space(self, space_name: str) -> bool:
        """Delete a Hugging Face Space."""
        try:
            username = self.api.whoami(token=self.token)['name']
            repo_id = f"{username}/{space_name}"
            
            self.api.delete_repo(repo_id=repo_id, repo_type="space", token=self.token)
            logger.info(f"Deleted Hugging Face Space: {repo_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete space: {e}")
            return False


class HuggingFaceInference:
    """
    Inference client for Hugging Face Spaces as an alternative to SageMaker endpoints.
    """

    def __init__(self, space_url: str):
        self.space_url = space_url.rstrip('/')
        self.api_url = f"{self.space_url}/api/predict"

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction request to the Hugging Face Space.
        
        Args:
            inputs: Input data for prediction
            
        Returns:
            Prediction results
        """
        try:
            response = requests.post(
                self.api_url,
                json={"data": list(inputs.values())},
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return {"generated_text": result["data"][0]}
            
        except Exception as e:
            logger.error(f"Hugging Face Space inference failed: {e}")
            raise

    def health_check(self) -> bool:
        """Check if the space is healthy and responding."""
        try:
            response = requests.get(f"{self.space_url}/", timeout=10)
            return response.status_code == 200
        except:
            return False
