"""
Local inference implementation as a free alternative to AWS SageMaker.
Runs models locally using transformers and torch.
"""

from typing import Any, Dict, Optional

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from llm_pipeline_system.domain.inference import Inference
from llm_pipeline_system.settings import settings


class LocalLLMInference(Inference):
    """
    Local inference implementation using Hugging Face transformers.
    Free alternative to AWS SageMaker endpoints.
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        super().__init__()
        
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        
        logger.info(f"Initializing local inference for model: {model_id}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": True,
        }
        
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        
        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        logger.info("Local inference initialized successfully")

    def set_payload(self, inputs, parameters=None):
        """
        Set the payload for inference (required by abstract base class).
        
        Args:
            inputs: Input data for inference
            parameters: Optional parameters for inference
        """
        self.inputs = inputs
        self.parameters = parameters or {}

    def inference(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Perform local inference using the loaded model.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        try:
            # Default generation parameters
            generation_params = {
                "max_length": kwargs.get("max_new_tokens", 150) + len(self.tokenizer.encode(prompt)),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "do_sample": kwargs.get("do_sample", True),
                "pad_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False,  # Only return generated text
            }
            
            # Update with any provided parameters
            generation_params.update(kwargs)
            
            logger.info(f"Generating text for prompt: {prompt[:50]}...")
            
            # Generate text
            outputs = self.pipeline(prompt, **generation_params)
            
            if isinstance(outputs, list) and len(outputs) > 0:
                generated_text = outputs[0]["generated_text"]
            else:
                generated_text = ""
            
            result = {
                "generated_text": generated_text,
                "model_id": self.model_id,
                "device": self.device,
            }
            
            logger.info("Local inference completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Local inference failed: {e}")
            raise

    def batch_inference(self, prompts: list[str], **kwargs) -> list[Dict[str, Any]]:
        """
        Perform batch inference for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters
            
        Returns:
            List of generated responses
        """
        results = []
        for prompt in prompts:
            try:
                result = self.inference(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process prompt: {prompt[:50]}... Error: {e}")
                results.append({"error": str(e), "prompt": prompt})
        
        return results

    def health_check(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        try:
            test_prompt = "Hello"
            result = self.inference(test_prompt, max_new_tokens=5)
            return "generated_text" in result
        except:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "vocab_size": self.tokenizer.vocab_size,
            "model_max_length": self.tokenizer.model_max_length,
        }


class LocalInferenceServer:
    """
    Simple local inference server that can replace SageMaker endpoints.
    """

    def __init__(self, model_id: str, port: int = 8000):
        self.model_id = model_id
        self.port = port
        self.inference_client = None

    def start_server(self):
        """Start the local inference server."""
        try:
            import uvicorn
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            
            # Initialize the inference client
            self.inference_client = LocalLLMInference(self.model_id)
            
            app = FastAPI(title="Local LLM Inference Server")
            
            class InferenceRequest(BaseModel):
                prompt: str
                max_new_tokens: int = 150
                temperature: float = 0.7
                top_p: float = 0.9
                do_sample: bool = True
            
            class InferenceResponse(BaseModel):
                generated_text: str
                model_id: str
                device: str
            
            @app.post("/inference", response_model=InferenceResponse)
            async def inference_endpoint(request: InferenceRequest):
                try:
                    result = self.inference_client.inference(
                        prompt=request.prompt,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        do_sample=request.do_sample,
                    )
                    return InferenceResponse(**result)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            @app.get("/health")
            async def health_check():
                is_healthy = self.inference_client.health_check()
                if is_healthy:
                    return {"status": "healthy"}
                else:
                    raise HTTPException(status_code=503, detail="Model not ready")
            
            @app.get("/model-info")
            async def model_info():
                return self.inference_client.get_model_info()
            
            logger.info(f"Starting local inference server on port {self.port}")
            uvicorn.run(app, host="0.0.0.0", port=self.port)
            
        except ImportError:
            logger.error("FastAPI and uvicorn are required for the local server. Install with: pip install fastapi uvicorn")
            raise
        except Exception as e:
            logger.error(f"Failed to start local inference server: {e}")
            raise


# Factory function to create the appropriate inference client
def create_inference_client(
    endpoint_name: Optional[str] = None,
    model_id: Optional[str] = None,
    use_local: bool = True,
) -> Inference:
    """
    Factory function to create inference client.
    
    Args:
        endpoint_name: SageMaker endpoint name (deprecated)
        model_id: Hugging Face model ID for local inference
        use_local: Whether to use local inference (default: True)
        
    Returns:
        Inference client instance
    """
    if use_local:
        if not model_id:
            model_id = settings.HF_MODEL_ID
        
        logger.info(f"Creating local inference client for model: {model_id}")
        return LocalLLMInference(model_id)
    else:
        # Fallback to SageMaker if explicitly requested (not recommended)
        logger.warning("SageMaker inference is deprecated. Consider using local inference.")
        from .inference import LLMInferenceSagemakerEndpoint
        return LLMInferenceSagemakerEndpoint(endpoint_name)
