# Migration from AWS to Free Alternatives

This guide helps you migrate from AWS SageMaker to free alternatives for model deployment and inference.

## 🎯 Overview

The LLM Engineers Handbook has been updated to use **free alternatives** to AWS services:

| AWS Service | Free Alternative | Benefits |
|-------------|------------------|----------|
| **SageMaker Endpoints** | Local Inference + HuggingFace Spaces | No costs, full control |
| **SageMaker Processing** | Local Evaluation | Run anywhere, no limits |
| **IAM Roles** | Local Authentication | Simplified setup |
| **S3 Storage** | Local Files + Git LFS | Version controlled |
| **ECR** | Docker Hub | Free public images |

## 🚀 Quick Start

### 1. Update Dependencies

```bash
# Install new dependencies
pip install transformers accelerate huggingface-hub gradio torch

# Or using poetry
poetry install
```

### 2. Update Environment Variables

Add to your `.env` file:

```bash
# Required for HuggingFace Spaces (optional)
HUGGINGFACE_ACCESS_TOKEN=your_token_here

# Local inference settings
USE_LOCAL_INFERENCE=true
LOCAL_INFERENCE_DEVICE=auto
LOCAL_INFERENCE_PORT=8000
LOAD_IN_8BIT=false
LOAD_IN_4BIT=false

# HuggingFace Spaces settings (optional)
USE_HF_SPACES=false
HF_SPACE_NAME=llm-twin-inference
HF_SPACE_HARDWARE=cpu-basic
HF_SPACE_SDK=gradio
```

### 3. Deploy Locally

```bash
# Deploy model locally (replaces SageMaker endpoints)
poetry poe deploy-local

# Check deployment status
poetry poe deployment-status --deployment-type local

# Test the endpoint
poetry poe test-local-endpoint
```

### 4. Run Evaluation

```bash
# Run evaluation locally (replaces SageMaker processing)
poetry poe run-evaluation-pipeline
```

## 📋 Migration Steps

### Step 1: Replace SageMaker Endpoints

**Before (AWS SageMaker):**
```bash
poetry poe create-sagemaker-role
poetry poe deploy-inference-endpoint
poetry poe test-sagemaker-endpoint
```

**After (Local Deployment):**
```bash
poetry poe deploy-local
poetry poe test-local-endpoint
```

### Step 2: Replace SageMaker Processing

**Before (AWS SageMaker):**
```python
from llm_engineering.model.evaluation.sagemaker import run_evaluation_on_sagemaker
run_evaluation_on_sagemaker(is_dummy=True)
```

**After (Local Evaluation):**
```python
from llm_engineering.model.evaluation.local_evaluation import run_evaluation_on_local
run_evaluation_on_local(is_dummy=True)
```

### Step 3: Update Inference Code

**Before (SageMaker Endpoint):**
```python
from llm_engineering.model.inference.inference import LLMInferenceSagemakerEndpoint

llm = LLMInferenceSagemakerEndpoint(
    endpoint_name="my-endpoint"
)
result = llm.inference()
```

**After (Local Inference):**
```python
from llm_engineering.model.inference.local_inference import LocalLLMInference

llm = LocalLLMInference(
    model_id="mlabonne/TwinLlama-3.1-8B-DPO"
)
result = llm.inference("Hello, world!")
```

## 🔧 Configuration Options

### Local Inference Settings

```python
# In settings.py or .env file
USE_LOCAL_INFERENCE=True          # Enable local inference
LOCAL_INFERENCE_DEVICE="auto"     # "auto", "cpu", "cuda"
LOCAL_INFERENCE_PORT=8000         # Port for local server
LOAD_IN_8BIT=True                 # Memory optimization
LOAD_IN_4BIT=False                # More aggressive optimization
```

### HuggingFace Spaces Settings

```python
# Optional cloud deployment (free tier)
USE_HF_SPACES=True                # Enable HF Spaces deployment
HF_SPACE_NAME="my-llm-space"      # Space name
HF_SPACE_HARDWARE="cpu-basic"     # Free tier hardware
HF_SPACE_SDK="gradio"             # UI framework
```

## 🛠️ Available Commands

### Deployment Commands

```bash
# Local deployment
poetry poe deploy-local                    # Deploy locally
poetry poe deploy-hf-spaces               # Deploy to HuggingFace Spaces
poetry poe deployment-status              # Check status
poetry poe list-deployments               # List all deployments
poetry poe delete-deployment              # Delete deployment

# Testing
poetry poe test-local-endpoint            # Test local endpoint
```

### Pipeline Commands

```bash
# Data pipelines (unchanged)
poetry poe run-digital-data-etl
poetry poe run-feature-engineering-pipeline
poetry poe run-generate-instruct-datasets-pipeline

# Training and evaluation (now uses local alternatives)
poetry poe run-training-pipeline         # Uses ClearML
poetry poe run-evaluation-pipeline       # Uses local evaluation
```

## 🎛️ Advanced Usage

### Custom Local Deployment

```python
from llm_engineering.infrastructure.deployment import deploy_model

# Deploy with custom settings
deployment_info = deploy_model(
    deployment_type="local",
    model_id="your-model-id",
    port=8080,
    background=True
)

print(f"Endpoint: {deployment_info['endpoint_url']}")
```

### HuggingFace Spaces Deployment

```python
from llm_engineering.infrastructure.huggingface import HuggingFaceSpacesDeployment

deployer = HuggingFaceSpacesDeployment()

# Create and deploy space
space_url = deployer.deploy_model_to_space(
    space_name="my-llm-app",
    model_id="mlabonne/TwinLlama-3.1-8B-DPO"
)

print(f"Deployed to: {space_url}")
```

### Memory Optimization

For limited hardware, use quantization:

```python
from llm_engineering.model.inference.local_inference import LocalLLMInference

# 8-bit quantization (saves ~50% memory)
llm = LocalLLMInference(
    model_id="mlabonne/TwinLlama-3.1-8B-DPO",
    load_in_8bit=True
)

# 4-bit quantization (saves ~75% memory)
llm = LocalLLMInference(
    model_id="mlabonne/TwinLlama-3.1-8B-DPO",
    load_in_4bit=True
)
```

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   ```bash
   # Enable memory optimization
   export LOAD_IN_8BIT=true
   # Or use CPU inference
   export LOCAL_INFERENCE_DEVICE=cpu
   ```

2. **Model Download Issues**
   ```bash
   # Set HuggingFace cache directory
   export HF_HOME=/path/to/cache
   # Use HuggingFace token for private models
   export HUGGINGFACE_ACCESS_TOKEN=your_token
   ```

3. **Port Already in Use**
   ```bash
   # Use different port
   poetry poe deploy-local --port 8001
   ```

### Performance Tips

1. **Use GPU if available**
   ```bash
   export LOCAL_INFERENCE_DEVICE=cuda
   ```

2. **Enable quantization for large models**
   ```bash
   export LOAD_IN_8BIT=true
   ```

3. **Use smaller models for faster inference**
   ```bash
   export HF_MODEL_ID=microsoft/DialoGPT-medium
   ```

## 📊 Cost Comparison

| Service | AWS Cost | Free Alternative | Savings |
|---------|----------|------------------|---------|
| SageMaker ml.g5.2xlarge | ~$1.20/hour | Local GPU | 100% |
| SageMaker ml.m5.large | ~$0.10/hour | Local CPU | 100% |
| SageMaker Processing | ~$0.05/hour | Local Processing | 100% |
| S3 Storage | ~$0.023/GB | Local Storage | 100% |
| **Total Monthly** | **~$200-500** | **$0** | **100%** |

## 🎉 Benefits of Migration

✅ **Zero Cloud Costs** - Run everything locally or on free tiers  
✅ **Full Control** - No vendor lock-in, customize everything  
✅ **Privacy** - Your data never leaves your infrastructure  
✅ **Faster Development** - No network latency, instant deployments  
✅ **Offline Capability** - Works without internet connection  
✅ **Educational** - Learn how ML systems work under the hood  

## 🆘 Need Help?

- Check the [local deployment guide](tools/deploy_local.py)
- Review [evaluation examples](llm_engineering/model/evaluation/local_evaluation.py)
- See [inference implementations](llm_engineering/model/inference/local_inference.py)

The migration maintains all functionality while eliminating AWS dependencies and costs! 🎊
