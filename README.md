<div align="center">
  <h1>👷 LLM Pipeline System</h1>
  <p class="tagline">Official repository of the LLM Pipeline System</p>
</div>


## 🌟 Features

This project implements a complete LLM Pipeline system using **100% free alternatives** to expensive cloud services:

### 🏗️ **Core Components**
- 📝 **Data Collection & Generation** - Web crawling and dataset creation
- 🔄 **Local Training Pipeline** - Fine-tuning with ClearML monitoring  
- 🖥️ **Local Inference Server** - FastAPI-based model serving (replaces AWS SageMaker)
- ☁️ **HuggingFace Spaces** - Free cloud deployment option
- 📊 **RAG System** - Retrieval-Augmented Generation with vector search
- 🔍 **Local Evaluation** - Comprehensive testing framework
- 🧪 **Monitoring & Tracking** - ClearML unified monitoring platform

### 💰 **Cost Benefits**
- **Traditional Setup**: $200-500/month (AWS SageMaker + cloud services)
- **This Setup**: **$0/month** (100% free alternatives)

### 📊 **System Architecture**
View our comprehensive system diagrams:
- [Overall System Architecture](docs/overall_system_architecture.mermaid) - Complete system overview
- [Training Pipeline](docs/training_pipeline.mermaid) - Local training workflow  
- [Deployment Pipeline](docs/deployment_pipeline.mermaid) - Free deployment options
- [Data Pipeline](docs/end_to_end_data_pipeline.mermaid) - End-to-end data flow
- [MLflow Integration](docs/mlflow_runner_pipeline.mermaid) - Experiment tracking




## 🔗 Dependencies

### Local dependencies

To install and run the project locally, you need the following dependencies.

| Tool | Version | Purpose | Installation Link |
|------|---------|---------|------------------|
| pyenv | ≥2.3.36 | Multiple Python versions (optional) | [Install Guide](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation) |
| Python | 3.11 | Runtime environment | [Download](https://www.python.org/downloads/) |
| Poetry | >= 1.8.3 and < 2.0 | Package management | [Install Guide](https://python-poetry.org/docs/#installation) |
| Docker | ≥27.1.1 | Containerization | [Install Guide](https://docs.docker.com/engine/install/) |
| Git | ≥2.44.0 | Version control | [Download](https://git-scm.com/downloads) |

### Free alternatives to cloud services

The code uses **free alternatives** to expensive cloud services, making it cost-effective to run:

| Service | Purpose | Cost |
|---------|---------|------|
| [HuggingFace](https://huggingface.com/) | Model registry & free deployment | Free |
| [MLflow](https://mlflow.org/) | Experiment tracking | Free (local) |
| [ClearML](https://clear.ml/) | Training monitoring | Free tier |
| [MongoDB](https://www.mongodb.com/) | NoSQL database | Free (local) |
| [Qdrant](https://qdrant.tech/) | Vector database | Free (local) |
| Local Inference | Model serving (replaces AWS SageMaker) | Free |
| HuggingFace Spaces | Cloud deployment (optional) | Free tier |
| [GitHub Actions](https://github.com/features/actions) | CI/CD pipeline | Free |

**💰 Cost Comparison:**
- **Traditional setup**: $200-500/month (AWS SageMaker + cloud services)
- **This setup**: $0/month (100% free alternatives)



## 🗂️ Project Structure

Here is the directory overview:

```bash
.
├── code_snippets/       # Standalone example code
├── configs/             # Pipeline configuration files
├── llm_pipeline_system/     # Core project package
│   ├── application/    
│   ├── domain/         
│   ├── infrastructure/ 
│   ├── model/         
├── pipelines/           # ML pipeline definitions
├── steps/               # Pipeline components
├── tests/               # Test examples
├── tools/               # Utility scripts
│   ├── run.py
│   ├── ml_service.py
│   ├── rag.py
│   ├── data_warehouse.py
```

`llm_pipeline_system/`  is the main Python package implementing LLM and RAG functionality. It follows Domain-Driven Design (DDD) principles:

- `domain/`: Core business entities and structures
- `application/`: Business logic, crawlers, and RAG implementation
- `model/`: LLM training and inference
- `infrastructure/`: External service integrations (Local Inference, HuggingFace Spaces, Qdrant, MongoDB, FastAPI)

The code logic and imports flow as follows: `infrastructure` → `model` → `application` → `domain`

`pipelines/`: Contains the ML pipelines, which serve as the entry point for all the ML pipelines. Coordinates the data processing and model training stages of the ML lifecycle.

`steps/`: Contains individual pipeline components, which are reusable for building and customizing pipelines. Steps perform specific tasks (e.g., data loading, preprocessing) and can be combined within the ML pipelines.

`tests/`: Covers a few sample tests used as examples within the CI pipeline.

`tools/`: Utility scripts used to call the pipelines and inference code:
- `run.py`: Entry point script to run pipelines.
- `deploy_local.py`: Local deployment CLI for model serving (replaces AWS SageMaker).
- `ml_service.py`: Starts the REST API inference server.
- `rag.py`: Demonstrates usage of the RAG retrieval module.
- `data_warehouse.py`: Used to export or import data from the MongoDB data warehouse through JSON files.

`configs/`: YAML configuration files to control the execution of pipelines and steps.

`code_snippets/`: Independent code examples that can be executed independently.

## 💻 Installation

> [!NOTE]
> If you are experiencing issues while installing and running the repository, consider checking the [Issues](https://github.com/PacktPublishing/LLM-Engineers-Handbook/issues) GitHub section for other people who solved similar problems or directly asking us for help.

### 1. Clone the Repository

Start by cloning the repository and navigating to the project directory:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository 
```

Next, we have to prepare your Python environment and its adjacent dependencies. 

### 2. Set Up Python Environment

The project requires Python 3.11. You can either use your global Python installation or set up a project-specific version using pyenv.

#### Option A: Using Global Python (if version 3.11 is installed)

Verify your Python version:

```bash
python --version  # Should show Python 3.11.x
```

#### Option B: Using pyenv (recommended)

1. Verify pyenv installation:

```bash
pyenv --version   # Should show pyenv 2.3.36 or later
```

2. Install Python 3.11.8:

```bash
pyenv install 3.11.8
```

3. Verify the installation:

```bash
python --version  # Should show Python 3.11.8
```

4. Confirm Python version in the project directory:

```bash
python --version
# Output: Python 3.11.8
```

> [!NOTE]  
> The project includes a `.python-version` file that automatically sets the correct Python version when you're in the project directory.

### 3. Install Dependencies

The project uses Poetry for dependency management.

1. Verify Poetry installation:

```bash
poetry --version  # Should show Poetry version 1.8.3 or later
```

2. Set up the project environment and install dependencies:

```bash
poetry env use 3.11
poetry install
poetry run pre-commit install
```

This will:

- Configure Poetry to use Python 3.11
- Install project dependencies (includes local inference and HuggingFace Spaces support)
- Set up pre-commit hooks for code verification

### 4. Activate the Environment

We use [Poe the Poet](https://poethepoet.natn.io/index.html) as a task runner to simplify command execution.

1. Activate the virtual environment:

```bash
poetry env activate
```

## 🚀 Quick Start

Get started with the free LLM Pipeline system in minutes:

### 1. **Data Pipeline** (Generate training data)
```bash
# Run end-to-end data pipeline
poetry poe run-end-to-end-data-pipeline

# Or run individual stages
poetry poe run-digital-data-etl
poetry poe run-feature-engineering-pipeline
poetry poe run-generate-instruct-datasets-pipeline
poetry poe run-generate-preference-datasets-pipeline   
```

### 2. **Training** (Fine-tune your model locally)
```bash
# Train model locally with ClearML monitoring
poetry poe run-training-pipeline
```

### 3. **Deployment** (Deploy for free)
```bash
# Option 1: Deploy locally (recommended)
poetry poe deploy-local

# Option 2: Deploy to HuggingFace Spaces (free cloud)
poetry poe deploy-hf-spaces

# Test your deployment
poetry poe test-local-endpoint
```

### 5. **Chat Interface** (Test your RAG system)
```bash
# Run the chat interface
poetry poe chat
```

2. Run project commands using Poe the Poet:

```bash
poetry poe ...
```

<details>
<summary>🔧 Troubleshooting Poe the Poet Installation</summary>

### Alternative Command Execution

If you're experiencing issues with `poethepoet`, you can still run the project commands directly through Poetry. Here's how:

1. Look up the command definition in `pyproject.toml`
2. Use `poetry run` with the underlying command

#### Example:
Use the direct command from pyproject.toml:
```bash
poetry run <actual-command-from-pyproject-toml>
```
Note: All project commands are defined in the [tool.poe.tasks] section of pyproject.toml
</details>

Now, let's configure our local project with all the necessary credentials and tokens to run the code locally.

### 5. Local Development Setup

After you have installed all the dependencies, you must create and fill a `.env` file with your credentials to appropriately interact with other services and run the project. Setting your sensitive credentials in a `.env` file is a good security practice, as this file won't be committed to GitHub or shared with anyone else. 

1. First, copy our example by running the following:

```bash
cp .env.example .env # The file must be at your repository's root!
```

2. Now, let's understand how to fill in all the essential variables within the `.env` file to get you started. The following are the mandatory settings we must complete when working locally:

#### OpenAI

To authenticate to OpenAI's API, you must fill out the `OPENAI_API_KEY` env var with an authentication token.

```env
OPENAI_API_KEY=your_api_key_here
```

→ Check out this [tutorial](https://platform.openai.com/docs/quickstart) to learn how to provide one from OpenAI.

#### Hugging Face

To authenticate to Hugging Face, you must fill out the `HUGGINGFACE_ACCESS_TOKEN` env var with an authentication token.

```env
HUGGINGFACE_ACCESS_TOKEN=your_token_here
```

→ Check out this [tutorial](https://huggingface.co/docs/hub/en/security-tokens) to learn how to provide one from Hugging Face.

#### ClearML

To configure ClearML, run the following command in your terminal and follow the on-screen instructions:

```bash
clearml-init
```

> [!NOTE]
> You can also specify the `OPENAI_MODEL_ID` and `GEMINI_MODEL_ID` in your `.env` file to switch between different models from OpenAI and Gemini.





## 🏗️ Infrastructure

### Local infrastructure (for testing and development)

When running the project locally, we host a MongoDB and Qdrant database using Docker.

Start the inference real-time RESTful API:
```bash
poetry poe run-inference-ml-service
```



#### Qdrant

REST API URL: `localhost:6333`

Dashboard URL: `localhost:6333/dashboard`

→ Find out more about using and setting up [Qdrant with Docker](https://qdrant.tech/documentation/quick-start/).

#### MongoDB

Database URI: `mongodb://llm_pipeline_system:llm_pipeline_system@127.0.0.1:27017`

Database name: `llm_pipeline_system`

Default credentials:
  - `username`: llm_pipeline_system
  - `password`: llm_pipeline_system

→ Find out more about using and setting up [MongoDB with Docker](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-community-with-docker).

You can search your MongoDB collections using your **IDEs MongoDB plugin** (which you have to install separately), where you have to use the database URI to connect to the MongoDB database hosted within the Docker container: `mongodb://llm_pipeline_system:llm_pipeline_system@127.0.0.1:27017`



### 💰 Running the Project Costs

We will mostly stick to free tiers for all the services except for OpenAI's API, which is a pay-as-you-go service.

## ⚡ Pipelines

All the ML pipelines will be orchestrated behind the scenes. A few exceptions exist when running utility scrips, such as exporting or importing from the data warehouse.

The pipelines are the entry point for most processes throughout this project. They are under the `pipelines/` folder. Thus, when you want to understand or debug a workflow, starting with the pipeline is the best approach.

Now, let's explore all the pipelines you can run. From data collection to training, we will present them in their natural order to go through the LLM project end-to-end.

### Data pipelines

Run the data collection ETL:
```bash
poetry poe run-digital-data-etl
```

To add additional links to collect from, go to `configs/digital_data_etl.yaml` and add them to the `links` field. Also, you can create a completely new file and specify it at run time, like this: `python -m llm_pipeline_system.interfaces.orchestrator.run --run-etl --etl-config-filename configs/digital_data_etl.yaml`

Run the feature engineering pipeline:
```bash
poetry poe run-feature-engineering-pipeline
```

Generate the instruct dataset:
```bash
poetry poe run-generate-instruct-datasets-pipeline
```

Generate the preference dataset:
```bash
poetry poe run-generate-preference-datasets-pipeline
```

Run all of the above compressed into a single pipeline:
```bash
poetry poe run-end-to-end-data-pipeline
```

### Utility pipelines

Export the data from the data warehouse to JSON files:
```bash
poetry poe run-export-data-warehouse-to-json
```

Import data to the data warehouse from JSON files (by default, it imports the data from the `data/data_warehouse_raw_data` directory):
```bash
poetry poe run-import-data-warehouse-from-json
```



### Training pipelines

Run the training pipeline:
```bash
poetry poe run-training-pipeline
```

Run the evaluation pipeline:
```bash
poetry poe run-evaluation-pipeline
```



### Inference pipelines

Call the RAG retrieval module with a test query:
```bash
poetry poe call-rag-retrieval-module
```

Start the inference real-time RESTful API:
```bash
poetry poe run-inference-ml-service
```

Call the inference real-time RESTful API with a test query:



```bash

poetry poe call-inference-ml-service

```

Run the chat interface to test the RAG system:
```bash
poetry poe chat
```



### Linting & formatting (QA)

Check or fix your linting issues:
```bash
poetry poe lint-check
poetry poe lint-fix
```

Check or fix your formatting issues:
```bash
poetry poe format-check
poetry poe format-fix
```

Check the code for leaked credentials:
```bash
poetry poe gitleaks-check
```

### Tests

Run all the tests using the following command:
```bash
poetry poe test
```

## 🏃 Run project

Based on the setup and usage steps described above, assuming the local and cloud infrastructure works and the `.env` is filled as expected, follow the next steps to run the LLM system end-to-end:

### Data

1. Collect data: `poetry poe run-digital-data-etl`

2. Compute features: `poetry poe run-feature-engineering-pipeline`

3. Compute instruct dataset: `poetry poe run-generate-instruct-datasets-pipeline`

4. Compute preference alignment dataset: `poetry poe run-generate-preference-datasets-pipeline`



5. SFT fine-tuning Llamma 3.1: `poetry poe run-training-pipeline`

6. For DPO, go to `configs/training.yaml`, change `finetuning_type` to `dpo`, and run `poetry poe run-training-pipeline` again

7. Evaluate fine-tuned models: `poetry poe run-evaluation-pipeline`

### Inference



8. Call only the RAG retrieval module: `poetry poe call-rag-retrieval-module`



11. Start end-to-end RAG server: `poetry poe run-inference-ml-service`

12. Test RAG server: `poetry poe call-inference-ml-service`

## 📄 License

This course is an open-source project released under the MIT license. Thus, as long you distribute our LICENSE and acknowledge our work, you can safely clone or fork this project and use it as a source of inspiration for whatever you want (e.g., university projects, college degree projects, personal projects, etc.).

## Remote Training on Google Colab

You can run the training pipeline on a free Google Colab GPU and still track your experiments on your local MLflow UI. This is achieved using `ngrok` to create a secure tunnel from Colab to your local machine.

### On Your Local Machine (Do This First)

1.  **Start the MLflow UI**
    Open a terminal and run the following command. Keep this terminal open.
    ```bash
    mlflow ui
    ```
    You can view your UI at `http://127.0.0.1:5000`.

2.  **Install and Configure ngrok**
    - Go to the [ngrok website](https://ngrok.com/download), sign up for a free account, and follow the instructions to download and add your authtoken.

3.  **Expose Your MLflow Server**
    Open a **new** terminal and run the following command to create a public URL for your MLflow server.
    ```bash
    ngrok http 5000
    ```
    Copy the `Forwarding` URL provided by ngrok (it will look similar to `https://<random-string>.ngrok-free.app`).

### In Your Google Colab Notebook

1.  **Set Runtime to GPU**
    - In the Colab menu, go to `Runtime` -> `Change runtime type`.
    - Select `T4 GPU` from the "Hardware accelerator" dropdown and click `Save`.

2.  **Clone Your Project**
    Add and run a code cell to clone your project's repository.
    ```python
    !git clone <your-github-repository-url>
    ```

3.  **Install Dependencies**
    Add and run the following cells to install the project's dependencies.
    ```python
    %cd <your-project-folder-name>
    !pip install poetry
    !poetry install --no-root
    ```

4.  **Set the MLflow Tracking URI**
    This is the most important step. It tells Colab where to send the experiment data. **Paste the public URL you copied from ngrok here.**
    ```python
    import os

    # Replace with your actual ngrok forwarding URL
    os.environ['MLFLOW_TRACKING_URI'] = 'https://<random-string>.ngrok-free.app'
    ```

#### Configure ClearML

Set up ClearML API keys as environment variables in your Colab notebook. You can get these from your ClearML account settings.

```python
import os

os.environ['CLEARML_API_ACCESS_KEY'] = 'YOUR_CLEARML_API_ACCESS_KEY'
os.environ['CLEARML_API_SECRET_KEY'] = 'YOUR_CLEARML_API_SECRET_KEY'
os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml' # Or your self-hosted server
```

> [!NOTE]
> The `project_name` and `task_name` for ClearML are defined in `steps/training/train.py`. The `project_name` is "LLM-Pipeline-System" and the `task_name` is "Training".

5.  **Run Training**
    Finally, run the training. All metrics, parameters, and model artifacts will be logged to your local MLflow UI in real-time.
    ```python
    !mlflow run . -e training --no-conda
    ```

## 🚀 Model Deployment

The project provides **free alternatives** to expensive cloud deployment services:

### Local Deployment (Recommended)

Deploy your trained model locally using our FastAPI-based inference server:

```bash
# Deploy model locally (replaces AWS SageMaker)
poetry poe deploy-local

# Check deployment status
poetry poe deployment-status --deployment-type local

# Test the endpoint
poetry poe test-local-endpoint
```

The local server provides:
- **FastAPI endpoints** for inference
- **Health checks** and monitoring
- **Memory optimization** (8-bit/4-bit quantization)
- **GPU/CPU support** with automatic detection

### HuggingFace Spaces (Free Cloud Alternative)

Deploy to HuggingFace Spaces for free cloud hosting:

```bash
# Deploy to HuggingFace Spaces (requires HUGGINGFACE_ACCESS_TOKEN)
poetry poe deploy-hf-spaces

# Check space status
poetry poe deployment-status --deployment-type huggingface_spaces
```

HuggingFace Spaces provides:
- **Free tier hosting** (cpu-basic hardware)
- **Gradio interface** for easy interaction
- **Public or private** spaces
- **Automatic scaling**

### Deployment Management

Use our comprehensive CLI for deployment management:

```bash
# List all deployments
poetry poe list-deployments

# Delete a deployment
poetry poe delete-deployment --deployment-type local

# Test any inference endpoint
python -m tools.deploy_local test-inference --endpoint-url http://your-endpoint
```

### Cost Comparison

| Deployment Option | Monthly Cost | Features |
|-------------------|--------------|----------|
| **AWS SageMaker** | $200-500+ | Cloud inference, auto-scaling |
| **Local Deployment** | $0 | Full control, privacy, offline |
| **HuggingFace Spaces** | $0 | Free cloud hosting, public access |


