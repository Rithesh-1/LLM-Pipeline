"""
Local deployment CLI tool as a free alternative to AWS SageMaker deployment.
Provides commands to deploy, manage, and monitor local and HuggingFace Spaces deployments.
"""

import json
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from llm_pipeline_system.infrastructure.deployment import LocalDeploymentManager, deploy_model
from llm_pipeline_system.settings import settings


@click.group()
def cli():
    """Local deployment CLI - Free alternative to AWS SageMaker."""
    pass


@cli.command()
@click.option("--model-id", default=None, help="Hugging Face model ID")
@click.option("--port", default=8000, help="Port for local server")
@click.option("--background/--foreground", default=True, help="Run server in background")
@click.option("--save-config", is_flag=True, help="Save deployment config to file")
def deploy_local(
    model_id: Optional[str],
    port: int,
    background: bool,
    save_config: bool,
):
    """Deploy model locally using FastAPI server."""
    logger.info("Deploying model locally")
    
    if model_id is None:
        model_id = settings.HF_MODEL_ID
        logger.info(f"Using default model: {model_id}")
    
    deployment_info = deploy_model(
        deployment_type="local",
        model_id=model_id,
        port=port,
        background=background,
    )
    
    if deployment_info["status"] == "deployed":
        logger.info("✅ Local deployment successful!")
        logger.info(f"🌐 Endpoint URL: {deployment_info['endpoint_url']}")
        logger.info(f"❤️ Health Check: {deployment_info['health_check_url']}")
        
        if save_config:
            config_file = Path("local_deployment.json")
            with open(config_file, 'w') as f:
                json.dump(deployment_info, f, indent=2)
            logger.info(f"💾 Config saved to: {config_file}")
            
    else:
        logger.error("❌ Local deployment failed!")
        logger.error(f"Error: {deployment_info.get('error', 'Unknown error')}")
        raise click.ClickException("Deployment failed")


@cli.command()
@click.option("--space-name", default=None, help="Name for the HuggingFace Space")
@click.option("--model-id", default=None, help="Hugging Face model ID")
@click.option("--hardware", default="cpu-basic", help="Hardware type (cpu-basic is free)")
@click.option("--private/--public", default=False, help="Make space private")
@click.option("--save-config", is_flag=True, help="Save deployment config to file")
def deploy_hf_spaces(
    space_name: Optional[str],
    model_id: Optional[str],
    hardware: str,
    private: bool,
    save_config: bool,
):
    """Deploy model to Hugging Face Spaces."""
    logger.info("Deploying model to Hugging Face Spaces")
    
    if not settings.HUGGINGFACE_ACCESS_TOKEN:
        logger.error("❌ HUGGINGFACE_ACCESS_TOKEN is required for Spaces deployment")
        raise click.ClickException("Missing Hugging Face token")
    
    if model_id is None:
        model_id = settings.HF_MODEL_ID
        logger.info(f"Using default model: {model_id}")
    
    if space_name is None:
        space_name = settings.HF_SPACE_NAME
        logger.info(f"Using default space name: {space_name}")
    
    deployment_info = deploy_model(
        deployment_type="huggingface_spaces",
        model_id=model_id,
        space_name=space_name,
        hardware=hardware,
        private=private,
    )
    
    if deployment_info["status"] == "deployed":
        logger.info("✅ Hugging Face Spaces deployment successful!")
        logger.info(f"🚀 Space URL: {deployment_info['space_url']}")
        logger.info(f"⚙️ Hardware: {deployment_info['hardware']}")
        
        if save_config:
            config_file = Path("hf_spaces_deployment.json")
            with open(config_file, 'w') as f:
                json.dump(deployment_info, f, indent=2)
            logger.info(f"💾 Config saved to: {config_file}")
            
    else:
        logger.error("❌ Hugging Face Spaces deployment failed!")
        logger.error(f"Error: {deployment_info.get('error', 'Unknown error')}")
        raise click.ClickException("Deployment failed")


@cli.command()
@click.option("--config-file", type=click.Path(exists=True), help="Deployment config file")
@click.option("--deployment-type", type=click.Choice(["local", "huggingface_spaces"]), help="Deployment type")
def status(config_file: Optional[str], deployment_type: Optional[str]):
    """Check deployment status."""
    
    if config_file:
        with open(config_file, 'r') as f:
            deployment_info = json.load(f)
    else:
        # Try to find default config files
        local_config = Path("local_deployment.json")
        hf_config = Path("hf_spaces_deployment.json")
        
        if deployment_type == "local" and local_config.exists():
            with open(local_config, 'r') as f:
                deployment_info = json.load(f)
        elif deployment_type == "huggingface_spaces" and hf_config.exists():
            with open(hf_config, 'r') as f:
                deployment_info = json.load(f)
        else:
            logger.error("❌ No deployment config found")
            logger.info("Use --config-file or specify --deployment-type with existing config")
            raise click.ClickException("No deployment config found")
    
    manager = LocalDeploymentManager()
    status_info = manager.get_deployment_status(deployment_info)
    
    logger.info(f"📊 Deployment Status: {status_info['status']}")
    
    if status_info["status"] == "healthy":
        logger.info("✅ Deployment is healthy and responding")
    elif status_info["status"] == "running":
        logger.info("🟡 Deployment is running")
    elif status_info["status"] == "unhealthy":
        logger.warning("⚠️ Deployment is unhealthy")
    else:
        logger.error(f"❌ Deployment status: {status_info['status']}")
        if "error" in status_info:
            logger.error(f"Error: {status_info['error']}")


@cli.command()
@click.option("--config-file", type=click.Path(exists=True), help="Deployment config file")
@click.option("--deployment-type", type=click.Choice(["local", "huggingface_spaces"]), help="Deployment type")
@click.confirmation_option(prompt="Are you sure you want to delete the deployment?")
def delete(config_file: Optional[str], deployment_type: Optional[str]):
    """Delete a deployment."""
    
    if config_file:
        with open(config_file, 'r') as f:
            deployment_info = json.load(f)
    else:
        # Try to find default config files
        local_config = Path("local_deployment.json")
        hf_config = Path("hf_spaces_deployment.json")
        
        if deployment_type == "local" and local_config.exists():
            with open(local_config, 'r') as f:
                deployment_info = json.load(f)
        elif deployment_type == "huggingface_spaces" and hf_config.exists():
            with open(hf_config, 'r') as f:
                deployment_info = json.load(f)
        else:
            logger.error("❌ No deployment config found")
            raise click.ClickException("No deployment config found")
    
    manager = LocalDeploymentManager()
    success = manager.delete_deployment(deployment_info)
    
    if success:
        logger.info("✅ Deployment deleted successfully")
        
        # Clean up config file
        if config_file:
            Path(config_file).unlink(missing_ok=True)
            logger.info(f"🗑️ Config file deleted: {config_file}")
    else:
        logger.error("❌ Failed to delete deployment")
        raise click.ClickException("Deletion failed")


@cli.command()
@click.option("--endpoint-url", required=True, help="Endpoint URL to test")
@click.option("--prompt", default="Hello, how are you?", help="Test prompt")
def test_inference(endpoint_url: str, prompt: str):
    """Test inference endpoint."""
    import requests
    
    logger.info(f"Testing inference endpoint: {endpoint_url}")
    logger.info(f"Test prompt: {prompt}")
    
    try:
        # Test health endpoint first
        health_response = requests.get(f"{endpoint_url}/health", timeout=10)
        if health_response.status_code != 200:
            logger.warning("⚠️ Health check failed")
        else:
            logger.info("✅ Health check passed")
        
        # Test inference
        inference_data = {
            "prompt": prompt,
            "max_new_tokens": 50,
            "temperature": 0.7,
        }
        
        response = requests.post(
            f"{endpoint_url}/inference",
            json=inference_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Inference test successful!")
            logger.info(f"Generated text: {result['generated_text']}")
        else:
            logger.error(f"❌ Inference test failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            
    except Exception as e:
        logger.error(f"❌ Inference test failed: {e}")
        raise click.ClickException("Inference test failed")


@cli.command()
def list_deployments():
    """List all deployment config files."""
    logger.info("📋 Searching for deployment configs...")
    
    configs_found = []
    
    # Check for local deployment config
    local_config = Path("local_deployment.json")
    if local_config.exists():
        configs_found.append(("local", str(local_config)))
    
    # Check for HF Spaces deployment config
    hf_config = Path("hf_spaces_deployment.json")
    if hf_config.exists():
        configs_found.append(("huggingface_spaces", str(hf_config)))
    
    if configs_found:
        logger.info(f"Found {len(configs_found)} deployment configs:")
        for deployment_type, config_path in configs_found:
            logger.info(f"  📄 {deployment_type}: {config_path}")
    else:
        logger.info("No deployment configs found")


if __name__ == "__main__":
    cli()
