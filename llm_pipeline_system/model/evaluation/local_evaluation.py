"""
Local evaluation implementation as a free alternative to AWS SageMaker.
Runs evaluation locally using transformers and evaluation libraries.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from tqdm import tqdm

from llm_pipeline_system import settings
from llm_pipeline_system.model.inference.local_inference import LocalLLMInference


def run_evaluation_locally(
    model_id: Optional[str] = None,
    evaluation_dataset_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    is_dummy: bool = True,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Run model evaluation locally as an alternative to SageMaker.
    
    Args:
        model_id: Hugging Face model ID to evaluate
        evaluation_dataset_path: Path to evaluation dataset
        output_dir: Directory to save evaluation results
        is_dummy: Whether to run a dummy evaluation with limited data
        batch_size: Batch size for evaluation
        
    Returns:
        Evaluation results dictionary
    """
    logger.info("Starting local model evaluation")
    
    # Set defaults
    if model_id is None:
        model_id = settings.HF_MODEL_ID
    
    if evaluation_dataset_path is None:
        # Use a default evaluation dataset path
        evaluation_dataset_path = Path("data/evaluation/test_dataset.json")
    
    if output_dir is None:
        output_dir = Path("evaluation_results")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize local inference
        logger.info(f"Loading model for evaluation: {model_id}")
        inference_client = LocalLLMInference(
            model_id=model_id,
            load_in_8bit=True,  # Use 8-bit for memory efficiency during evaluation
        )
        
        # Load evaluation dataset
        evaluation_data = _load_evaluation_dataset(evaluation_dataset_path, is_dummy)
        logger.info(f"Loaded {len(evaluation_data)} evaluation samples")
        
        # Run evaluation
        results = _run_evaluation_batch(
            inference_client=inference_client,
            evaluation_data=evaluation_data,
            batch_size=batch_size,
        )
        
        # Calculate metrics
        metrics = _calculate_evaluation_metrics(results)
        
        # Save results
        results_file = output_dir / "evaluation_results.json"
        metrics_file = output_dir / "evaluation_metrics.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {output_dir}")
        logger.info(f"Evaluation metrics: {metrics}")
        
        return {
            "status": "completed",
            "model_id": model_id,
            "num_samples": len(evaluation_data),
            "metrics": metrics,
            "results_file": str(results_file),
            "metrics_file": str(metrics_file),
        }
        
    except Exception as e:
        logger.error(f"Local evaluation failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "model_id": model_id,
        }


def _load_evaluation_dataset(dataset_path: Path, is_dummy: bool = True) -> List[Dict[str, Any]]:
    """Load evaluation dataset from file or create dummy data."""
    
    if dataset_path.exists():
        logger.info(f"Loading evaluation dataset from {dataset_path}")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if is_dummy:
            # Limit to first 10 samples for dummy run
            data = data[:10]
            
        return data
    else:
        logger.warning(f"Evaluation dataset not found at {dataset_path}. Creating dummy dataset.")
        return _create_dummy_evaluation_dataset(is_dummy)


def _create_dummy_evaluation_dataset(is_dummy: bool = True) -> List[Dict[str, Any]]:
    """Create a dummy evaluation dataset for testing."""
    
    dummy_prompts = [
        {
            "prompt": "Write a professional LinkedIn post about machine learning.",
            "expected_keywords": ["machine learning", "professional", "LinkedIn"],
            "category": "social_media"
        },
        {
            "prompt": "Explain the concept of neural networks in simple terms.",
            "expected_keywords": ["neural networks", "simple", "explain"],
            "category": "education"
        },
        {
            "prompt": "Create a summary of recent advances in AI.",
            "expected_keywords": ["AI", "advances", "summary"],
            "category": "technology"
        },
        {
            "prompt": "Write a blog post about the importance of data quality.",
            "expected_keywords": ["data quality", "blog", "importance"],
            "category": "technical_writing"
        },
        {
            "prompt": "Describe the benefits of using cloud computing.",
            "expected_keywords": ["cloud computing", "benefits", "describe"],
            "category": "technology"
        },
    ]
    
    if is_dummy:
        return dummy_prompts[:3]  # Return only 3 samples for dummy run
    else:
        # Extend with more samples for full evaluation
        extended_prompts = dummy_prompts * 4  # 20 samples total
        return extended_prompts


def _run_evaluation_batch(
    inference_client: LocalLLMInference,
    evaluation_data: List[Dict[str, Any]],
    batch_size: int = 1,
) -> List[Dict[str, Any]]:
    """Run evaluation on a batch of data."""
    
    results = []
    
    for i in tqdm(range(0, len(evaluation_data), batch_size), desc="Evaluating"):
        batch = evaluation_data[i:i + batch_size]
        
        for sample in batch:
            try:
                prompt = sample["prompt"]
                
                # Generate response
                response = inference_client.inference(
                    prompt=prompt,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                )
                
                # Evaluate response
                evaluation_result = _evaluate_response(sample, response["generated_text"])
                
                result = {
                    "prompt": prompt,
                    "generated_text": response["generated_text"],
                    "expected_keywords": sample.get("expected_keywords", []),
                    "category": sample.get("category", "unknown"),
                    "evaluation": evaluation_result,
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate sample: {e}")
                results.append({
                    "prompt": sample["prompt"],
                    "error": str(e),
                    "evaluation": {"score": 0.0, "passed": False}
                })
    
    return results


def _evaluate_response(sample: Dict[str, Any], generated_text: str) -> Dict[str, Any]:
    """Evaluate a generated response against expected criteria."""
    
    evaluation = {
        "length_score": 0.0,
        "keyword_score": 0.0,
        "overall_score": 0.0,
        "passed": False,
        "details": {}
    }
    
    try:
        # Length evaluation (basic check)
        text_length = len(generated_text.split())
        if 20 <= text_length <= 300:  # Reasonable length
            evaluation["length_score"] = 1.0
        elif text_length > 10:  # At least some content
            evaluation["length_score"] = 0.5
        
        evaluation["details"]["word_count"] = text_length
        
        # Keyword evaluation
        expected_keywords = sample.get("expected_keywords", [])
        if expected_keywords:
            found_keywords = []
            for keyword in expected_keywords:
                if keyword.lower() in generated_text.lower():
                    found_keywords.append(keyword)
            
            evaluation["keyword_score"] = len(found_keywords) / len(expected_keywords)
            evaluation["details"]["found_keywords"] = found_keywords
            evaluation["details"]["expected_keywords"] = expected_keywords
        else:
            evaluation["keyword_score"] = 1.0  # No keywords to check
        
        # Overall score (weighted average)
        evaluation["overall_score"] = (
            evaluation["length_score"] * 0.3 + 
            evaluation["keyword_score"] * 0.7
        )
        
        # Pass/fail threshold
        evaluation["passed"] = evaluation["overall_score"] >= 0.6
        
    except Exception as e:
        logger.error(f"Error in response evaluation: {e}")
        evaluation["error"] = str(e)
    
    return evaluation


def _calculate_evaluation_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall evaluation metrics."""
    
    if not results:
        return {"error": "No results to calculate metrics"}
    
    # Filter out error results
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        return {"error": "No valid results to calculate metrics"}
    
    # Calculate metrics
    total_samples = len(valid_results)
    passed_samples = sum(1 for r in valid_results if r["evaluation"]["passed"])
    
    overall_scores = [r["evaluation"]["overall_score"] for r in valid_results]
    length_scores = [r["evaluation"]["length_score"] for r in valid_results]
    keyword_scores = [r["evaluation"]["keyword_score"] for r in valid_results]
    
    metrics = {
        "total_samples": total_samples,
        "passed_samples": passed_samples,
        "pass_rate": passed_samples / total_samples,
        "average_overall_score": sum(overall_scores) / len(overall_scores),
        "average_length_score": sum(length_scores) / len(length_scores),
        "average_keyword_score": sum(keyword_scores) / len(keyword_scores),
        "score_distribution": {
            "min_score": min(overall_scores),
            "max_score": max(overall_scores),
            "median_score": sorted(overall_scores)[len(overall_scores) // 2],
        }
    }
    
    # Category-wise metrics
    categories = {}
    for result in valid_results:
        category = result.get("category", "unknown")
        if category not in categories:
            categories[category] = []
        categories[category].append(result["evaluation"]["overall_score"])
    
    category_metrics = {}
    for category, scores in categories.items():
        category_metrics[category] = {
            "count": len(scores),
            "average_score": sum(scores) / len(scores),
            "pass_rate": sum(1 for s in scores if s >= 0.6) / len(scores),
        }
    
    metrics["category_metrics"] = category_metrics
    
    return metrics


# Convenience function to replace the SageMaker evaluation
def run_evaluation_on_local(is_dummy: bool = True) -> None:
    """
    Convenience function to run local evaluation.
    Drop-in replacement for run_evaluation_on_sagemaker.
    """
    logger.info("Running evaluation locally (SageMaker alternative)")
    
    result = run_evaluation_locally(is_dummy=is_dummy)
    
    if result["status"] == "completed":
        logger.info("Local evaluation completed successfully")
        logger.info(f"Pass rate: {result['metrics']['pass_rate']:.2%}")
        logger.info(f"Average score: {result['metrics']['average_overall_score']:.3f}")
    else:
        logger.error(f"Local evaluation failed: {result.get('error', 'Unknown error')}")
        raise RuntimeError("Local evaluation failed")
