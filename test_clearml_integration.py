#!/usr/bin/env python3
"""
Test script to verify ClearML integration across the system.
This script tests the ClearML monitoring implementation that replaced CometML/Opik.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger

from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.infrastructure.clearml_utils import configure_clearml, get_or_create_clearml_task
from llm_engineering.model.inference.local_inference import LocalLLMInference
from llm_engineering.settings import settings


def test_clearml_configuration():
    """Test ClearML configuration and task creation."""
    logger.info("🧪 Testing ClearML configuration...")
    
    try:
        # Test configuration
        task = configure_clearml()
        if task:
            logger.info("✅ ClearML configuration successful")
            logger.info(f"   Task ID: {task.id}")
            logger.info(f"   Project: {task.get_project_name()}")
            logger.info(f"   Task Name: {task.name}")
            return True
        else:
            logger.warning("⚠️ ClearML configuration returned None (likely no credentials)")
            return False
            
    except Exception as e:
        logger.error(f"❌ ClearML configuration failed: {e}")
        return False


def test_rag_monitoring():
    """Test RAG system with ClearML monitoring."""
    logger.info("🧪 Testing RAG system with ClearML monitoring...")
    
    try:
        # Initialize task for RAG testing
        task = get_or_create_clearml_task("RAG-Test")
        
        if task:
            logger.info("✅ ClearML task created for RAG testing")
            
            # Test RAG retrieval with monitoring
            retriever = ContextRetriever(mock=True)  # Use mock to avoid dependencies
            query = "Test query for ClearML monitoring"
            
            # This should log to ClearML if properly integrated
            documents = retriever.search(query, k=3)
            
            logger.info(f"✅ RAG search completed with {len(documents)} documents")
            logger.info("✅ ClearML logging should be visible in the web UI")
            
            return True
        else:
            logger.warning("⚠️ Could not create ClearML task for RAG testing")
            return False
            
    except Exception as e:
        logger.error(f"❌ RAG monitoring test failed: {e}")
        return False


def test_local_inference():
    """Test local inference setup (without actual model loading)."""
    logger.info("🧪 Testing local inference configuration...")
    
    try:
        # Test LocalLLMInference class instantiation
        llm = LocalLLMInference(
            model_id="microsoft/DialoGPT-small",  # Small model for testing
            device="cpu",
            load_in_8bit=False  # Disable for testing
        )
        
        logger.info("✅ LocalLLMInference class instantiated successfully")
        logger.info(f"   Model ID: {llm.model_id}")
        logger.info(f"   Device: {llm.device}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Local inference test failed: {e}")
        return False


def test_settings_migration():
    """Test that CometML settings have been removed and ClearML settings added."""
    logger.info("🧪 Testing settings migration...")
    
    try:
        # Check that CometML settings are removed
        comet_removed = True
        try:
            _ = settings.COMET_API_KEY
            comet_removed = False
        except AttributeError:
            pass
        
        # Check that ClearML settings are added
        clearml_added = hasattr(settings, 'CLEARML_PROJECT')
        
        if comet_removed and clearml_added:
            logger.info("✅ Settings migration successful")
            logger.info("   - CometML settings removed")
            logger.info("   - ClearML settings added")
            return True
        else:
            logger.error("❌ Settings migration incomplete")
            if not comet_removed:
                logger.error("   - CometML settings still present")
            if not clearml_added:
                logger.error("   - ClearML settings missing")
            return False
            
    except Exception as e:
        logger.error(f"❌ Settings migration test failed: {e}")
        return False


def main():
    """Run all ClearML integration tests."""
    logger.info("🚀 Starting ClearML Integration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("ClearML Configuration", test_clearml_configuration),
        ("Settings Migration", test_settings_migration),
        ("Local Inference Setup", test_local_inference),
        ("RAG Monitoring", test_rag_monitoring),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        result = test_func()
        results.append((test_name, result))
        logger.info(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! ClearML integration is working correctly.")
        logger.info("\n📋 Next steps:")
        logger.info("   1. Run 'clearml-init' to configure ClearML credentials")
        logger.info("   2. Test the RAG system: python tools/rag.py")
        logger.info("   3. Start inference server: poetry poe run-inference-ml-service")
        return True
    else:
        logger.error("💥 Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
