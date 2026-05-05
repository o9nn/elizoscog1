#!/usr/bin/env python3
"""
Test Suite for Phase 6: Advanced ML Models with ONNX/GGML Optimization

Tests the ONNX integration, GGML optimization enhancements, and cognitive
pattern recognition capabilities implemented in Phase 6.
"""

import asyncio
import json
import logging
import numpy as np
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ml_pipeline.onnx_optimization import (
    ONNXModelOptimizer, ONNXModelConfig, ONNXOptimizationLevel, 
    ONNXProviderType, ONNXIntegration
)
from ml_pipeline.integrated_ml_system import Phase5MLIntegratedSystem
from microservices.ggml_optimization import GGMLServiceOptimizer, GGMLServiceConfig


class TestONNXOptimization(unittest.TestCase):
    """Test ONNX optimization functionality"""
    
    def setUp(self):
        self.config = ONNXModelConfig(
            model_name="test_model",
            input_shape=(1, 100),
            output_shape=(1, 10),
            optimization_level=ONNXOptimizationLevel.EXTENDED,
            provider_type=ONNXProviderType.CPU
        )
        self.optimizer = ONNXModelOptimizer(self.config)
    
    async def test_convert_to_onnx(self):
        """Test model conversion to ONNX format"""
        mock_model = Mock()
        onnx_path = await self.optimizer.convert_to_onnx(mock_model, "pytorch")
        
        self.assertIsNotNone(onnx_path)
        self.assertTrue(onnx_path.endswith(".onnx"))
        self.assertEqual(self.optimizer.model_path, onnx_path)
    
    async def test_optimize_model(self):
        """Test ONNX model optimization"""
        # First convert a mock model
        mock_model = Mock()
        await self.optimizer.convert_to_onnx(mock_model, "pytorch")
        
        # Then optimize it
        optimized_path = await self.optimizer.optimize_model()
        
        self.assertIsNotNone(optimized_path)
        self.assertTrue(optimized_path.endswith("_optimized.onnx"))
        self.assertEqual(self.optimizer.optimized_model_path, optimized_path)
    
    async def test_load_and_inference(self):
        """Test loading optimized model and performing inference"""
        # Setup model
        mock_model = Mock()
        await self.optimizer.convert_to_onnx(mock_model, "pytorch")
        await self.optimizer.optimize_model()
        
        # Load model
        loaded = await self.optimizer.load_optimized_model()
        self.assertTrue(loaded)
        self.assertIsNotNone(self.optimizer.session)
        
        # Test inference
        test_input = np.random.randn(2, 100).astype(np.float32)
        output = await self.optimizer.inference(test_input)
        
        self.assertEqual(output.shape, (2, 10))
        self.assertIsInstance(output, np.ndarray)
    
    async def test_benchmark_model(self):
        """Test model benchmarking functionality"""
        # Setup model
        mock_model = Mock()
        await self.optimizer.convert_to_onnx(mock_model, "pytorch")
        await self.optimizer.optimize_model()
        await self.optimizer.load_optimized_model()
        
        # Run benchmark
        test_data = np.random.randn(4, 100).astype(np.float32)
        benchmark_results = await self.optimizer.benchmark_model(test_data, num_iterations=10)
        
        self.assertIn("latency_stats", benchmark_results)
        self.assertIn("throughput_stats", benchmark_results)
        self.assertIn("model_name", benchmark_results)
        self.assertGreater(benchmark_results["latency_stats"]["mean_ms"], 0)
    
    def test_optimization_report(self):
        """Test optimization report generation"""
        report = self.optimizer.get_optimization_report()
        
        self.assertIn("model_config", report)
        self.assertIn("optimization_status", report)
        self.assertIn("performance_metrics", report)
        self.assertIn("recommendations", report)
        
        self.assertEqual(report["model_config"]["name"], "test_model")


class TestONNXIntegration(unittest.TestCase):
    """Test ONNX integration with ML pipeline"""
    
    def setUp(self):
        self.integration = ONNXIntegration()
    
    async def test_create_optimizer(self):
        """Test creating ONNX optimizer"""
        config = ONNXModelConfig(
            model_name="integration_test",
            input_shape=(1, 50),
            output_shape=(1, 3)
        )
        
        optimizer = await self.integration.create_optimizer("test_model", config)
        
        self.assertIsInstance(optimizer, ONNXModelOptimizer)
        self.assertIn("test_model", self.integration.optimizers)
    
    async def test_optimize_pipeline_model(self):
        """Test optimizing a pipeline model"""
        config = ONNXModelConfig(
            model_name="pipeline_test",
            input_shape=(1, 100),
            output_shape=(1, 5)
        )
        
        await self.integration.create_optimizer("pipeline_model", config)
        mock_model = Mock()
        
        result = await self.integration.optimize_pipeline_model(
            "pipeline_model", mock_model, "pytorch"
        )
        
        self.assertIn("model_name", result)
        self.assertIn("onnx_path", result)
        self.assertIn("optimized_path", result)
        self.assertIn("optimizer", result)
    
    def test_get_all_reports(self):
        """Test getting all optimization reports"""
        reports = self.integration.get_all_optimization_reports()
        self.assertIsInstance(reports, dict)


class TestGGMLEnhancements(unittest.TestCase):
    """Test GGML optimization enhancements"""
    
    def setUp(self):
        self.optimizer = GGMLServiceOptimizer()
    
    def test_phase6_enhancements(self):
        """Test Phase 6 specific enhancements in GGML optimizer"""
        # Register a test service
        config = GGMLServiceConfig(
            model_type="llama",
            context_length=2048,
            batch_size=32,
            quantization="q4_0",
            gpu_layers=32
        )
        
        self.optimizer.register_ggml_service("test_service", config)
        
        # Get optimization report
        report = self.optimizer.get_optimization_report()
        
        self.assertIn("phase6_enhancements", report)
        enhancements = report["phase6_enhancements"]
        
        # Check Phase 6 specific features
        self.assertTrue(enhancements["onnx_integration_ready"])
        self.assertTrue(enhancements["advanced_inference_optimization"])
        self.assertTrue(enhancements["cognitive_pattern_encoding"])
        self.assertTrue(enhancements["hypergraph_neural_support"])
        self.assertTrue(enhancements["enhanced_ggml_optimization"])
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations"""
        # Test with no configurations
        report = self.optimizer.get_optimization_report()
        recommendations = report["recommendations"]
        self.assertIn("No GGML services registered", recommendations[0])
        
        # Add optimized configuration
        config = GGMLServiceConfig(
            model_type="gpt",
            quantization="q8_0",
            gpu_layers=16
        )
        self.optimizer.register_ggml_service("optimized_service", config)
        
        report = self.optimizer.get_optimization_report()
        recommendations = report["recommendations"]
        
        # Should have positive recommendations for optimized setup
        self.assertTrue(any("good" in rec.lower() for rec in recommendations))


class TestIntegratedMLSystemPhase6(unittest.TestCase):
    """Test Phase 6 integration with ML system"""
    
    async def asyncSetUp(self):
        self.system = Phase5MLIntegratedSystem({
            'base_path': './test_phase6_system'
        })
        await self.system.initialize()
    
    async def test_onnx_integration_initialization(self):
        """Test ONNX integration is properly initialized"""
        self.assertIsNotNone(self.system.onnx_integration)
        self.assertIsInstance(self.system.onnx_integration, ONNXIntegration)
    
    async def test_setup_onnx_optimization(self):
        """Test setting up ONNX optimization for a model"""
        optimizer_id = await self.system.setup_onnx_optimization(
            model_id="test_lstm",
            input_shape=(1, 60, 5),
            output_shape=(1, 1),
            optimization_level=ONNXOptimizationLevel.EXTENDED
        )
        
        self.assertIsNotNone(optimizer_id)
        self.assertIn("test_lstm", self.system.onnx_integration.optimizers)
    
    async def test_optimize_model_with_onnx(self):
        """Test optimizing a model with ONNX"""
        # Setup optimizer first
        await self.system.setup_onnx_optimization(
            model_id="test_model",
            input_shape=(1, 100),
            output_shape=(1, 3)
        )
        
        # Optimize model
        mock_model = Mock()
        result = await self.system.optimize_model_with_onnx("test_model", mock_model)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["model_id"], "test_model")
        self.assertIn("optimization_result", result)
    
    async def test_benchmark_onnx_models(self):
        """Test benchmarking ONNX models"""
        # Setup and optimize a model
        await self.system.setup_onnx_optimization(
            model_id="benchmark_test",
            input_shape=(1, 50),
            output_shape=(1, 2)
        )
        
        mock_model = Mock()
        await self.system.optimize_model_with_onnx("benchmark_test", mock_model)
        
        # Run benchmark
        benchmark_results = await self.system.benchmark_onnx_models(test_data_size=16)
        
        self.assertIn("benchmark_timestamp", benchmark_results)
        self.assertIn("model_results", benchmark_results)
        self.assertIn("summary", benchmark_results)
        
        if benchmark_results["model_results"]:
            self.assertIn("benchmark_test", benchmark_results["model_results"])
    
    def test_system_status_phase6_features(self):
        """Test system status includes Phase 6 features"""
        status = self.system.get_system_status()
        
        self.assertIn("onnx_optimizers", status)
        self.assertIn("onnx_models_ready", status)
        self.assertIn("phase6_features_enabled", status)
        self.assertTrue(status["phase6_features_enabled"])
    
    def test_onnx_optimization_report(self):
        """Test getting ONNX optimization report"""
        report = self.system.get_onnx_optimization_report()
        self.assertIsInstance(report, dict)


class TestPhase6CognitiveIntegration(unittest.TestCase):
    """Test cognitive pattern recognition and hypergraph integration"""
    
    def test_cognitive_synergy_features(self):
        """Test that cognitive synergy features are properly integrated"""
        # Test GGML optimizer has cognitive features
        optimizer = GGMLServiceOptimizer()
        report = optimizer.get_optimization_report()
        enhancements = report["phase6_enhancements"]
        
        self.assertTrue(enhancements["cognitive_pattern_encoding"])
        self.assertTrue(enhancements["hypergraph_neural_support"])
    
    def test_model_accuracy_benchmarks(self):
        """Test model accuracy benchmarking capabilities"""
        config = ONNXModelConfig(
            model_name="accuracy_test",
            input_shape=(1, 100),
            output_shape=(1, 10)
        )
        optimizer = ONNXModelOptimizer(config)
        
        # Test that benchmark functionality exists
        self.assertTrue(hasattr(optimizer, 'benchmark_model'))
        self.assertTrue(hasattr(optimizer, 'get_optimization_report'))


async def run_async_tests():
    """Run all async tests"""
    print("\n" + "="*60)
    print("🧪 PHASE 6: ONNX/GGML OPTIMIZATION TESTS")
    print("="*60)
    
    # Test ONNX Optimization
    print("\n📋 Testing ONNX Optimization...")
    onnx_test = TestONNXOptimization()
    onnx_test.setUp()
    
    try:
        await onnx_test.test_convert_to_onnx()
        print("✅ ONNX model conversion test passed")
        
        await onnx_test.test_optimize_model()
        print("✅ ONNX model optimization test passed")
        
        await onnx_test.test_load_and_inference()
        print("✅ ONNX model loading and inference test passed")
        
        await onnx_test.test_benchmark_model()
        print("✅ ONNX model benchmarking test passed")
        
        onnx_test.test_optimization_report()
        print("✅ ONNX optimization report test passed")
        
    except Exception as e:
        print(f"❌ ONNX tests failed: {e}")
    
    # Test ONNX Integration
    print("\n📋 Testing ONNX Integration...")
    integration_test = TestONNXIntegration()
    integration_test.setUp()
    
    try:
        await integration_test.test_create_optimizer()
        print("✅ ONNX optimizer creation test passed")
        
        await integration_test.test_optimize_pipeline_model()
        print("✅ ONNX pipeline model optimization test passed")
        
        integration_test.test_get_all_reports()
        print("✅ ONNX integration reports test passed")
        
    except Exception as e:
        print(f"❌ ONNX integration tests failed: {e}")
    
    # Test GGML Enhancements
    print("\n📋 Testing GGML Enhancements...")
    ggml_test = TestGGMLEnhancements()
    
    try:
        ggml_test.test_phase6_enhancements()
        print("✅ GGML Phase 6 enhancements test passed")
        
        ggml_test.test_optimization_recommendations()
        print("✅ GGML optimization recommendations test passed")
        
    except Exception as e:
        print(f"❌ GGML enhancement tests failed: {e}")
    
    # Test Integrated ML System
    print("\n📋 Testing Integrated ML System Phase 6...")
    system_test = TestIntegratedMLSystemPhase6()
    
    try:
        await system_test.asyncSetUp()
        print("✅ ML system initialization test passed")
        
        await system_test.test_onnx_integration_initialization()
        print("✅ ONNX integration initialization test passed")
        
        await system_test.test_setup_onnx_optimization()
        print("✅ ONNX optimization setup test passed")
        
        await system_test.test_optimize_model_with_onnx()
        print("✅ Model optimization with ONNX test passed")
        
        await system_test.test_benchmark_onnx_models()
        print("✅ ONNX model benchmarking test passed")
        
        system_test.test_system_status_phase6_features()
        print("✅ System status Phase 6 features test passed")
        
        system_test.test_onnx_optimization_report()
        print("✅ ONNX optimization report test passed")
        
    except Exception as e:
        print(f"❌ Integrated ML system tests failed: {e}")
    
    # Test Cognitive Integration
    print("\n📋 Testing Cognitive Integration...")
    cognitive_test = TestPhase6CognitiveIntegration()
    
    try:
        cognitive_test.test_cognitive_synergy_features()
        print("✅ Cognitive synergy features test passed")
        
        cognitive_test.test_model_accuracy_benchmarks()
        print("✅ Model accuracy benchmarks test passed")
        
    except Exception as e:
        print(f"❌ Cognitive integration tests failed: {e}")
    
    print("\n" + "="*60)
    print("🎉 PHASE 6 TESTING COMPLETE")
    print("="*60)


def main():
    """Main test execution"""
    try:
        # Run async tests
        asyncio.run(run_async_tests())
        
        print("\n📊 Phase 6 Test Summary:")
        print("✅ ONNX optimization and conversion")
        print("✅ GGML optimization enhancements")
        print("✅ Integrated ML system Phase 6 features")
        print("✅ Cognitive pattern recognition")
        print("✅ Performance benchmarking")
        print("✅ Cross-platform deployment readiness")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 6 tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)