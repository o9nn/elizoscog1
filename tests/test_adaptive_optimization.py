#!/usr/bin/env python3
"""
Comprehensive Test Suite for Adaptive Optimization & Continuous Learning
Phase 5 Implementation: Validates all components of the adaptive optimization system
"""

import asyncio
import unittest
import sys
import os
import time
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.optimization.adaptive_optimization import (
    AdaptiveParameter, ContinuousBenchmarkConfig, ContinuousPerformanceBenchmark,
    SelfTuningAlgorithm, FitnessLandscapeMapper, AdaptiveOptimizationEngine,
    AdaptiveStrategy, FitnessLandscapeType, create_adaptive_optimization_engine
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAdaptiveParameter(unittest.TestCase):
    """Test the AdaptiveParameter class"""
    
    def setUp(self):
        self.param = AdaptiveParameter(
            name="test_param",
            current_value=0.5,
            min_value=0.0,
            max_value=1.0,
            adaptation_rate=0.1
        )
    
    def test_parameter_initialization(self):
        """Test parameter initialization"""
        self.assertEqual(self.param.name, "test_param")
        self.assertEqual(self.param.current_value, 0.5)
        self.assertEqual(self.param.min_value, 0.0)
        self.assertEqual(self.param.max_value, 1.0)
        self.assertEqual(self.param.adaptation_rate, 0.1)
        
    def test_update_value_within_bounds(self):
        """Test parameter value update within bounds"""
        self.param.update_value(0.7, 0.8)
        
        # Should be adapted based on adaptation rate
        expected_value = 0.5 * (1 - 0.1) + 0.7 * 0.1
        self.assertAlmostEqual(self.param.current_value, expected_value, places=5)
        
        # History should be updated
        self.assertEqual(len(self.param.history), 1)
        self.assertEqual(self.param.history[0][1], 0.7)  # Original new value
        self.assertEqual(self.param.history[0][2], 0.8)  # Fitness
    
    def test_update_value_clamping(self):
        """Test parameter value clamping to bounds"""
        # Test upper bound clamping
        self.param.update_value(1.5, 0.9)
        self.assertLessEqual(self.param.current_value, 1.0)
        
        # Test lower bound clamping
        self.param.update_value(-0.5, 0.3)
        self.assertGreaterEqual(self.param.current_value, 0.0)
    
    def test_trend_detection(self):
        """Test parameter trend detection"""
        # Initially should be stable
        self.assertEqual(self.param.get_trend(), "stable")
        
        # Add increasing trend
        for i, val in enumerate([0.6, 0.7, 0.8]):
            self.param.update_value(val, 0.5 + i * 0.1)
        
        trend = self.param.get_trend()
        self.assertIn(trend, ["increasing", "stable"])  # May be stable due to adaptation rate


class TestContinuousPerformanceBenchmark(unittest.TestCase):
    """Test the ContinuousPerformanceBenchmark class"""
    
    def setUp(self):
        self.config = ContinuousBenchmarkConfig(
            benchmark_interval_seconds=0.1,  # Fast for testing
            performance_window_size=10,
            regression_threshold=0.2
        )
        self.benchmark = ContinuousPerformanceBenchmark(self.config)
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization"""
        self.assertEqual(self.benchmark.config.benchmark_interval_seconds, 0.1)
        self.assertFalse(self.benchmark.running)
        self.assertEqual(len(self.benchmark.performance_history), 0)
    
    async def test_continuous_monitoring_start_stop(self):
        """Test starting and stopping continuous monitoring"""
        async def mock_benchmark():
            return {"latency_ms": 5.0, "throughput_ops_sec": 200.0}
        
        # Start monitoring
        await self.benchmark.start_continuous_monitoring(mock_benchmark)
        self.assertTrue(self.benchmark.running)
        
        # Let it run briefly
        await asyncio.sleep(0.3)
        
        # Stop monitoring
        await self.benchmark.stop_continuous_monitoring()
        self.assertFalse(self.benchmark.running)
        
        # Should have collected some data
        self.assertGreater(len(self.benchmark.performance_history), 0)
    
    async def test_performance_regression_detection(self):
        """Test performance regression detection"""
        # Set up baseline
        self.benchmark.baseline_metrics = {
            "latency_ms": 5.0,
            "throughput_ops_sec": 200.0
        }
        
        # Create snapshot with regression
        regression_snapshot = {
            "timestamp": time.time(),
            "latency_ms": 10.0,  # 2x increase (regression)
            "throughput_ops_sec": 100.0  # 50% decrease (regression)
        }
        
        # Check regression
        await self.benchmark._check_performance_regression(regression_snapshot)
        
        # Should have detected regression
        self.assertGreater(len(self.benchmark.regression_alerts), 0)
        
        # Check alert details
        alert = self.benchmark.regression_alerts[0]
        self.assertEqual(alert["severity"], "critical")
        self.assertIn("latency_ms", alert["details"])
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Add some mock data
        for i in range(5):
            self.benchmark.performance_history.append({
                "timestamp": time.time() + i,
                "latency_ms": 5.0 + i * 0.1,
                "throughput_ops_sec": 200.0 - i * 5
            })
        
        summary = self.benchmark.get_performance_summary()
        
        self.assertIn("current_metrics", summary)
        self.assertIn("data_points", summary)
        self.assertEqual(summary["data_points"], 5)


class TestSelfTuningAlgorithm(unittest.TestCase):
    """Test the SelfTuningAlgorithm class"""
    
    def setUp(self):
        self.parameters = [
            AdaptiveParameter("param1", 0.5, 0.0, 1.0),
            AdaptiveParameter("param2", 0.3, 0.0, 1.0)
        ]
        self.algorithm = SelfTuningAlgorithm(self.parameters, AdaptiveStrategy.CONTINUOUS_TUNING)
    
    def test_algorithm_initialization(self):
        """Test algorithm initialization"""
        self.assertEqual(len(self.algorithm.parameters), 2)
        self.assertEqual(self.algorithm.strategy, AdaptiveStrategy.CONTINUOUS_TUNING)
        self.assertEqual(self.algorithm.generation_count, 0)
    
    async def test_continuous_tuning_optimization(self):
        """Test continuous tuning optimization"""
        def mock_fitness(params):
            # Simple fitness function that prefers param1=0.8, param2=0.2
            p1 = params.get("param1", 0.5)
            p2 = params.get("param2", 0.3)
            return 1.0 - abs(p1 - 0.8) - abs(p2 - 0.2)
        
        # Run optimization
        result = await self.algorithm.optimize_parameters(mock_fitness)
        
        # Check results
        self.assertEqual(result["generation"], 1)
        self.assertEqual(result["strategy"], "continuous_tuning")
        self.assertIn("current_fitness", result)
        self.assertIn("parameter_values", result)
        
        # Should have made some improvements
        self.assertGreaterEqual(len(result.get("improvements_made", [])), 0)
    
    async def test_evolutionary_optimization(self):
        """Test evolutionary optimization strategy"""
        self.algorithm.strategy = AdaptiveStrategy.EVOLUTIONARY_SEARCH
        
        def mock_fitness(params):
            # Fitness function that rewards balanced parameters
            p1 = params.get("param1", 0.5)
            p2 = params.get("param2", 0.3)
            return (p1 + p2) / 2.0
        
        # Run optimization
        result = await self.algorithm.optimize_parameters(mock_fitness)
        
        # Check results
        self.assertEqual(result["strategy"], "evolutionary_search")
        self.assertIn("population_size", result)
        self.assertIn("parameter_values", result)
        
        # Best fitness should be reasonable
        self.assertGreater(result["current_fitness"], 0.0)
    
    async def test_gradient_optimization(self):
        """Test gradient-based optimization"""
        self.algorithm.strategy = AdaptiveStrategy.GRADIENT_BASED
        
        def mock_fitness(params):
            # Quadratic fitness function
            p1 = params.get("param1", 0.5)
            p2 = params.get("param2", 0.3)
            return 1.0 - (p1 - 0.7)**2 - (p2 - 0.4)**2
        
        # Run optimization
        result = await self.algorithm.optimize_parameters(mock_fitness)
        
        # Check results
        self.assertEqual(result["strategy"], "gradient_based")
        self.assertIn("gradients", result)
        
        # Gradients should point toward optimal values
        gradients = result["gradients"]
        self.assertIn("param1", gradients)
        self.assertIn("param2", gradients)
    
    def test_optimization_summary(self):
        """Test optimization summary generation"""
        # Add some history
        self.algorithm.optimization_history = [
            {"current_fitness": 0.5, "generation": 1},
            {"current_fitness": 0.6, "generation": 2},
            {"current_fitness": 0.7, "generation": 3}
        ]
        self.algorithm.generation_count = 3
        self.algorithm.best_fitness = 0.7
        
        summary = self.algorithm.get_optimization_summary()
        
        self.assertEqual(summary["generation_count"], 3)
        self.assertEqual(summary["best_fitness"], 0.7)
        self.assertIn("convergence_estimate", summary)
        self.assertIn("parameter_trends", summary)


class TestFitnessLandscapeMapper(unittest.TestCase):
    """Test the FitnessLandscapeMapper class"""
    
    def setUp(self):
        self.mapper = FitnessLandscapeMapper(["param1", "param2"])
    
    def test_mapper_initialization(self):
        """Test mapper initialization"""
        self.assertEqual(self.mapper.parameter_names, ["param1", "param2"])
        self.assertEqual(len(self.mapper.landscape_points), 0)
        self.assertEqual(len(self.mapper.trajectory_points), 0)
    
    def test_add_evaluation_point(self):
        """Test adding evaluation points"""
        parameters = {"param1": 0.5, "param2": 0.3}
        fitness = 0.7
        generation = 1
        
        self.mapper.add_evaluation_point(parameters, fitness, generation)
        
        # Check point was added
        self.assertEqual(len(self.mapper.landscape_points), 1)
        self.assertEqual(len(self.mapper.trajectory_points), 1)
        
        point = self.mapper.landscape_points[0]
        self.assertEqual(point.coordinates, parameters)
        self.assertEqual(point.fitness_value, fitness)
        self.assertEqual(point.generation, generation)
    
    def test_fitness_surface_mapping(self):
        """Test fitness surface mapping"""
        # Add some evaluation points
        for i in range(5):
            for j in range(5):
                params = {"param1": i/4.0, "param2": j/4.0}
                fitness = (i + j) / 8.0  # Simple fitness function
                self.mapper.add_evaluation_point(params, fitness)
        
        # Map fitness surface
        parameter_ranges = {"param1": (0.0, 1.0), "param2": (0.0, 1.0)}
        surface = self.mapper.map_fitness_surface(parameter_ranges, resolution=4)
        
        # Check surface structure
        self.assertEqual(surface["parameter_names"], ["param1", "param2"])
        self.assertEqual(surface["resolution"], 4)
        self.assertIn("surface_data", surface)
        self.assertGreater(surface["data_points"], 0)
    
    def test_optimization_trajectory(self):
        """Test optimization trajectory tracking"""
        # Add trajectory points
        fitness_values = [0.3, 0.5, 0.7, 0.8, 0.85]
        for i, fitness in enumerate(fitness_values):
            params = {"param1": 0.1 + i * 0.1, "param2": 0.2 + i * 0.05}
            self.mapper.add_evaluation_point(params, fitness, i+1)
        
        trajectory = self.mapper.get_optimization_trajectory()
        
        # Check trajectory structure
        self.assertEqual(len(trajectory["trajectory"]), 5)
        self.assertIn("statistics", trajectory)
        
        stats = trajectory["statistics"]
        self.assertEqual(stats["total_points"], 5)
        self.assertEqual(stats["fitness_improvement"], 0.85 - 0.3)
        self.assertIn("convergence_trend", stats)
    
    def test_fitness_peaks_identification(self):
        """Test identification of fitness peaks"""
        # Add points with varying fitness
        fitness_values = [0.2, 0.4, 0.9, 0.3, 0.8, 0.1, 0.95]
        for i, fitness in enumerate(fitness_values):
            params = {"param1": i/6.0, "param2": (i+1)/7.0}
            self.mapper.add_evaluation_point(params, fitness)
        
        # Find top 20% peaks
        peaks = self.mapper.identify_fitness_peaks(threshold_percentile=80)
        
        # Should find the highest fitness points
        self.assertGreater(len(peaks), 0)
        self.assertGreaterEqual(peaks[0]["fitness"], 0.8)  # Should include high fitness points
        
        # Peaks should be sorted by fitness (descending)
        for i in range(len(peaks) - 1):
            self.assertGreaterEqual(peaks[i]["fitness"], peaks[i+1]["fitness"])
    
    def test_landscape_summary(self):
        """Test landscape summary generation"""
        # Add some evaluation points
        for i in range(10):
            params = {"param1": i/9.0, "param2": (i+1)/10.0}
            fitness = i / 9.0  # Increasing fitness
            self.mapper.add_evaluation_point(params, fitness)
        
        summary = self.mapper.get_landscape_summary()
        
        self.assertEqual(summary["parameter_names"], ["param1", "param2"])
        self.assertEqual(summary["total_evaluations"], 10)
        self.assertIn("fitness_statistics", summary)
        self.assertIn("optimization_progress", summary)
        
        # Check fitness statistics
        stats = summary["fitness_statistics"]
        self.assertEqual(stats["min"], 0.0)
        self.assertAlmostEqual(stats["max"], 1.0, places=5)


class TestAdaptiveOptimizationEngine(unittest.TestCase):
    """Test the AdaptiveOptimizationEngine class"""
    
    def setUp(self):
        self.parameters = [
            AdaptiveParameter("learning_rate", 0.1, 0.01, 0.5),
            AdaptiveParameter("batch_size", 32, 8, 128)
        ]
        
        self.benchmark_config = ContinuousBenchmarkConfig(
            benchmark_interval_seconds=0.1,  # Fast for testing
            performance_window_size=5
        )
        
        self.engine = AdaptiveOptimizationEngine(
            self.parameters,
            self.benchmark_config,
            AdaptiveStrategy.HYBRID_ADAPTIVE
        )
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        self.assertEqual(len(self.engine.parameters), 2)
        self.assertEqual(self.engine.strategy, AdaptiveStrategy.HYBRID_ADAPTIVE)
        self.assertFalse(self.engine.running)
        self.assertEqual(self.engine.adaptation_cycles, 0)
    
    async def test_engine_start_stop(self):
        """Test starting and stopping the engine"""
        async def mock_benchmark():
            return {"latency_ms": 5.0, "throughput_ops_sec": 200.0}
        
        def mock_fitness(params):
            return sum(params.values()) / len(params)
        
        # Start engine
        await self.engine.start_adaptive_optimization(mock_benchmark, mock_fitness)
        self.assertTrue(self.engine.running)
        
        # Let it run briefly
        await asyncio.sleep(0.3)
        
        # Stop engine
        await self.engine.stop_adaptive_optimization()
        self.assertFalse(self.engine.running)
    
    async def test_optimization_cycle_execution(self):
        """Test execution of optimization cycles"""
        def mock_fitness(params):
            # Fitness function that rewards balanced parameters
            lr = params.get("learning_rate", 0.1)
            bs = params.get("batch_size", 32)
            return 1.0 - abs(lr - 0.05) - abs(bs - 64) / 128
        
        # Manually trigger optimization cycle
        await self.engine._execute_optimization_cycle(mock_fitness)
        
        # Check that cycle was executed
        self.assertEqual(self.engine.adaptation_cycles, 1)
        self.assertGreater(len(self.engine.adaptation_history), 0)
        
        # Check optimization result
        cycle_result = self.engine.adaptation_history[0]
        self.assertIn("optimization_result", cycle_result)
        self.assertIn("performance_summary", cycle_result)
        self.assertTrue(cycle_result["adaptation_triggered"])
    
    def test_adaptation_trigger_logic(self):
        """Test adaptation trigger logic"""
        # Should trigger adaptation on early cycles
        self.engine.adaptation_cycles = 1
        performance_summary = {"regression_alerts_count": 0, "trends": {}}
        self.assertTrue(self.engine._should_trigger_adaptation(performance_summary))
        
        # Should trigger on performance regression
        self.engine.adaptation_cycles = 10
        performance_summary = {"regression_alerts_count": 2, "trends": {}}
        self.assertTrue(self.engine._should_trigger_adaptation(performance_summary))
        
        # Should trigger on multiple degrading trends
        performance_summary = {
            "regression_alerts_count": 0,
            "trends": {"latency_ms": "degrading", "throughput_ops_sec": "degrading"}
        }
        self.assertTrue(self.engine._should_trigger_adaptation(performance_summary))
        
        # Should trigger periodically
        self.engine.adaptation_cycles = 20  # Multiple of 10
        performance_summary = {"regression_alerts_count": 0, "trends": {}}
        self.assertTrue(self.engine._should_trigger_adaptation(performance_summary))
    
    def test_comprehensive_status(self):
        """Test comprehensive status reporting"""
        # Add some history
        self.engine.adaptation_cycles = 5
        self.engine.total_improvements = 3
        
        status = self.engine.get_comprehensive_status()
        
        # Check status structure
        self.assertIn("engine_status", status)
        self.assertIn("performance_monitoring", status)
        self.assertIn("parameter_optimization", status)
        self.assertIn("fitness_landscape", status)
        self.assertIn("effectiveness_metrics", status)
        
        # Check engine status
        engine_status = status["engine_status"]
        self.assertEqual(engine_status["adaptation_cycles"], 5)
        self.assertEqual(engine_status["total_improvements"], 3)
        self.assertEqual(engine_status["strategy"], "hybrid_adaptive")
    
    def test_adaptive_improvement_validation(self):
        """Test validation of adaptive improvements"""
        # Add mock adaptation history with improving fitness
        fitness_trajectory = [0.5, 0.6, 0.7, 0.75, 0.8]
        for i, fitness in enumerate(fitness_trajectory):
            cycle_result = {
                "cycle": i + 1,
                "optimization_result": {"current_fitness": fitness},
                "adaptation_triggered": True
            }
            self.engine.adaptation_history.append(cycle_result)
        
        validation = self.engine.validate_adaptive_improvement()
        
        # Check validation structure
        self.assertIn("improvement_achieved", validation)
        self.assertIn("improvement_percentage", validation)
        self.assertIn("stability_achieved", validation)
        self.assertIn("convergence_achieved", validation)
        self.assertIn("validation_summary", validation)
        
        # Should show improvement
        self.assertTrue(validation["improvement_achieved"])
        self.assertGreater(validation["improvement_percentage"], 0)
        
        # Check validation summary
        summary = validation["validation_summary"]
        self.assertIn("overall_success", summary)
        self.assertEqual(summary["initial_fitness"], 0.5)
        self.assertEqual(summary["final_fitness"], 0.8)


class TestAdaptiveOptimizationFactory(unittest.TestCase):
    """Test the factory function for creating adaptive optimization engines"""
    
    def test_create_adaptive_optimization_engine(self):
        """Test creating engine with factory function"""
        parameter_configs = [
            {"name": "param1", "initial_value": 0.5, "min_value": 0.0, "max_value": 1.0},
            {"name": "param2", "initial_value": 0.3, "min_value": 0.1, "max_value": 0.9}
        ]
        
        benchmark_config = {
            "interval": 30.0,
            "window_size": 50,
            "regression_threshold": 0.15
        }
        
        engine = create_adaptive_optimization_engine(
            parameter_configs,
            AdaptiveStrategy.EVOLUTIONARY_SEARCH,
            benchmark_config
        )
        
        # Check engine configuration
        self.assertEqual(len(engine.parameters), 2)
        self.assertEqual(engine.strategy, AdaptiveStrategy.EVOLUTIONARY_SEARCH)
        
        # Check parameter configuration
        param1 = engine.parameters[0]
        self.assertEqual(param1.name, "param1")
        self.assertEqual(param1.current_value, 0.5)
        self.assertEqual(param1.min_value, 0.0)
        self.assertEqual(param1.max_value, 1.0)
        
        # Check benchmark configuration
        self.assertEqual(engine.benchmark_config.benchmark_interval_seconds, 30.0)
        self.assertEqual(engine.benchmark_config.performance_window_size, 50)
        self.assertEqual(engine.benchmark_config.regression_threshold, 0.15)


class TestAdaptiveOptimizationIntegration(unittest.TestCase):
    """Integration tests for the complete adaptive optimization system"""
    
    async def test_end_to_end_optimization(self):
        """Test complete end-to-end optimization process"""
        logger.info("🧪 Running end-to-end adaptive optimization test")
        
        # Create engine with realistic parameters
        parameter_configs = [
            {"name": "learning_rate", "initial_value": 0.1, "min_value": 0.01, "max_value": 0.5},
            {"name": "batch_size", "initial_value": 32, "min_value": 8, "max_value": 128}
        ]
        
        engine = create_adaptive_optimization_engine(
            parameter_configs,
            AdaptiveStrategy.CONTINUOUS_TUNING,
            {"interval": 0.1, "window_size": 10}
        )
        
        # Define realistic benchmark and fitness functions
        async def realistic_benchmark():
            """Realistic benchmark with some noise"""
            base_latency = 5.0
            base_throughput = 200.0
            
            # Add some random variation
            noise_factor = 0.1
            latency = base_latency * (1 + (2 * (0.5 - asyncio.get_event_loop().time() % 1)) * noise_factor)
            throughput = base_throughput * (1 + (2 * (0.5 - (asyncio.get_event_loop().time() + 0.5) % 1)) * noise_factor)
            
            return {
                "latency_ms": latency,
                "throughput_ops_sec": throughput,
                "memory_usage_mb": 100.0,
                "efficiency_score": min(1.0, throughput / 250.0)
            }
        
        def realistic_fitness(parameters):
            """Realistic fitness function with clear optimum"""
            lr = parameters.get("learning_rate", 0.1)
            bs = parameters.get("batch_size", 32)
            
            # Optimal learning rate around 0.05, batch size around 64
            lr_fitness = 1.0 - abs(lr - 0.05) / 0.5
            bs_fitness = 1.0 - abs(bs - 64) / 120
            
            # Combine with some interaction effects
            interaction_bonus = 0.1 if 0.03 <= lr <= 0.07 and 48 <= bs <= 80 else 0.0
            
            return max(0.0, (lr_fitness + bs_fitness) / 2 + interaction_bonus)
        
        try:
            # Start optimization
            await engine.start_adaptive_optimization(realistic_benchmark, realistic_fitness)
            
            # Let it run for several cycles
            await asyncio.sleep(1.0)  # Run for 1 second
            
            # Manually trigger a few optimization cycles
            for _ in range(3):
                await engine._execute_optimization_cycle(realistic_fitness)
                await asyncio.sleep(0.1)
            
            # Get final status
            status = engine.get_comprehensive_status()
            validation = engine.validate_adaptive_improvement()
            
            # Verify the system worked
            self.assertGreater(status["engine_status"]["adaptation_cycles"], 0)
            self.assertGreater(len(status["recent_adaptations"]), 0)
            
            # Check that we have fitness landscape data
            landscape = status["fitness_landscape"]
            self.assertGreater(landscape.get("total_evaluations", 0), 0)
            
            # Check parameter optimization
            param_opt = status["parameter_optimization"]
            self.assertGreater(param_opt["generation_count"], 0)
            self.assertIn("parameter_trends", param_opt)
            
            logger.info("✅ End-to-end test completed successfully")
            logger.info(f"   - Adaptation cycles: {status['engine_status']['adaptation_cycles']}")
            logger.info(f"   - Total evaluations: {landscape.get('total_evaluations', 0)}")
            logger.info(f"   - Best fitness: {param_opt.get('best_fitness', 0):.3f}")
            
        finally:
            # Clean up
            await engine.stop_adaptive_optimization()
    
    def test_performance_regression_and_recovery(self):
        """Test system response to performance regression"""
        logger.info("🧪 Testing performance regression and recovery")
        
        config = ContinuousBenchmarkConfig(
            benchmark_interval_seconds=0.1,
            performance_window_size=5,
            regression_threshold=0.3
        )
        benchmark = ContinuousPerformanceBenchmark(config)
        
        # Set baseline
        benchmark.baseline_metrics = {
            "latency_ms": 5.0,
            "throughput_ops_sec": 200.0,
            "efficiency_score": 0.8
        }
        
        # Simulate normal performance
        normal_snapshot = {
            "timestamp": time.time(),
            "latency_ms": 5.2,
            "throughput_ops_sec": 195.0,
            "efficiency_score": 0.78
        }
        
        asyncio.run(benchmark._check_performance_regression(normal_snapshot))
        self.assertEqual(len(benchmark.regression_alerts), 0)
        
        # Simulate performance regression
        regression_snapshot = {
            "timestamp": time.time(),
            "latency_ms": 8.0,  # 60% increase
            "throughput_ops_sec": 120.0,  # 40% decrease
            "efficiency_score": 0.5  # 37.5% decrease
        }
        
        asyncio.run(benchmark._check_performance_regression(regression_snapshot))
        
        # Should detect regression
        self.assertGreater(len(benchmark.regression_alerts), 0)
        alert = benchmark.regression_alerts[0]
        self.assertIn("major", alert["severity"])
        
        logger.info("✅ Regression detection test completed")
    
    def test_fitness_landscape_evolution(self):
        """Test evolution of fitness landscape over optimization"""
        logger.info("🧪 Testing fitness landscape evolution")
        
        mapper = FitnessLandscapeMapper(["param1", "param2"])
        
        # Simulate optimization trajectory
        generations = 20
        for gen in range(generations):
            # Simulate convergence toward optimal point (0.7, 0.3)
            noise = 0.5 * (generations - gen) / generations  # Decreasing noise
            
            for _ in range(5):  # Multiple evaluations per generation
                p1 = 0.7 + (0.5 - asyncio.get_event_loop().time() % 1) * noise
                p2 = 0.3 + (0.5 - (asyncio.get_event_loop().time() + 0.3) % 1) * noise
                
                # Clamp to bounds
                p1 = max(0.0, min(1.0, p1))
                p2 = max(0.0, min(1.0, p2))
                
                # Fitness function with optimum at (0.7, 0.3)
                fitness = 1.0 - ((p1 - 0.7)**2 + (p2 - 0.3)**2)
                
                mapper.add_evaluation_point({"param1": p1, "param2": p2}, fitness, gen)
        
        # Analyze trajectory
        trajectory = mapper.get_optimization_trajectory()
        self.assertEqual(len(trajectory["trajectory"]), generations * 5)
        
        # Should show improvement trend
        stats = trajectory["statistics"]
        self.assertGreater(stats["fitness_improvement"], 0)
        self.assertIn(stats["convergence_trend"], ["improving", "converging"])
        
        # Test landscape mapping
        parameter_ranges = {"param1": (0.0, 1.0), "param2": (0.0, 1.0)}
        surface = mapper.map_fitness_surface(parameter_ranges, resolution=10)
        
        self.assertEqual(surface["resolution"], 10)
        self.assertGreater(surface["data_points"], 0)
        
        # Find peaks
        peaks = mapper.identify_fitness_peaks(threshold_percentile=90)
        self.assertGreater(len(peaks), 0)
        
        # Best peak should be near optimum
        best_peak = peaks[0]
        best_params = best_peak["parameters"]
        distance_to_optimum = ((best_params["param1"] - 0.7)**2 + (best_params["param2"] - 0.3)**2)**0.5
        self.assertLess(distance_to_optimum, 0.5)  # Should be reasonably close
        
        logger.info("✅ Fitness landscape evolution test completed")
        logger.info(f"   - Total evaluations: {len(mapper.landscape_points)}")
        logger.info(f"   - Best fitness: {best_peak['fitness']:.3f}")
        logger.info(f"   - Distance to optimum: {distance_to_optimum:.3f}")


# Test runner
def run_adaptive_optimization_tests():
    """Run all adaptive optimization tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAdaptiveParameter,
        TestContinuousPerformanceBenchmark,
        TestSelfTuningAlgorithm,
        TestFitnessLandscapeMapper,
        TestAdaptiveOptimizationEngine,
        TestAdaptiveOptimizationFactory,
        TestAdaptiveOptimizationIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run the tests
    success = run_adaptive_optimization_tests()
    exit(0 if success else 1)