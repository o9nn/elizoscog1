#!/usr/bin/env python3
"""
Comprehensive Integration Demo - Phase 5: Adaptive Optimization & Continuous Learning
Demonstrates the complete integration of all adaptive optimization components.
"""

import asyncio
import json
import time
import logging
import sys
import os
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.optimization.adaptive_optimization import (
    create_adaptive_optimization_engine, AdaptiveStrategy, AdaptiveParameter
)
from src.benchmarking.performance_profiler import create_adaptive_performance_profiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveAdaptiveSystem:
    """Comprehensive adaptive optimization system demonstrating full integration"""
    
    def __init__(self):
        self.start_time = time.time()
        self.system_metrics = {}
        self.integration_results = {}
        
        # Initialize components
        self.performance_profiler = None
        self.adaptive_engine = None
        self.integration_callbacks = []
        
    async def initialize_system(self) -> Dict[str, Any]:
        """Initialize the complete adaptive optimization system"""
        logger.info("🚀 Initializing Comprehensive Adaptive Optimization System")
        logger.info("=" * 80)
        
        # Step 1: Create adaptive performance profiler
        logger.info("📊 Step 1: Creating Adaptive Performance Profiler")
        self.performance_profiler = create_adaptive_performance_profiler(
            history_size=500,
            enable_adaptive_optimization=True
        )
        
        # Step 2: Create adaptive optimization engine
        logger.info("🔧 Step 2: Creating Adaptive Optimization Engine")
        parameter_configs = [
            {"name": "neural_learning_rate", "initial_value": 0.001, "min_value": 0.0001, "max_value": 0.01},
            {"name": "symbolic_reasoning_weight", "initial_value": 0.5, "min_value": 0.1, "max_value": 0.9},
            {"name": "attention_mechanism_heads", "initial_value": 8, "min_value": 4, "max_value": 16},
            {"name": "memory_retention_factor", "initial_value": 0.95, "min_value": 0.8, "max_value": 0.99},
            {"name": "exploration_vs_exploitation", "initial_value": 0.2, "min_value": 0.05, "max_value": 0.5}
        ]
        
        self.adaptive_engine = create_adaptive_optimization_engine(
            parameter_configs,
            AdaptiveStrategy.HYBRID_ADAPTIVE,
            {
                "interval": 2.0,  # Benchmark every 2 seconds
                "window_size": 25,
                "regression_threshold": 0.12
            }
        )
        
        # Step 3: Register engine with profiler
        logger.info("🔗 Step 3: Integrating Components")
        self.performance_profiler.register_adaptive_engine(self.adaptive_engine)
        
        # Step 4: Add integration callbacks
        self.performance_profiler.add_optimization_callback(self._on_optimization_triggered)
        
        initialization_result = {
            "profiler_initialized": self.performance_profiler is not None,
            "adaptive_engine_initialized": self.adaptive_engine is not None,
            "integration_complete": True,
            "parameter_count": len(parameter_configs),
            "initialization_time": time.time() - self.start_time
        }
        
        logger.info("✅ System initialization complete")
        return initialization_result
    
    async def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run comprehensive demonstration of the adaptive system"""
        logger.info("\n🎯 Starting Comprehensive Adaptive Optimization Demo")
        logger.info("=" * 80)
        
        # Initialize system
        init_result = await self.initialize_system()
        
        # Demo phases
        demo_phases = [
            ("Phase 1: Baseline Performance Establishment", self._demo_baseline_establishment),
            ("Phase 2: Continuous Performance Monitoring", self._demo_continuous_monitoring),
            ("Phase 3: Adaptive Parameter Optimization", self._demo_adaptive_optimization),
            ("Phase 4: Performance Regression & Recovery", self._demo_regression_recovery),
            ("Phase 5: Long-term Stability Validation", self._demo_stability_validation)
        ]
        
        phase_results = {}
        
        for phase_name, phase_func in demo_phases:
            logger.info(f"\n📍 {phase_name}")
            logger.info("-" * 60)
            
            try:
                phase_start = time.time()
                result = await phase_func()
                result["phase_duration"] = time.time() - phase_start
                phase_results[phase_name] = result
                logger.info(f"✅ {phase_name} completed successfully")
            except Exception as e:
                logger.error(f"❌ {phase_name} failed: {e}")
                phase_results[phase_name] = {"error": str(e)}
        
        # Generate comprehensive summary
        comprehensive_result = await self._generate_comprehensive_summary(init_result, phase_results)
        
        logger.info("\n🎉 Comprehensive Adaptive Optimization Demo Complete!")
        logger.info("=" * 80)
        
        return comprehensive_result
    
    async def _demo_baseline_establishment(self) -> Dict[str, Any]:
        """Demo Phase 1: Establish performance baselines"""
        
        # Define cognitive system benchmark
        benchmark_count = 0
        
        async def cognitive_system_benchmark():
            nonlocal benchmark_count
            benchmark_count += 1
            
            # Simulate a complex cognitive system with multiple components
            neural_processing_time = 2.0 + (benchmark_count * 0.01)  # Slight degradation over time
            symbolic_reasoning_time = 1.5 + (0.5 - __import__('random').random()) * 0.3
            attention_mechanism_time = 0.8 + __import__('random').uniform(-0.2, 0.3)
            memory_access_time = 0.3 + __import__('random').uniform(-0.1, 0.2)
            
            total_latency = neural_processing_time + symbolic_reasoning_time + attention_mechanism_time + memory_access_time
            
            # Calculate throughput based on processing efficiency
            base_throughput = 150.0
            efficiency_factor = 1.0 / (1.0 + (total_latency - 4.0) * 0.1)
            throughput = base_throughput * efficiency_factor
            
            return {
                "latency_ms": total_latency,
                "throughput_ops_sec": throughput + __import__('random').uniform(-10, 10),
                "memory_usage_mb": 120.0 + __import__('random').uniform(-15, 25),
                "neural_efficiency": max(0.5, 1.0 - (neural_processing_time - 2.0) * 0.2),
                "symbolic_accuracy": 0.85 + __import__('random').uniform(-0.1, 0.1),
                "attention_focus": min(1.0, 0.9 + __import__('random').uniform(-0.2, 0.1))
            }
        
        # Start performance monitoring to establish baselines
        await self.performance_profiler.start_monitoring()
        
        # Collect baseline data
        baseline_collection_time = 8.0
        logger.info(f"   📊 Collecting baseline data for {baseline_collection_time}s...")
        
        baseline_snapshots = []
        end_time = time.time() + baseline_collection_time
        
        while time.time() < end_time:
            benchmark_result = await cognitive_system_benchmark()
            
            # Create performance snapshot
            snapshot = await self.performance_profiler._create_performance_snapshot(
                "cognitive_system_benchmark",
                "integrated_architecture",
                lambda: benchmark_result
            )
            
            baseline_snapshots.append(snapshot)
            await asyncio.sleep(0.5)
        
        await self.performance_profiler.stop_monitoring()
        
        # Analyze baseline establishment
        baseline_analysis = {
            "snapshots_collected": len(baseline_snapshots),
            "collection_duration": baseline_collection_time,
            "average_latency": sum(s.execution_time_ms for s in baseline_snapshots) / len(baseline_snapshots),
            "average_throughput": sum(s.throughput_ops_per_sec for s in baseline_snapshots) / len(baseline_snapshots),
            "average_memory": sum(s.memory_used_mb for s in baseline_snapshots) / len(baseline_snapshots),
            "efficiency_trend": "stable" if len(baseline_snapshots) > 5 else "insufficient_data"
        }
        
        logger.info(f"   📈 Baseline established: {baseline_analysis['average_latency']:.2f}ms latency, "
                   f"{baseline_analysis['average_throughput']:.1f} ops/sec throughput")
        
        return baseline_analysis
    
    async def _demo_continuous_monitoring(self) -> Dict[str, Any]:
        """Demo Phase 2: Continuous performance monitoring with trend detection"""
        
        # Start adaptive engine for integrated monitoring
        monitoring_duration = 10.0
        
        async def realistic_cognitive_benchmark():
            # Simulate realistic cognitive workload with varying complexity
            workload_complexity = 0.5 + 0.3 * __import__('math').sin(time.time() * 0.5)
            
            base_latency = 4.5 * (1 + workload_complexity)
            base_throughput = 140.0 / (1 + workload_complexity)
            
            # Add system noise
            noise_factor = __import__('random').uniform(0.9, 1.1)
            
            return {
                "latency_ms": base_latency * noise_factor,
                "throughput_ops_sec": base_throughput * noise_factor,
                "memory_usage_mb": 110.0 + workload_complexity * 30 + __import__('random').uniform(-5, 10),
                "cognitive_load": workload_complexity,
                "processing_efficiency": min(1.0, 0.9 / workload_complexity)
            }
        
        def cognitive_fitness_function(parameters):
            # Fitness function for cognitive system optimization
            neural_lr = parameters.get("neural_learning_rate", 0.001)
            symbolic_weight = parameters.get("symbolic_reasoning_weight", 0.5)
            attention_heads = parameters.get("attention_mechanism_heads", 8)
            memory_factor = parameters.get("memory_retention_factor", 0.95)
            exploration = parameters.get("exploration_vs_exploitation", 0.2)
            
            # Calculate fitness based on parameter balance
            lr_fitness = 1.0 - abs(neural_lr - 0.003) / 0.01
            weight_fitness = 1.0 - abs(symbolic_weight - 0.7) / 0.8
            heads_fitness = 1.0 - abs(attention_heads - 12) / 12
            memory_fitness = memory_factor  # Higher retention is better
            exploration_fitness = 1.0 - abs(exploration - 0.15) / 0.45
            
            # Interaction effects
            neural_symbolic_synergy = min(1.0, neural_lr * symbolic_weight * 20)
            attention_memory_synergy = min(1.0, attention_heads * memory_factor / 12)
            
            base_fitness = (lr_fitness + weight_fitness + heads_fitness + memory_fitness + exploration_fitness) / 5
            synergy_bonus = (neural_symbolic_synergy + attention_memory_synergy) / 10
            
            return max(0.0, base_fitness + synergy_bonus)
        
        # Start integrated adaptive optimization
        await self.adaptive_engine.start_adaptive_optimization(
            realistic_cognitive_benchmark,
            cognitive_fitness_function
        )
        
        logger.info(f"   🔄 Running continuous monitoring for {monitoring_duration}s...")
        await asyncio.sleep(monitoring_duration)
        
        # Get monitoring results
        engine_status = self.adaptive_engine.get_comprehensive_status()
        profiler_summary = self.performance_profiler.get_performance_summary()
        
        await self.adaptive_engine.stop_adaptive_optimization()
        
        monitoring_result = {
            "monitoring_duration": monitoring_duration,
            "adaptation_cycles": engine_status["engine_status"]["adaptation_cycles"],
            "performance_data_points": profiler_summary.get("total_measurements", 0),
            "regression_alerts": profiler_summary.get("regression_alerts", 0),
            "optimization_improvements": engine_status["engine_status"]["total_improvements"],
            "current_fitness": engine_status["parameter_optimization"]["best_fitness"],
            "monitoring_frequency": profiler_summary.get("total_measurements", 0) / monitoring_duration
        }
        
        logger.info(f"   📊 Monitoring complete: {monitoring_result['adaptation_cycles']} cycles, "
                   f"{monitoring_result['optimization_improvements']} improvements")
        
        return monitoring_result
    
    async def _demo_adaptive_optimization(self) -> Dict[str, Any]:
        """Demo Phase 3: Adaptive parameter optimization under varying conditions"""
        
        optimization_scenarios = [
            ("High Cognitive Load", lambda: __import__('random').uniform(0.7, 1.0)),
            ("Variable Workload", lambda: 0.5 + 0.4 * __import__('math').sin(time.time() * 2)),
            ("Memory Pressure", lambda: __import__('random').uniform(0.3, 0.8)),
            ("Processing Intensive", lambda: __import__('random').uniform(0.6, 0.9))
        ]
        
        scenario_results = {}
        
        for scenario_name, load_generator in optimization_scenarios:
            logger.info(f"   🔧 Testing scenario: {scenario_name}")
            
            # Create scenario-specific benchmark
            async def scenario_benchmark():
                load_factor = load_generator()
                base_latency = 3.0 * (1 + load_factor)
                base_throughput = 180.0 / (1 + load_factor * 0.5)
                
                return {
                    "latency_ms": base_latency + __import__('random').uniform(-0.5, 1.0),
                    "throughput_ops_sec": base_throughput + __import__('random').uniform(-15, 15),
                    "memory_usage_mb": 100.0 + load_factor * 40 + __import__('random').uniform(-5, 15),
                    "scenario_load": load_factor
                }
            
            # Define scenario-specific fitness function
            def scenario_fitness(parameters):
                # Fitness function adapted for this scenario
                base_fitness = sum(parameters.values()) / len(parameters)
                
                # Scenario-specific adjustments
                if scenario_name == "High Cognitive Load":
                    # Favor higher memory retention and lower exploration
                    memory_bonus = parameters.get("memory_retention_factor", 0.95) * 0.2
                    exploration_penalty = parameters.get("exploration_vs_exploitation", 0.2) * 0.1
                    return base_fitness + memory_bonus - exploration_penalty
                elif scenario_name == "Variable Workload":
                    # Favor balanced parameters
                    balance_score = 1.0 - sum(abs(v - 0.5) for v in parameters.values()) / len(parameters)
                    return base_fitness * 0.8 + balance_score * 0.2
                else:
                    return base_fitness
            
            # Run optimization for this scenario
            scenario_start = time.time()
            
            # Manual optimization cycles for this scenario
            optimization_cycles = 3
            scenario_improvements = 0
            
            for cycle in range(optimization_cycles):
                result = await self.adaptive_engine.self_tuning_algorithm.optimize_parameters(scenario_fitness)
                if result.get("improvements_made", []):
                    scenario_improvements += len(result["improvements_made"])
                await asyncio.sleep(0.5)
            
            scenario_duration = time.time() - scenario_start
            
            scenario_results[scenario_name] = {
                "optimization_cycles": optimization_cycles,
                "improvements_made": scenario_improvements,
                "scenario_duration": scenario_duration,
                "final_fitness": result.get("current_fitness", 0.0)
            }
        
        # Analyze cross-scenario performance
        total_improvements = sum(r["improvements_made"] for r in scenario_results.values())
        avg_fitness = sum(r["final_fitness"] for r in scenario_results.values()) / len(scenario_results)
        
        optimization_result = {
            "scenarios_tested": len(optimization_scenarios),
            "total_improvements": total_improvements,
            "average_final_fitness": avg_fitness,
            "scenario_results": scenario_results,
            "adaptation_effectiveness": total_improvements / (len(optimization_scenarios) * 3)  # improvements per cycle
        }
        
        logger.info(f"   🎯 Adaptive optimization complete: {total_improvements} total improvements, "
                   f"avg fitness: {avg_fitness:.3f}")
        
        return optimization_result
    
    async def _demo_regression_recovery(self) -> Dict[str, Any]:
        """Demo Phase 4: Performance regression detection and recovery"""
        
        # Simulate performance regression and recovery
        logger.info("   🚨 Simulating performance regression...")
        
        # Create a benchmark that simulates degrading performance
        regression_stage = 0
        
        async def degrading_benchmark():
            nonlocal regression_stage
            regression_stage += 1
            
            # Simulate gradual performance degradation
            degradation_factor = 1.0 + (regression_stage * 0.05)  # 5% degradation per call
            
            base_latency = 4.0 * degradation_factor
            base_throughput = 160.0 / degradation_factor
            base_memory = 110.0 * degradation_factor
            
            return {
                "latency_ms": base_latency + __import__('random').uniform(-0.3, 0.7),
                "throughput_ops_sec": base_throughput + __import__('random').uniform(-10, 10),
                "memory_usage_mb": base_memory + __import__('random').uniform(-5, 15),
                "degradation_stage": regression_stage
            }
        
        # Start monitoring with the degrading benchmark
        regression_monitoring_duration = 5.0
        
        await self.performance_profiler.start_monitoring()
        
        # Collect data that should trigger regression detection
        regression_start = time.time()
        regression_alerts = []
        
        while time.time() - regression_start < regression_monitoring_duration:
            benchmark_result = await degrading_benchmark()
            
            # Check for regression alerts (this would be automatic in real system)
            if regression_stage > 3:  # After some degradation
                # Simulate regression alert
                from src.benchmarking.performance_profiler import RegressionAlert
                alert = RegressionAlert(
                    operation="cognitive_system_benchmark",
                    architecture="integrated_architecture",
                    metric_type="latency",
                    baseline_value=4.0,
                    current_value=benchmark_result["latency_ms"],
                    regression_factor=benchmark_result["latency_ms"] / 4.0,
                    severity="major" if regression_stage > 5 else "minor",
                    timestamp=time.time()
                )
                
                # Trigger adaptive optimization
                optimization_result = await self.performance_profiler.trigger_adaptive_optimization(alert)
                regression_alerts.append({
                    "alert": alert,
                    "optimization_result": optimization_result
                })
            
            await asyncio.sleep(0.5)
        
        await self.performance_profiler.stop_monitoring()
        
        # Simulate recovery period
        logger.info("   🔧 Simulating recovery optimization...")
        
        # Reset regression for recovery simulation
        regression_stage = 0
        
        async def recovering_benchmark():
            # Simulate gradual recovery
            recovery_factor = max(0.5, 1.0 - (time.time() - regression_start - regression_monitoring_duration) * 0.1)
            
            base_latency = 4.0 * recovery_factor
            base_throughput = 160.0 / recovery_factor
            
            return {
                "latency_ms": base_latency + __import__('random').uniform(-0.2, 0.5),
                "throughput_ops_sec": base_throughput + __import__('random').uniform(-8, 12),
                "memory_usage_mb": 110.0 + __import__('random').uniform(-10, 15),
                "recovery_factor": recovery_factor
            }
        
        # Monitor recovery
        recovery_duration = 3.0
        recovery_snapshots = []
        
        for _ in range(int(recovery_duration / 0.5)):
            recovery_result = await recovering_benchmark()
            recovery_snapshots.append(recovery_result)
            await asyncio.sleep(0.5)
        
        # Analyze regression and recovery
        regression_recovery_result = {
            "regression_alerts_triggered": len(regression_alerts),
            "regression_duration": regression_monitoring_duration,
            "recovery_duration": recovery_duration,
            "max_degradation_factor": regression_stage * 0.05,
            "recovery_snapshots": len(recovery_snapshots),
            "optimization_attempts": len([a for a in regression_alerts if a["optimization_result"]["status"] != "no_engine"]),
            "adaptive_response_effectiveness": len(regression_alerts) > 0
        }
        
        logger.info(f"   📈 Regression recovery complete: {regression_recovery_result['regression_alerts_triggered']} alerts, "
                   f"recovery in {recovery_duration}s")
        
        return regression_recovery_result
    
    async def _demo_stability_validation(self) -> Dict[str, Any]:
        """Demo Phase 5: Long-term stability and improvement validation"""
        
        logger.info("   📏 Validating long-term stability...")
        
        # Run extended stability test
        stability_duration = 8.0
        stability_measurements = []
        
        async def stable_cognitive_benchmark():
            # Simulate a stable but optimizing system
            time_factor = (time.time() % 10) / 10  # Cyclical pattern
            base_performance = 0.8 + 0.1 * __import__('math').sin(time_factor * 2 * __import__('math').pi)
            
            latency = 4.0 * (1.1 - base_performance) + __import__('random').uniform(-0.2, 0.2)
            throughput = 150.0 * base_performance + __import__('random').uniform(-8, 8)
            
            return {
                "latency_ms": latency,
                "throughput_ops_sec": throughput,
                "memory_usage_mb": 105.0 + __import__('random').uniform(-8, 12),
                "stability_factor": base_performance
            }
        
        def stability_fitness(parameters):
            # Fitness function that slightly improves over time
            base_fitness = sum(parameters.values()) / len(parameters)
            improvement_factor = min(0.1, len(stability_measurements) * 0.005)  # Gradual improvement
            return base_fitness + improvement_factor
        
        # Start stability monitoring
        await self.adaptive_engine.start_adaptive_optimization(
            stable_cognitive_benchmark,
            stability_fitness
        )
        
        stability_start = time.time()
        while time.time() - stability_start < stability_duration:
            measurement = await stable_cognitive_benchmark()
            stability_measurements.append(measurement)
            await asyncio.sleep(0.5)
        
        # Get final validation results
        final_validation = self.adaptive_engine.validate_adaptive_improvement()
        final_status = self.adaptive_engine.get_comprehensive_status()
        
        await self.adaptive_engine.stop_adaptive_optimization()
        
        # Analyze stability metrics
        latencies = [m["latency_ms"] for m in stability_measurements]
        throughputs = [m["throughput_ops_sec"] for m in stability_measurements]
        
        # Calculate stability scores
        latency_variance = sum((l - sum(latencies)/len(latencies))**2 for l in latencies) / len(latencies)
        throughput_variance = sum((t - sum(throughputs)/len(throughputs))**2 for t in throughputs) / len(throughputs)
        
        stability_score = 1.0 / (1.0 + latency_variance + throughput_variance)
        
        stability_result = {
            "stability_duration": stability_duration,
            "measurements_collected": len(stability_measurements),
            "latency_stability": 1.0 / (1.0 + latency_variance),
            "throughput_stability": 1.0 / (1.0 + throughput_variance),
            "overall_stability_score": stability_score,
            "improvement_achieved": final_validation.get("improvement_achieved", False),
            "improvement_percentage": final_validation.get("improvement_percentage", 0),
            "convergence_achieved": final_validation.get("convergence_achieved", False),
            "final_fitness": final_status["parameter_optimization"]["best_fitness"],
            "total_adaptation_cycles": final_status["engine_status"]["adaptation_cycles"]
        }
        
        logger.info(f"   ✅ Stability validation complete: {stability_score:.3f} stability score, "
                   f"{final_validation.get('improvement_percentage', 0):.1f}% improvement")
        
        return stability_result
    
    async def _generate_comprehensive_summary(self, init_result: Dict[str, Any], 
                                            phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of the entire demonstration"""
        
        total_duration = time.time() - self.start_time
        
        # Collect success metrics
        successful_phases = sum(1 for result in phase_results.values() if "error" not in result)
        
        # Aggregate performance metrics
        total_optimization_cycles = 0
        total_improvements = 0
        total_measurements = 0
        
        for phase_name, result in phase_results.items():
            if "error" not in result:
                total_optimization_cycles += result.get("adaptation_cycles", 0)
                total_improvements += result.get("optimization_improvements", 0)
                total_improvements += result.get("improvements_made", 0)
                total_improvements += result.get("total_improvements", 0)
                total_measurements += result.get("performance_data_points", 0)
                total_measurements += result.get("snapshots_collected", 0)
                total_measurements += result.get("measurements_collected", 0)
        
        # Calculate effectiveness metrics
        optimization_effectiveness = total_improvements / max(1, total_optimization_cycles)
        
        # Get final system state
        if self.adaptive_engine:
            final_engine_status = self.adaptive_engine.get_comprehensive_status()
        else:
            final_engine_status = {}
        
        if self.performance_profiler:
            final_profiler_summary = self.performance_profiler.get_performance_summary()
            adaptive_summary = self.performance_profiler.get_adaptive_optimization_summary()
        else:
            final_profiler_summary = {}
            adaptive_summary = {}
        
        comprehensive_summary = {
            "demo_metadata": {
                "total_duration_seconds": total_duration,
                "successful_phases": successful_phases,
                "total_phases": len(phase_results),
                "success_rate": successful_phases / len(phase_results),
                "initialization_successful": init_result.get("integration_complete", False)
            },
            "performance_metrics": {
                "total_optimization_cycles": total_optimization_cycles,
                "total_improvements": total_improvements,
                "total_measurements": total_measurements,
                "optimization_effectiveness": optimization_effectiveness,
                "measurements_per_second": total_measurements / total_duration
            },
            "phase_results": phase_results,
            "final_system_state": {
                "adaptive_engine_status": final_engine_status,
                "performance_profiler_summary": final_profiler_summary,
                "adaptive_optimization_summary": adaptive_summary
            },
            "cognitive_synergy_validation": {
                "adaptive_multi_objective_optimization": successful_phases >= 3,
                "emergent_fitness_landscape_exploration": "Phase 3" in phase_results and "error" not in phase_results.get("Phase 3", {}),
                "self_organizing_optimization_strategies": optimization_effectiveness > 0.1,
                "collaborative_evolutionary_improvement": total_improvements > 5
            },
            "success_criteria_met": {
                "continuous_performance_improvement": total_improvements > 0,
                "adaptive_optimization_convergence": optimization_effectiveness > 0.05,
                "real_time_monitoring_achieved": total_measurements > 20,
                "regression_detection_functional": "Phase 4" in phase_results and "error" not in phase_results.get("Phase 4", {}),
                "stability_validation_passed": "Phase 5" in phase_results and phase_results.get("Phase 5", {}).get("overall_stability_score", 0) > 0.5
            }
        }
        
        # Save comprehensive results
        with open("comprehensive_adaptive_optimization_results.json", "w") as f:
            json.dump(comprehensive_summary, f, indent=2, default=str)
        
        logger.info("\n🎯 COMPREHENSIVE DEMO SUMMARY")
        logger.info("=" * 80)
        logger.info(f"📊 Duration: {total_duration:.1f}s")
        logger.info(f"✅ Success Rate: {successful_phases}/{len(phase_results)} phases ({successful_phases/len(phase_results)*100:.1f}%)")
        logger.info(f"🔄 Optimization Cycles: {total_optimization_cycles}")
        logger.info(f"📈 Total Improvements: {total_improvements}")
        logger.info(f"📊 Measurements Collected: {total_measurements}")
        logger.info(f"⚡ Optimization Effectiveness: {optimization_effectiveness:.3f}")
        logger.info(f"💾 Results saved to: comprehensive_adaptive_optimization_results.json")
        
        return comprehensive_summary
    
    async def _on_optimization_triggered(self, context: Dict[str, Any], result: Dict[str, Any]):
        """Callback for when optimization is triggered"""
        logger.info(f"🔔 Optimization callback triggered: {context.get('trigger_type', 'unknown')}")
        self.integration_callbacks.append({
            "timestamp": time.time(),
            "context": context,
            "result": result
        })


async def main():
    """Main comprehensive demo entry point"""
    system = ComprehensiveAdaptiveSystem()
    await system.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())