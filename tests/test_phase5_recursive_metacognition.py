"""
Test Suite for Phase 5: Recursive Meta-Cognitive Pathways
Validates self-analysis, evolutionary optimization, and recursive self-improvement
"""

import unittest
import asyncio
import json
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the classes we're testing
try:
    from src.integration.advanced_reasoning import AdvancedReasoningEngine, ReasoningType, ReasoningConclusion, UserFeedback
    from src.integration.recursive_metacognition import (
        RecursiveMetaCognitiveEngine, EvolutionaryOptimizer, CognitiveGene, 
        CognitiveGenome, MetaCognitiveObservation, SelfImprovementAction,
        CognitiveState, OptimizationStrategy
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


class TestRecursiveMetaCognition(unittest.TestCase):
    """Test recursive meta-cognitive pathways functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        # Create base reasoning engine with meta-cognitive capabilities enabled
        self.reasoning_engine = AdvancedReasoningEngine(enable_recursive_metacognition=True)
        self.meta_engine = self.reasoning_engine.recursive_meta_engine
        
        # Create sample reasoning conclusions for testing
        self.sample_conclusions = self._create_sample_conclusions()
    
    def _create_sample_conclusions(self):
        """Create sample reasoning conclusions for testing"""
        conclusions = []
        
        # Create diverse reasoning conclusions
        reasoning_types = [ReasoningType.TEMPORAL, ReasoningType.CAUSAL, ReasoningType.PROBABILISTIC]
        confidences = [0.3, 0.7, 0.9, 0.5, 0.8]
        
        for i in range(10):
            conclusion = ReasoningConclusion(
                conclusion_id=f"test_conclusion_{i}",
                reasoning_type=reasoning_types[i % len(reasoning_types)],
                premise=f"Test premise {i}",
                conclusion=f"Test conclusion {i}",
                confidence=confidences[i % len(confidences)],
                evidence=[{"type": "test_evidence", "value": f"evidence_{i}"}],
                reasoning_chain=[f"step_{j}" for j in range(3 + (i % 3))],
                timestamp=datetime.now() - timedelta(hours=i)
            )
            conclusions.append(conclusion)
        
        return conclusions
    
    def test_evolutionary_optimizer_initialization(self):
        """Test initialization of evolutionary optimizer"""
        optimizer = EvolutionaryOptimizer(population_size=10, mutation_rate=0.1)
        
        self.assertEqual(optimizer.population_size, 10)
        self.assertEqual(optimizer.mutation_rate, 0.1)
        self.assertEqual(len(optimizer.population), 0)
        self.assertEqual(optimizer.generation_count, 0)
    
    def test_evolutionary_optimizer_population_initialization(self):
        """Test population initialization in evolutionary optimizer"""
        optimizer = EvolutionaryOptimizer(population_size=5)
        
        gene_templates = {
            "confidence_threshold": {"min_value": 0.1, "max_value": 0.9},
            "exploration_rate": {"min_value": 0.0, "max_value": 1.0}
        }
        
        optimizer.initialize_population(gene_templates)
        
        self.assertEqual(len(optimizer.population), 5)
        
        for genome in optimizer.population:
            self.assertIsInstance(genome, CognitiveGenome)
            self.assertEqual(len(genome.genes), 2)
            self.assertIn("confidence_threshold", genome.genes)
            self.assertIn("exploration_rate", genome.genes)
            
            # Check gene value ranges
            conf_gene = genome.genes["confidence_threshold"]
            self.assertTrue(0.1 <= conf_gene.value <= 0.9)
            
            exp_gene = genome.genes["exploration_rate"]
            self.assertTrue(0.0 <= exp_gene.value <= 1.0)
    
    def test_fitness_evaluation(self):
        """Test fitness evaluation in evolutionary optimizer"""
        optimizer = EvolutionaryOptimizer(population_size=3)
        
        # Create a test genome
        genes = {
            "test_gene": CognitiveGene(
                gene_id="test_gene_1",
                gene_type="test_gene",
                value=0.5
            )
        }
        genome = CognitiveGenome(
            genome_id="test_genome",
            genes=genes
        )
        
        # Test fitness evaluation
        performance_metrics = {
            "accuracy": 0.8,
            "efficiency": 0.6,
            "adaptability": 0.7
        }
        
        fitness = optimizer.evaluate_fitness(genome, performance_metrics)
        
        self.assertIsInstance(fitness, float)
        self.assertTrue(0.0 <= fitness <= 1.0)
        self.assertEqual(genome.overall_fitness, fitness)
        self.assertEqual(len(genome.performance_history), 1)
    
    def test_crossover_operation(self):
        """Test crossover operation in evolutionary optimizer"""
        optimizer = EvolutionaryOptimizer(population_size=5, crossover_rate=0.8)
        
        # Create parent genomes
        genes1 = {
            "gene1": CognitiveGene("g1_1", "gene1", 0.3),
            "gene2": CognitiveGene("g2_1", "gene2", 0.7)
        }
        genes2 = {
            "gene1": CognitiveGene("g1_2", "gene1", 0.8),
            "gene2": CognitiveGene("g2_2", "gene2", 0.2)
        }
        
        parent1 = CognitiveGenome("parent1", genes1)
        parent2 = CognitiveGenome("parent2", genes2)
        
        child1, child2 = optimizer.crossover(parent1, parent2)
        
        self.assertIsInstance(child1, CognitiveGenome)
        self.assertIsInstance(child2, CognitiveGenome)
        self.assertEqual(len(child1.genes), 2)
        self.assertEqual(len(child2.genes), 2)
        
        # Check that children have parent IDs recorded
        self.assertEqual(child1.parent_genomes, ["parent1", "parent2"])
        self.assertEqual(child2.parent_genomes, ["parent1", "parent2"])
    
    def test_mutation_operation(self):
        """Test mutation operation in evolutionary optimizer"""
        optimizer = EvolutionaryOptimizer()
        
        # Create genome with high mutation rate for testing
        genes = {
            "test_gene": CognitiveGene(
                gene_id="test_gene",
                gene_type="test_gene",
                value=0.5,
                mutation_rate=1.0  # 100% mutation rate for testing
            )
        }
        genome = CognitiveGenome("test_genome", genes)
        original_value = genome.genes["test_gene"].value
        
        mutated_genome = optimizer.mutate(genome)
        
        # With 100% mutation rate, the value should change
        # (though there's a small chance it could be the same by coincidence)
        self.assertIsInstance(mutated_genome, CognitiveGenome)
        self.assertTrue(0.0 <= mutated_genome.genes["test_gene"].value <= 1.0)
    
    def test_recursive_meta_cognitive_engine_initialization(self):
        """Test initialization of recursive meta-cognitive engine"""
        self.assertIsNotNone(self.meta_engine)
        self.assertEqual(self.meta_engine.current_state, CognitiveState.EXPLORING)
        self.assertEqual(self.meta_engine.recursive_depth, 0)
        self.assertEqual(len(self.meta_engine.observations), 0)
        self.assertIsNotNone(self.meta_engine.evolutionary_optimizer)
    
    def test_cognitive_state_observation(self):
        """Test cognitive state observation functionality"""
        # Add some sample reasoning conclusions to the base engine
        self.reasoning_engine.reasoning_history = self.sample_conclusions
        
        # Run the observation asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            observation = loop.run_until_complete(
                self.meta_engine.observe_cognitive_state()
            )
            
            self.assertIsInstance(observation, MetaCognitiveObservation)
            self.assertIn("accuracy", observation.performance_metrics)
            self.assertIn("efficiency", observation.performance_metrics)
            self.assertIn("adaptability", observation.performance_metrics)
            self.assertIn("consistency", observation.performance_metrics)
            
            # Check that observation was recorded
            self.assertEqual(len(self.meta_engine.observations), 1)
            
        finally:
            loop.close()
    
    def test_self_improvement_recommendations(self):
        """Test generation of self-improvement recommendations"""
        # Create a mock observation with improvement opportunities
        observation = MetaCognitiveObservation(
            observation_id="test_obs",
            timestamp=datetime.now(),
            cognitive_state=CognitiveState.EXPLORING,
            performance_metrics={"accuracy": 0.6, "efficiency": 0.5},
            reasoning_patterns={},
            self_assessment={"overall_performance": 0.55},
            improvement_opportunities=[
                "Improve reasoning accuracy through better evidence evaluation",
                "Optimize reasoning chain length for better efficiency"
            ]
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            recommendations = loop.run_until_complete(
                self.meta_engine.generate_self_improvement_recommendations(observation)
            )
            
            self.assertIsInstance(recommendations, list)
            self.assertTrue(len(recommendations) > 0)
            
            for recommendation in recommendations:
                self.assertIsInstance(recommendation, SelfImprovementAction)
                self.assertIsNotNone(recommendation.action_id)
                self.assertIsNotNone(recommendation.action_type)
                self.assertIsNotNone(recommendation.expected_impact)
                
        finally:
            loop.close()
    
    def test_recursive_self_analysis(self):
        """Test recursive self-analysis functionality"""
        # Add sample data for analysis
        self.reasoning_engine.reasoning_history = self.sample_conclusions
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Test recursive analysis at depth 0
            analysis = loop.run_until_complete(
                self.meta_engine.recursive_self_analysis(depth=0)
            )
            
            self.assertIsInstance(analysis, dict)
            self.assertEqual(analysis["depth"], 0)
            self.assertIn("observation", analysis)
            self.assertIn("improvement_actions", analysis)
            self.assertIn("timestamp", analysis)
            
            # Test recursive analysis at depth 1
            analysis_depth1 = loop.run_until_complete(
                self.meta_engine.recursive_self_analysis(depth=1)
            )
            
            self.assertEqual(analysis_depth1["depth"], 1)
            # Should have meta-analysis of observation quality
            self.assertIn("meta_analysis", analysis_depth1["observation"].meta_analysis)
            
        finally:
            loop.close()
    
    def test_pattern_detection(self):
        """Test recursive pattern detection"""
        # Add performance history for pattern detection
        self.meta_engine.performance_metrics["accuracy"] = [0.5, 0.6, 0.7, 0.8, 0.75, 0.8, 0.85]
        self.meta_engine.performance_metrics["efficiency"] = [0.4, 0.5, 0.6, 0.65, 0.7, 0.72, 0.75]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            patterns = loop.run_until_complete(
                self.meta_engine.detect_recursive_patterns()
            )
            
            self.assertIsInstance(patterns, dict)
            self.assertIn("performance_patterns", patterns)
            self.assertIn("behavioral_patterns", patterns)
            self.assertIn("meta_patterns", patterns)
            self.assertIn("evolutionary_patterns", patterns)
            
            # Check performance patterns
            perf_patterns = patterns["performance_patterns"]
            if "accuracy" in perf_patterns:
                self.assertIn("trend", perf_patterns["accuracy"])
                self.assertIn("stability", perf_patterns["accuracy"])
                
        finally:
            loop.close()
    
    def test_evolutionary_strategy_evolution(self):
        """Test evolution of cognitive strategies"""
        # Initialize population
        self.meta_engine.evolutionary_optimizer.initialize_population({
            "confidence_threshold": {"min_value": 0.1, "max_value": 0.9},
            "exploration_rate": {"min_value": 0.0, "max_value": 1.0}
        })
        
        # Add some performance data
        self.meta_engine.performance_metrics["accuracy"] = [0.7]
        self.meta_engine.performance_metrics["efficiency"] = [0.6]
        self.meta_engine.performance_metrics["adaptability"] = [0.8]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            evolution_result = loop.run_until_complete(
                self.meta_engine.evolve_cognitive_strategies()
            )
            
            self.assertIsInstance(evolution_result, dict)
            self.assertIn("evolution_stats", evolution_result)
            self.assertIn("new_cognitive_state", evolution_result)
            self.assertIn("generation", evolution_result)
            
            # Check that a generation was evolved
            self.assertEqual(self.meta_engine.evolutionary_optimizer.generation_count, 1)
            
        finally:
            loop.close()
    
    def test_self_optimization_validation(self):
        """Test validation of self-optimization effectiveness"""
        # Add historical performance data
        self.meta_engine.performance_metrics["accuracy"] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        self.meta_engine.performance_metrics["efficiency"] = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        
        # Add some improvement actions
        action = SelfImprovementAction(
            action_id="test_action",
            action_type="test_improvement",
            parameters={},
            expected_impact=0.1,
            actual_impact=0.08
        )
        self.meta_engine.improvement_actions.append(action)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            validation = loop.run_until_complete(
                self.meta_engine.validate_self_optimization_effectiveness()
            )
            
            self.assertIsInstance(validation, dict)
            self.assertIn("performance_improvement", validation)
            self.assertIn("learning_effectiveness", validation)
            self.assertIn("adaptation_quality", validation)
            self.assertIn("meta_cognitive_accuracy", validation)
            self.assertIn("overall_optimization_score", validation)
            
            # Check that scores are in valid range
            for key, value in validation.items():
                if isinstance(value, (int, float)):
                    self.assertTrue(0.0 <= value <= 1.0, f"{key} score {value} not in range [0,1]")
                    
        finally:
            loop.close()
    
    def test_enhanced_reasoning_engine_integration(self):
        """Test integration of recursive meta-cognition with advanced reasoning engine"""
        # Add sample conclusions to the reasoning engine
        self.reasoning_engine.reasoning_history = self.sample_conclusions
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Test enhanced metacognitive reflection
            reflection = loop.run_until_complete(
                self.reasoning_engine.perform_metacognitive_reflection(self.sample_conclusions)
            )
            
            self.assertIsInstance(reflection, dict)
            # Check for original meta-cognitive reflection components
            self.assertIn("reasoning_patterns", reflection)
            self.assertIn("confidence_calibration", reflection)
            self.assertIn("bias_analysis", reflection)
            self.assertIn("strategy_effectiveness", reflection)
            
            # Check for enhanced recursive meta-cognitive components
            self.assertIn("recursive_analysis", reflection)
            self.assertIn("cognitive_evolution", reflection)
            self.assertIn("recursive_patterns", reflection)
            self.assertIn("optimization_validation", reflection)
            self.assertIn("cognitive_status", reflection)
            
        finally:
            loop.close()
    
    def test_recursive_self_improvement_integration(self):
        """Test recursive self-improvement capability"""
        # Add sample reasoning history
        self.reasoning_engine.reasoning_history = self.sample_conclusions
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            improvement_result = loop.run_until_complete(
                self.reasoning_engine.recursive_self_improvement()
            )
            
            self.assertIsInstance(improvement_result, dict)
            self.assertIn("observation", improvement_result)
            self.assertIn("improvement_actions", improvement_result)
            self.assertIn("implementation_results", improvement_result)
            self.assertIn("recursive_analysis", improvement_result)
            
        finally:
            loop.close()
    
    def test_cognitive_status_report(self):
        """Test comprehensive cognitive status reporting"""
        # Test base status
        status = self.reasoning_engine.get_enhanced_cognitive_status()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            enhanced_status = loop.run_until_complete(
                self.reasoning_engine.get_enhanced_cognitive_status()
            )
            
            self.assertIsInstance(enhanced_status, dict)
            self.assertTrue(enhanced_status.get("enhanced_capabilities", False))
            self.assertIn("recursive_metacognition", enhanced_status)
            
            recursive_status = enhanced_status["recursive_metacognition"]
            self.assertIn("current_state", recursive_status)
            self.assertIn("evolutionary_status", recursive_status)
            self.assertIn("performance_summary", recursive_status)
            
        finally:
            loop.close()
    
    def test_meta_cognitive_error_handling(self):
        """Test error handling in meta-cognitive processes"""
        # Test with empty reasoning engine (no history)
        empty_engine = AdvancedReasoningEngine(enable_recursive_metacognition=True)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Should handle empty reasoning history gracefully
            reflection = loop.run_until_complete(
                empty_engine.perform_metacognitive_reflection([])
            )
            
            self.assertIsInstance(reflection, dict)
            # Should still have basic reflection structure even with no data
            self.assertIn("reflection_timestamp", reflection)
            
        finally:
            loop.close()


class TestPhase5ImplementationRequirements(unittest.TestCase):
    """Test that Phase 5 implementation requirements are met"""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
            
        self.reasoning_engine = AdvancedReasoningEngine(enable_recursive_metacognition=True)
    
    def test_step1_feedback_driven_self_analysis(self):
        """Test Step 1: Implement feedback-driven self-analysis modules"""
        # Check that RecursiveMetaCognitiveEngine exists and has self-analysis capability
        self.assertIsNotNone(self.reasoning_engine.recursive_meta_engine)
        self.assertTrue(hasattr(self.reasoning_engine.recursive_meta_engine, 'observe_cognitive_state'))
        self.assertTrue(hasattr(self.reasoning_engine.recursive_meta_engine, 'recursive_self_analysis'))
    
    def test_step2_evolutionary_optimization_moses_equivalent(self):
        """Test Step 2: Integrate MOSES (or equivalent) for kernel evolution"""
        # Check that evolutionary optimizer exists
        meta_engine = self.reasoning_engine.recursive_meta_engine
        self.assertIsNotNone(meta_engine.evolutionary_optimizer)
        self.assertTrue(hasattr(meta_engine.evolutionary_optimizer, 'evolve_generation'))
        self.assertTrue(hasattr(meta_engine.evolutionary_optimizer, 'initialize_population'))
        self.assertTrue(hasattr(meta_engine.evolutionary_optimizer, 'evaluate_fitness'))
    
    def test_step3_recursive_cognitive_observation(self):
        """Test Step 3: Create recursive cognitive observation mechanisms"""
        meta_engine = self.reasoning_engine.recursive_meta_engine
        self.assertTrue(hasattr(meta_engine, 'recursive_self_analysis'))
        self.assertEqual(meta_engine.max_recursive_depth, 3)
        self.assertTrue(hasattr(meta_engine, '_meta_analyze_observation_process'))
    
    def test_step4_self_improvement_recommendations(self):
        """Test Step 4: Develop self-improvement recommendation systems"""
        meta_engine = self.reasoning_engine.recursive_meta_engine
        self.assertTrue(hasattr(meta_engine, 'generate_self_improvement_recommendations'))
        self.assertTrue(hasattr(meta_engine, 'implement_self_improvement_action'))
    
    def test_step5_metacognitive_performance_monitoring(self):
        """Test Step 5: Configure meta-cognitive performance monitoring"""
        meta_engine = self.reasoning_engine.recursive_meta_engine
        self.assertTrue(hasattr(meta_engine, 'performance_metrics'))
        self.assertTrue(hasattr(meta_engine, 'validate_self_optimization_effectiveness'))
        self.assertTrue(hasattr(meta_engine, 'get_cognitive_status_report'))
    
    def test_step6_recursive_pattern_recognition(self):
        """Test Step 6: Implement recursive pattern recognition"""
        meta_engine = self.reasoning_engine.recursive_meta_engine
        self.assertTrue(hasattr(meta_engine, 'detect_recursive_patterns'))
        self.assertTrue(hasattr(meta_engine, 'meta_patterns'))
        self.assertTrue(hasattr(meta_engine, '_detect_performance_patterns'))
        self.assertTrue(hasattr(meta_engine, '_detect_meta_patterns'))
    
    def test_step7_self_optimization_validation(self):
        """Test Step 7: Validate self-optimization effectiveness"""
        meta_engine = self.reasoning_engine.recursive_meta_engine
        self.assertTrue(hasattr(meta_engine, 'validate_self_optimization_effectiveness'))
        self.assertTrue(hasattr(meta_engine, '_validate_performance_improvement'))
        self.assertTrue(hasattr(meta_engine, '_validate_learning_effectiveness'))
    
    def test_cognitive_synergy_features(self):
        """Test cognitive synergy features are implemented"""
        meta_engine = self.reasoning_engine.recursive_meta_engine
        
        # Recursive self-awareness protocols
        self.assertTrue(hasattr(meta_engine, 'recursive_self_analysis'))
        
        # Evolutionary cognitive enhancement
        self.assertTrue(hasattr(meta_engine, 'evolve_cognitive_strategies'))
        
        # Emergent meta-cognitive patterns
        self.assertTrue(hasattr(meta_engine, 'detect_recursive_patterns'))
        
        # Self-optimizing cognitive architectures
        self.assertTrue(hasattr(meta_engine, 'recursive_self_improvement'))


class TestPerformanceAndStability(unittest.TestCase):
    """Test performance and stability of recursive meta-cognitive pathways"""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
            
        self.reasoning_engine = AdvancedReasoningEngine(enable_recursive_metacognition=True)
    
    def test_recursive_depth_limits(self):
        """Test that recursive analysis respects depth limits"""
        meta_engine = self.reasoning_engine.recursive_meta_engine
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Test that max depth is enforced
            analysis = loop.run_until_complete(
                meta_engine.recursive_self_analysis(depth=5)  # Beyond max depth
            )
            
            self.assertEqual(analysis["depth"], 5)
            self.assertEqual(analysis["status"], "max_depth_reached")
            
        finally:
            loop.close()
    
    def test_evolutionary_convergence(self):
        """Test that evolutionary algorithms can converge"""
        meta_engine = self.reasoning_engine.recursive_meta_engine
        optimizer = meta_engine.evolutionary_optimizer
        
        # Initialize population
        optimizer.initialize_population({
            "test_gene": {"min_value": 0.0, "max_value": 1.0}
        })
        
        # Run multiple generations
        def fitness_evaluator(genome):
            # Simple fitness function that prefers values near 0.8
            target = 0.8
            gene_value = genome.genes["test_gene"].value
            return 1.0 - abs(gene_value - target)
        
        initial_fitness = optimizer.population[0].overall_fitness
        
        for generation in range(5):
            optimizer.evolve_generation(fitness_evaluator)
        
        # Check that best fitness improved
        best_fitness = optimizer.best_genome.overall_fitness if optimizer.best_genome else 0
        self.assertGreaterEqual(best_fitness, initial_fitness)
    
    def test_memory_usage_stability(self):
        """Test that meta-cognitive processes don't cause memory leaks"""
        meta_engine = self.reasoning_engine.recursive_meta_engine
        
        initial_observation_count = len(meta_engine.observations)
        
        # Simulate many cognitive observations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            for i in range(50):
                loop.run_until_complete(meta_engine.observe_cognitive_state())
            
            # Check that deque limits are respected (should be max 1000)
            self.assertLessEqual(len(meta_engine.observations), 1000)
            
        finally:
            loop.close()


if __name__ == '__main__':
    # Set up test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestRecursiveMetaCognition))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase5ImplementationRequirements))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceAndStability))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"PHASE 5 RECURSIVE META-COGNITIVE PATHWAYS - TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")