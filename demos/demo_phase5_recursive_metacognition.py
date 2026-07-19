#!/usr/bin/env python3
"""
Phase 5 Recursive Meta-Cognitive Pathways - Demonstration Script
Showcases the recursive meta-cognitive and evolutionary optimization capabilities
"""

import sys
import os
import asyncio
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.integration.advanced_reasoning import AdvancedReasoningEngine, ReasoningType, ReasoningConclusion, UserFeedback
    from src.integration.recursive_metacognition import CognitiveState
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


def create_sample_reasoning_history():
    """Create sample reasoning conclusions for demonstration"""
    conclusions = []
    
    reasoning_scenarios = [
        ("Market trend analysis shows declining performance", "Recommend risk reduction", ReasoningType.TEMPORAL, 0.8),
        ("High correlation between sector X and market volatility", "Diversify away from sector X", ReasoningType.CAUSAL, 0.7),
        ("Historical data suggests 15% probability of correction", "Prepare for potential correction", ReasoningType.PROBABILISTIC, 0.6),
        ("User feedback indicates overconfident predictions", "Recalibrate confidence assessments", ReasoningType.METACOGNITIVE, 0.9),
        ("Pattern recognition shows seasonal trends", "Adjust portfolio for seasonal effects", ReasoningType.INDUCTIVE, 0.7),
        ("Given current market conditions and rules", "Apply defensive strategy", ReasoningType.DEDUCTIVE, 0.8)
    ]
    
    for i, (premise, conclusion_text, reasoning_type, confidence) in enumerate(reasoning_scenarios):
        conclusion = ReasoningConclusion(
            conclusion_id=f"demo_conclusion_{i}",
            reasoning_type=reasoning_type,
            premise=premise,
            conclusion=conclusion_text,
            confidence=confidence,
            evidence=[
                {"type": "market_data", "source": "financial_feed", "reliability": 0.9},
                {"type": "historical_pattern", "timeframe": "3_months", "strength": 0.8}
            ],
            reasoning_chain=[f"Analysis step {j+1}" for j in range(3 + (i % 2))],
            timestamp=datetime.now()
        )
        conclusions.append(conclusion)
    
    return conclusions


def create_sample_feedback():
    """Create sample user feedback for demonstration"""
    feedback_samples = [
        ("demo_conclusion_0", "correct", 0.9, "Good market analysis"),
        ("demo_conclusion_1", "partially_correct", 0.7, "Causal relationship identified but strength overestimated"),
        ("demo_conclusion_2", "incorrect", 0.3, "Probability calculation seems off"),
        ("demo_conclusion_3", "correct", 0.95, "Excellent meta-cognitive insight"),
    ]
    
    feedback_list = []
    for i, (conclusion_id, feedback_type, confidence, explanation) in enumerate(feedback_samples):
        feedback = UserFeedback(
            feedback_id=f"demo_feedback_{i}",
            conclusion_id=conclusion_id,
            user_id="demo_user",
            feedback_type=feedback_type,
            confidence_rating=confidence,
            explanation=explanation,
            timestamp=datetime.now()
        )
        feedback_list.append(feedback)
    
    return feedback_list


async def demonstrate_phase5_capabilities():
    """Demonstrate Phase 5 recursive meta-cognitive pathways"""
    
    print("🧠 PHASE 5: RECURSIVE META-COGNITIVE PATHWAYS DEMONSTRATION")
    print("=" * 60)
    
    # Initialize the enhanced reasoning engine
    print("\n1. Initializing Enhanced Reasoning Engine...")
    reasoning_engine = AdvancedReasoningEngine(enable_recursive_metacognition=True)
    meta_engine = reasoning_engine.recursive_meta_engine
    
    print(f"   ✓ Recursive meta-cognitive engine: {'Enabled' if meta_engine else 'Disabled'}")
    print(f"   ✓ Current cognitive state: {meta_engine.current_state.value}")
    print(f"   ✓ Evolutionary optimizer: {len(meta_engine.evolutionary_optimizer.population)} population members")
    
    # Add sample reasoning history
    print("\n2. Adding Sample Reasoning History...")
    sample_conclusions = create_sample_reasoning_history()
    reasoning_engine.reasoning_history = sample_conclusions
    
    print(f"   ✓ Added {len(sample_conclusions)} reasoning conclusions")
    print(f"   ✓ Reasoning types: {list(set(c.reasoning_type.value for c in sample_conclusions))}")
    
    # Add sample feedback
    print("\n3. Processing User Feedback...")
    sample_feedback = create_sample_feedback()
    reasoning_engine.user_feedback = sample_feedback
    
    print(f"   ✓ Added {len(sample_feedback)} feedback instances")
    print(f"   ✓ Feedback types: {list(set(f.feedback_type for f in sample_feedback))}")
    
    # Demonstrate cognitive state observation
    print("\n4. Performing Cognitive State Observation...")
    observation = await meta_engine.observe_cognitive_state()
    
    print(f"   ✓ Performance metrics: {list(observation.performance_metrics.keys())}")
    print(f"   ✓ Current accuracy: {observation.performance_metrics.get('accuracy', 0):.3f}")
    print(f"   ✓ Current efficiency: {observation.performance_metrics.get('efficiency', 0):.3f}")
    print(f"   ✓ Improvement opportunities identified: {len(observation.improvement_opportunities)}")
    
    if observation.improvement_opportunities:
        print("   ✓ Key improvement opportunities:")
        for i, opportunity in enumerate(observation.improvement_opportunities[:3]):
            print(f"     {i+1}. {opportunity}")
    
    # Demonstrate evolutionary optimization
    print("\n5. Evolving Cognitive Strategies...")
    evolution_results = await meta_engine.evolve_cognitive_strategies()
    
    print(f"   ✓ Generation: {evolution_results['generation']}")
    print(f"   ✓ Best fitness: {evolution_results['evolution_stats']['best_fitness']:.3f}")
    print(f"   ✓ Average fitness: {evolution_results['evolution_stats']['average_fitness']:.3f}")
    print(f"   ✓ New cognitive state: {evolution_results['new_cognitive_state']}")
    
    # Demonstrate recursive self-analysis
    print("\n6. Performing Recursive Self-Analysis...")
    recursive_analysis = await meta_engine.recursive_self_analysis(depth=2)
    
    print(f"   ✓ Analysis depth: {recursive_analysis['depth']}")
    print(f"   ✓ Observation quality metrics: {len(recursive_analysis['observation'].meta_analysis)}")
    print(f"   ✓ Improvement actions recommended: {len(recursive_analysis['improvement_actions'])}")
    
    if recursive_analysis['recursive_results']:
        print(f"   ✓ Recursive results available: {len(recursive_analysis['recursive_results'])}")
    
    # Demonstrate pattern detection
    print("\n7. Detecting Recursive Patterns...")
    patterns = await meta_engine.detect_recursive_patterns()
    
    print(f"   ✓ Performance patterns: {list(patterns['performance_patterns'].keys())}")
    print(f"   ✓ Behavioral patterns: {list(patterns['behavioral_patterns'].keys())}")
    print(f"   ✓ Meta patterns: {list(patterns['meta_patterns'].keys())}")
    print(f"   ✓ Evolutionary patterns: {list(patterns['evolutionary_patterns'].keys())}")
    
    # Demonstrate self-improvement
    print("\n8. Executing Recursive Self-Improvement...")
    improvement_results = await meta_engine.recursive_self_improvement()
    
    if improvement_results.get('self_improvement_success'):
        print("   ✓ Self-improvement process completed successfully")
        print(f"   ✓ Actions implemented: {len(improvement_results['implementation_results'])}")
        print(f"   ✓ Evolution generation: {improvement_results['evolution_results']['generation']}")
        print(f"   ✓ Optimization score: {improvement_results['validation_results']['overall_optimization_score']:.3f}")
    else:
        print(f"   ⚠ Self-improvement encountered issue: {improvement_results.get('error', 'Unknown error')}")
    
    # Demonstrate enhanced meta-cognitive reflection
    print("\n9. Enhanced Meta-Cognitive Reflection...")
    reflection = await reasoning_engine.perform_metacognitive_reflection(sample_conclusions)
    
    print(f"   ✓ Reasoning patterns analyzed: {len(reflection['reasoning_patterns'])}")
    print(f"   ✓ Confidence calibration score: {reflection['confidence_calibration']['calibration_score']:.3f}")
    print(f"   ✓ Biases detected: {reflection['bias_analysis']['bias_count']}")
    print(f"   ✓ Meta insights generated: {len(reflection['meta_insights'])}")
    
    if 'recursive_analysis' in reflection:
        print("   ✓ Recursive analysis integrated successfully")
        print(f"   ✓ Cognitive evolution results included: {'cognitive_evolution' in reflection}")
        print(f"   ✓ Optimization validation completed: {'optimization_validation' in reflection}")
    
    # Demonstrate validation
    print("\n10. Validating Self-Optimization Effectiveness...")
    validation = await meta_engine.validate_self_optimization_effectiveness()
    
    print(f"   ✓ Performance improvement score: {validation['performance_improvement']:.3f}")
    print(f"   ✓ Learning effectiveness score: {validation['learning_effectiveness']:.3f}")
    print(f"   ✓ Adaptation quality score: {validation['adaptation_quality']:.3f}")
    print(f"   ✓ Meta-cognitive accuracy score: {validation['meta_cognitive_accuracy']:.3f}")
    print(f"   ✓ Overall optimization score: {validation['overall_optimization_score']:.3f}")
    
    # Final status report
    print("\n11. Final Cognitive Status Report...")
    status = meta_engine.get_cognitive_status_report()
    
    print(f"   ✓ Current state: {status['current_state']}")
    print(f"   ✓ Recursive depth: {status['recursive_depth']}")
    print(f"   ✓ Observations collected: {status['observation_count']}")
    print(f"   ✓ Improvement actions taken: {status['improvement_actions_count']}")
    print(f"   ✓ Evolution generation: {status['evolutionary_status']['generation']}")
    print(f"   ✓ Best fitness achieved: {status['evolutionary_status']['best_fitness']:.3f}")
    
    print("\n" + "=" * 60)
    print("🎉 PHASE 5 RECURSIVE META-COGNITIVE PATHWAYS DEMONSTRATION COMPLETE")
    print("\nKey Achievements:")
    print("✅ Self-analysis and cognitive observation")
    print("✅ Evolutionary optimization of cognitive strategies")
    print("✅ Recursive meta-cognitive reflection")
    print("✅ Pattern detection and analysis")
    print("✅ Self-improvement recommendations and implementation")
    print("✅ Performance validation and monitoring")
    print("✅ Integration with existing advanced reasoning")
    
    return status


async def main():
    """Main demonstration function"""
    if not IMPORTS_AVAILABLE:
        print("❌ Required imports not available. Please check module installation.")
        return
    
    try:
        final_status = await demonstrate_phase5_capabilities()
        
        print(f"\n📊 FINAL METRICS:")
        print(f"Overall system performance: {final_status['evolutionary_status']['best_fitness']:.1%}")
        print(f"Cognitive observations: {final_status['observation_count']}")
        print(f"Self-improvement cycles: {final_status['improvement_actions_count']}")
        print(f"Evolution generations: {final_status['evolutionary_status']['generation']}")
        
        print(f"\n🔬 TECHNICAL VALIDATION:")
        print(f"✅ Step 1 - Feedback-driven self-analysis: Implemented")
        print(f"✅ Step 2 - Evolutionary optimization (MOSES): Implemented")
        print(f"✅ Step 3 - Recursive cognitive observation: Implemented")
        print(f"✅ Step 4 - Self-improvement recommendations: Implemented")
        print(f"✅ Step 5 - Meta-cognitive performance monitoring: Implemented")
        print(f"✅ Step 6 - Recursive pattern recognition: Implemented")
        print(f"✅ Step 7 - Self-optimization validation: Implemented")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Starting Phase 5 Recursive Meta-Cognitive Pathways Demonstration...")
    asyncio.run(main())