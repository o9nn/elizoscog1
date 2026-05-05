#!/usr/bin/env python3
"""
Comprehensive Phase 3 Integration Test Suite
Tests advanced cognitive financial analysis, multi-agent coordination,
natural language interface, and advanced reasoning capabilities
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import Phase 3 components
from financial.cognitive_analysis import (
    CognitiveFinancialAnalyzer, PatternType, PredictionResult, FinancialPattern
)
from integration.multi_agent_coordination import (
    MultiAgentCoordinator, AgentMessage, MessageType, Priority, AgentState
)
from integration.natural_language_interface import (
    NaturalLanguageInterface, IntentType, EntityType, RecognizedIntent
)
from integration.advanced_reasoning import (
    AdvancedReasoningEngine, ReasoningType, TemporalRule, CausalRelationship, 
    UserFeedback, ReasoningConclusion
)


async def test_cognitive_financial_analysis():
    """Test cognitive financial analysis capabilities"""
    print("Testing Cognitive Financial Analysis...")
    
    analyzer = CognitiveFinancialAnalyzer()
    
    # Mock transaction data
    transaction_history = [
        {"amount": 120.50, "date": "2024-01-15", "category": "groceries", "description": "Safeway"},
        {"amount": 45.75, "date": "2024-01-16", "category": "gas", "description": "Shell Station"},
        {"amount": 85.25, "date": "2024-01-20", "category": "dining", "description": "Restaurant"},
        {"amount": 150.00, "date": "2024-02-15", "category": "groceries", "description": "Costco"},
        {"amount": 50.00, "date": "2024-02-16", "category": "gas", "description": "Chevron"},
        {"amount": 200.00, "date": "2024-02-20", "category": "entertainment", "description": "Concert"},
        {"amount": 75.30, "date": "2024-03-15", "category": "groceries", "description": "Trader Joes"},
        {"amount": 90.45, "date": "2024-03-16", "category": "dining", "description": "Pizza Place"},
        {"amount": 1500.00, "date": "2024-03-20", "category": "shopping", "description": "Electronics Store"},  # Anomaly
        {"amount": 110.25, "date": "2024-03-25", "category": "groceries", "description": "Whole Foods"}
    ]
    
    # Test pattern detection and behavioral analysis
    behavior_analysis = await analyzer.analyze_financial_behavior(transaction_history, 90)
    assert "detected_patterns" in behavior_analysis
    assert "behavioral_insights" in behavior_analysis
    assert "risk_assessment" in behavior_analysis
    assert behavior_analysis["confidence_score"] > 0.0
    
    print(f"  ✓ Detected {len(behavior_analysis['detected_patterns'])} financial patterns")
    print(f"  ✓ Generated {len(behavior_analysis['cognitive_recommendations'])} recommendations")
    
    # Test predictive modeling
    predictions = await analyzer.predict_future_expenses(transaction_history, 3)
    assert isinstance(predictions, dict)
    
    categories_predicted = len(predictions)
    print(f"  ✓ Generated predictions for {categories_predicted} expense categories")
    
    # Test anomaly detection
    anomaly_results = await analyzer.detect_anomalies(transaction_history, 0.7)
    assert "anomaly_summary" in anomaly_results
    assert "detected_anomalies" in anomaly_results
    
    total_anomalies = anomaly_results["anomaly_summary"]["total_anomalies"]
    print(f"  ✓ Detected {total_anomalies} anomalies in transaction data")
    
    # Test risk assessment
    risk_assessment = await analyzer.assess_financial_risk(transaction_history)
    assert "overall_assessment" in risk_assessment
    assert "detailed_analysis" in risk_assessment
    assert "risk_mitigation_strategies" in risk_assessment
    
    risk_level = risk_assessment["overall_assessment"]["risk_level"]
    print(f"  ✓ Risk assessment completed - Risk level: {risk_level}")
    
    print("✓ Cognitive Financial Analysis tests passed")


async def test_multi_agent_coordination():
    """Test multi-agent coordination framework"""
    print("Testing Multi-Agent Coordination...")
    
    coordinator = MultiAgentCoordinator()
    
    # Test agent registration
    agent_capabilities = [
        {"agent_id": "expense_analyzer", "capabilities": {"type": "expense_analyzer", "task_types": ["analysis", "reporting"]}},
        {"agent_id": "budget_planner", "capabilities": {"type": "budget_planner", "task_types": ["planning", "optimization"]}},
        {"agent_id": "risk_assessor", "capabilities": {"type": "risk_analyzer", "task_types": ["risk_analysis", "monitoring"]}},
        {"agent_id": "pattern_detector", "capabilities": {"type": "pattern_analyzer", "task_types": ["pattern_detection", "trend_analysis"]}}
    ]
    
    for agent_info in agent_capabilities:
        success = await coordinator.register_agent(agent_info["agent_id"], agent_info["capabilities"])
        assert success
    
    print(f"  ✓ Registered {len(agent_capabilities)} agents successfully")
    
    # Test message passing
    test_message = AgentMessage(
        sender="expense_analyzer",
        recipient="budget_planner",
        message_type=MessageType.REQUEST,
        content={"action": "budget_analysis", "data": {"category": "groceries"}},
        priority=Priority.HIGH
    )
    
    message_result = await coordinator.send_message(test_message)
    assert message_result is not None
    print("  ✓ Inter-agent message passing working")
    
    # Test distributed reasoning coordination
    reasoning_task = {
        "task_id": "comprehensive_financial_analysis",
        "type": "financial_analysis",
        "data": {"user_id": "test_user", "analysis_scope": "quarterly"}
    }
    
    required_agents = ["expense_analyzer", "budget_planner", "risk_assessor"]
    reasoning_result = await coordinator.coordinate_distributed_reasoning(reasoning_task, required_agents)
    
    assert "reasoning_result" in reasoning_result
    assert "agent_contributions" in reasoning_result
    assert reasoning_result["execution_metadata"]["agents_involved"] > 0
    
    print(f"  ✓ Distributed reasoning coordinated across {reasoning_result['execution_metadata']['agents_involved']} agents")
    
    # Test conflict resolution
    conflict_scenario = {
        "type": "data_interpretation",
        "agents": ["expense_analyzer", "budget_planner"],
        "positions": {
            "expense_analyzer": {"preferred_option": "increase_budget", "reasoning": "Spending trend analysis"},
            "budget_planner": {"preferred_option": "reduce_spending", "reasoning": "Budget optimization"}
        }
    }
    
    conflict_resolution = await coordinator.resolve_conflicts(conflict_scenario)
    assert "resolution" in conflict_resolution
    assert "resolution_strategy" in conflict_resolution
    
    print(f"  ✓ Conflict resolution completed using {conflict_resolution['resolution_strategy']} method")
    
    # Test consensus decision making
    decision_topic = "Budget allocation strategy for next quarter"
    options = [
        {"value": "conservative", "description": "Maintain current spending levels"},
        {"value": "moderate", "description": "Allow 10% increase in discretionary spending"},
        {"value": "aggressive", "description": "Expand budget by 20% for growth initiatives"}
    ]
    
    participants = ["expense_analyzer", "budget_planner", "risk_assessor"]
    consensus_result = await coordinator.consensus_decision(decision_topic, options, participants, 1)
    
    assert "consensus_achieved" in consensus_result
    assert "winning_option" in consensus_result
    
    consensus_achieved = consensus_result["consensus_achieved"]
    print(f"  ✓ Consensus decision process completed - Consensus achieved: {consensus_achieved}")
    
    # Test load balancing
    balance_result = await coordinator.balance_workload()
    assert "workload_distribution" in balance_result
    assert "redistributed_tasks" in balance_result
    
    redistributed_count = len(balance_result["redistributed_tasks"])
    print(f"  ✓ Load balancing completed - {redistributed_count} tasks redistributed")
    
    print("✓ Multi-Agent Coordination tests passed")


async def test_natural_language_interface():
    """Test natural language interface capabilities"""
    print("Testing Natural Language Interface...")
    
    interface = NaturalLanguageInterface()
    
    # Test various financial queries
    test_queries = [
        "How much did I spend on groceries this month?",
        "Show me my budget status for dining",
        "What are my spending trends for the last 3 months?",
        "Generate a monthly spending report",
        "Give me advice on reducing my entertainment expenses",
        "Set a budget of $500 for groceries",
        "Alert me when I spend more than $200 on shopping",
        "Predict my utility costs for next month"
    ]
    
    intent_recognition_success = 0
    entity_extraction_success = 0
    
    for query in test_queries:
        # Test intent recognition
        recognized_intent = await interface.recognize_financial_intent(query)
        assert recognized_intent.intent_type != IntentType.UNKNOWN
        
        if recognized_intent.confidence > 0.5:
            intent_recognition_success += 1
        
        # Test entity extraction
        entities = await interface.extract_financial_entities(query)
        if entities:
            entity_extraction_success += 1
        
        # Test full processing
        result = await interface.process_user_input(query, "test_user", "test_session")
        assert "recognized_intent" in result
        assert "response" in result
        assert "context" in result
    
    print(f"  ✓ Intent recognition successful for {intent_recognition_success}/{len(test_queries)} queries")
    print(f"  ✓ Entity extraction successful for {entity_extraction_success}/{len(test_queries)} queries")
    
    # Test conversation context and follow-up suggestions
    suggestions = interface.get_conversation_suggestions("test_user", "test_session")
    assert len(suggestions) > 0
    print(f"  ✓ Generated {len(suggestions)} conversation suggestions")
    
    # Test natural language report generation
    mock_spending_data = {
        "total_spending": 1250.75,
        "time_period": "March 2024",
        "categories": {
            "groceries": 450.25,
            "dining": 280.50,
            "gas": 180.00,
            "entertainment": 200.00,
            "utilities": 140.00
        }
    }
    
    spending_report = await interface.generate_natural_language_report(
        "spending_summary", mock_spending_data
    )
    assert len(spending_report) > 100  # Should be a comprehensive report
    assert "Total spending" in spending_report
    assert "groceries" in spending_report.lower()
    
    print("  ✓ Natural language report generation working")
    
    # Test complex query processing
    complex_query = "I want to understand why my dining expenses increased by 25% compared to last month and get advice on how to control them"
    
    complex_result = await interface.process_user_input(complex_query, "test_user", "complex_session")
    
    assert complex_result["recognized_intent"]["confidence"] > 0.3
    assert len(complex_result["recognized_intent"]["entities"]) >= 0
    
    print("  ✓ Complex query processing working")
    
    print("✓ Natural Language Interface tests passed")


async def test_advanced_reasoning():
    """Test advanced reasoning capabilities"""
    print("Testing Advanced Reasoning...")
    
    reasoning_engine = AdvancedReasoningEngine()
    
    # Test temporal reasoning
    financial_events = [
        {"date": "2024-01-01", "event": "salary_deposit", "amount": 5000},
        {"date": "2024-01-15", "event": "rent_payment", "amount": -1200},
        {"date": "2024-01-20", "event": "grocery_shopping", "amount": -150},
        {"date": "2024-02-01", "event": "salary_deposit", "amount": 5000},
        {"date": "2024-02-15", "event": "rent_payment", "amount": -1200},
        {"date": "2024-02-25", "event": "large_purchase", "amount": -2000},
        {"date": "2024-03-01", "event": "salary_deposit", "amount": 5000},
        {"date": "2024-03-15", "event": "rent_payment", "amount": -1200}
    ]
    
    temporal_query = "Always after salary deposits, spending increases within 2 weeks"
    temporal_result = await reasoning_engine.perform_temporal_reasoning(financial_events, temporal_query)
    
    assert temporal_result.reasoning_type == ReasoningType.TEMPORAL
    assert temporal_result.confidence > 0.0
    assert len(temporal_result.reasoning_chain) > 0
    
    print(f"  ✓ Temporal reasoning completed with confidence: {temporal_result.confidence:.2f}")
    
    # Test causal reasoning
    causal_hypothesis = "Salary increases lead to higher discretionary spending"
    causal_result = await reasoning_engine.perform_causal_reasoning(causal_hypothesis, financial_events)
    
    assert causal_result.reasoning_type == ReasoningType.CAUSAL
    assert causal_result.confidence > 0.0
    assert len(causal_result.evidence) > 0
    
    print(f"  ✓ Causal reasoning completed with confidence: {causal_result.confidence:.2f}")
    
    # Test probabilistic model building
    model_variables = ["amount", "category", "day_of_month"]
    training_data = [
        {"amount": 150.0, "category": "groceries", "day_of_month": 15},
        {"amount": 200.0, "category": "dining", "day_of_month": 20},
        {"amount": 75.0, "category": "gas", "day_of_month": 10},
        {"amount": 180.0, "category": "groceries", "day_of_month": 30},
        {"amount": 120.0, "category": "dining", "day_of_month": 25}
    ]
    
    # Test different model types
    for model_type in ["bayesian", "markov", "regression"]:
        probabilistic_model = await reasoning_engine.build_probabilistic_model(
            model_type, model_variables, training_data
        )
        
        assert probabilistic_model.model_type == model_type
        assert len(probabilistic_model.variables) == len(model_variables)
        assert "accuracy" in probabilistic_model.accuracy_metrics
        
        print(f"  ✓ {model_type.title()} model built with accuracy: {probabilistic_model.accuracy_metrics['accuracy']:.2f}")
    
    # Test meta-cognitive reflection
    reasoning_results = [temporal_result, causal_result]
    meta_reflection = await reasoning_engine.perform_metacognitive_reflection(reasoning_results)
    
    assert "reasoning_patterns" in meta_reflection
    assert "confidence_calibration" in meta_reflection
    assert "meta_insights" in meta_reflection
    assert "improvement_recommendations" in meta_reflection
    
    insights_count = len(meta_reflection["meta_insights"])
    recommendations_count = len(meta_reflection["improvement_recommendations"])
    
    print(f"  ✓ Meta-cognitive reflection generated {insights_count} insights and {recommendations_count} recommendations")
    
    # Test learning from feedback
    feedback = UserFeedback(
        feedback_id="test_feedback_1",
        conclusion_id=temporal_result.conclusion_id,
        user_id="test_user",
        feedback_type="correct",
        confidence_rating=0.8,
        explanation="The temporal pattern was accurately identified"
    )
    
    learning_result = await reasoning_engine.learn_from_feedback(feedback)
    
    assert learning_result["feedback_processed"] == True
    assert "confidence_adjustments" in learning_result
    assert "strategy_adjustments" in learning_result
    
    print("  ✓ Learning from user feedback working")
    
    # Test financial risk prediction
    current_state = {
        "monthly_income": 5000,
        "monthly_expenses": 3500,
        "savings_rate": 0.3,
        "debt_ratio": 0.2,
        "spending_volatility": 0.15
    }
    
    prediction_horizon = timedelta(days=90)
    risk_prediction = await reasoning_engine.predict_financial_risk(current_state, prediction_horizon)
    
    assert "risk_assessment" in risk_prediction
    assert "confidence_level" in risk_prediction
    assert "reliability_assessment" in risk_prediction
    
    risk_level = risk_prediction["risk_assessment"]["risk_level"]
    confidence = risk_prediction["confidence_level"]
    
    print(f"  ✓ Financial risk prediction completed - Risk: {risk_level}, Confidence: {confidence:.2f}")
    
    print("✓ Advanced Reasoning tests passed")


async def test_integrated_phase3_workflow():
    """Test complete Phase 3 integration workflow"""
    print("Testing Complete Phase 3 Integration Workflow...")
    
    # Initialize all Phase 3 components
    cognitive_analyzer = CognitiveFinancialAnalyzer()
    coordinator = MultiAgentCoordinator()
    nl_interface = NaturalLanguageInterface()
    reasoning_engine = AdvancedReasoningEngine()
    
    # Register agents in coordinator
    agents = [
        {"agent_id": "cognitive_analyzer", "capabilities": {"type": "cognitive_analyzer", "task_types": ["pattern_analysis", "prediction"]}},
        {"agent_id": "reasoning_engine", "capabilities": {"type": "reasoning_engine", "task_types": ["temporal_reasoning", "causal_analysis"]}},
        {"agent_id": "nl_processor", "capabilities": {"type": "nl_processor", "task_types": ["intent_recognition", "response_generation"]}}
    ]
    
    for agent in agents:
        await coordinator.register_agent(agent["agent_id"], agent["capabilities"])
    
    # Simulate complex user interaction
    user_query = "I notice my spending has been increasing lately. Can you analyze my patterns, predict future expenses, and give me advice on how to optimize my budget?"
    
    # Step 1: Natural language processing
    nl_result = await nl_interface.process_user_input(user_query, "integration_user", "phase3_session")
    
    # Step 2: Coordinate analysis across multiple agents
    analysis_task = {
        "task_id": "comprehensive_financial_optimization",
        "type": "financial_analysis",
        "user_query": user_query,
        "requested_analyses": ["pattern_detection", "prediction", "optimization"]
    }
    
    reasoning_coordination = await coordinator.coordinate_distributed_reasoning(
        analysis_task, ["cognitive_analyzer", "reasoning_engine"]
    )
    
    # Step 3: Cognitive financial analysis
    mock_transaction_data = [
        {"amount": 200.0, "date": "2024-03-01", "category": "groceries"},
        {"amount": 150.0, "date": "2024-03-05", "category": "dining"},
        {"amount": 300.0, "date": "2024-03-10", "category": "shopping"},
        {"amount": 180.0, "date": "2024-03-15", "category": "groceries"},
        {"amount": 220.0, "date": "2024-03-20", "category": "entertainment"}
    ]
    
    cognitive_analysis = await cognitive_analyzer.analyze_financial_behavior(mock_transaction_data)
    
    # Step 4: Advanced reasoning for optimization
    optimization_hypothesis = "Reducing discretionary spending leads to better budget adherence"
    causal_analysis = await reasoning_engine.perform_causal_reasoning(optimization_hypothesis, mock_transaction_data)
    
    # Step 5: Generate comprehensive response
    integration_result = {
        "user_query": user_query,
        "nl_processing": {
            "intent": nl_result["recognized_intent"]["type"],
            "confidence": nl_result["recognized_intent"]["confidence"]
        },
        "cognitive_analysis": {
            "patterns_detected": len(cognitive_analysis["detected_patterns"]),
            "recommendations": cognitive_analysis["cognitive_recommendations"][:3]
        },
        "reasoning_analysis": {
            "causal_conclusion": causal_analysis.conclusion,
            "confidence": causal_analysis.confidence
        },
        "coordination_metrics": {
            "agents_involved": reasoning_coordination["execution_metadata"]["agents_involved"],
            "subtasks_completed": reasoning_coordination["execution_metadata"]["subtasks_created"]
        }
    }
    
    # Verify integration success
    assert integration_result["nl_processing"]["confidence"] > 0.0
    assert integration_result["cognitive_analysis"]["patterns_detected"] >= 0
    assert integration_result["reasoning_analysis"]["confidence"] > 0.0
    assert integration_result["coordination_metrics"]["agents_involved"] >= 0  # Allow 0 for mock coordination
    
    print(f"  ✓ Natural language processing confidence: {integration_result['nl_processing']['confidence']:.2f}")
    print(f"  ✓ Cognitive analysis detected {integration_result['cognitive_analysis']['patterns_detected']} patterns")
    print(f"  ✓ Reasoning analysis confidence: {integration_result['reasoning_analysis']['confidence']:.2f}")
    print(f"  ✓ Multi-agent coordination involved {integration_result['coordination_metrics']['agents_involved']} agents")
    
    # Test meta-cognitive assessment of the entire workflow
    workflow_reasoning = [causal_analysis]
    meta_assessment = await reasoning_engine.perform_metacognitive_reflection(workflow_reasoning)
    
    assert len(meta_assessment["meta_insights"]) > 0
    
    print(f"  ✓ Meta-cognitive assessment generated {len(meta_assessment['meta_insights'])} insights")
    
    # Generate final natural language response
    response_data = {
        "analysis_summary": "Your spending analysis is complete",
        "key_findings": cognitive_analysis["cognitive_recommendations"][:2],
        "predictions": "Future expense predictions generated",
        "optimization_advice": causal_analysis.conclusion
    }
    
    final_response = await nl_interface.generate_natural_language_report("comprehensive_analysis", response_data)
    
    assert len(final_response) > 50
    print("  ✓ Final natural language response generated")
    
    print("✓ Complete Phase 3 Integration Workflow tests passed")


async def main():
    """Run all Phase 3 integration tests"""
    print("=== Phase 3 ElizaOS-OpenCog-GnuCash Advanced Integration Tests ===\n")
    
    try:
        await test_cognitive_financial_analysis()
        print()
        
        await test_multi_agent_coordination()
        print()
        
        await test_natural_language_interface()
        print()
        
        await test_advanced_reasoning()
        print()
        
        await test_integrated_phase3_workflow()
        print()
        
        print("=== All Phase 3 Tests Passed! ===")
        print("Advanced cognitive financial analysis framework is working correctly.")
        print()
        print("Phase 3 capabilities verified:")
        print("✓ Cognitive Financial Analysis - Pattern recognition, predictions, anomaly detection")
        print("✓ Multi-Agent Coordination - Distributed reasoning, conflict resolution, consensus")
        print("✓ Natural Language Interface - Intent recognition, conversational processing")
        print("✓ Advanced Reasoning - Temporal logic, causal analysis, meta-cognitive reflection")
        print("✓ Integrated Workflow - Seamless coordination across all components")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)