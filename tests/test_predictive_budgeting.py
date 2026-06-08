"""
Unit Tests for Predictive Budgeting and Investment Advisor Modules
Tests the Phase 6 implementation features.
"""

import pytest
import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.financial.predictive_budgeting import (
    PredictiveBudgetingEngine,
    BudgetPrediction,
    CashFlowPrediction,
    BudgetRecommendation,
    PredictionModel
)

from src.financial.investment_advisor import (
    InvestmentAdviceCoordinator,
    InvestmentGoal,
    RiskProfile,
    AssetClass,
    GoalPriority
)


# Helper to run async tests
def run_async(coro):
    """Run async coroutine in test context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================================
# Test Data Fixtures
# ============================================================================

def generate_test_transactions(months: int = 12) -> List[Dict[str, Any]]:
    """Generate test transaction data."""
    transactions = []
    categories = ['groceries', 'utilities', 'dining', 'transportation', 'entertainment']
    base_amounts = {'groceries': 400, 'utilities': 150, 'dining': 200, 
                   'transportation': 100, 'entertainment': 80}
    
    base_date = datetime.now() - timedelta(days=months * 30)
    
    for month_offset in range(months):
        month_date = base_date + timedelta(days=month_offset * 30)
        
        for category in categories:
            # Add some variance
            base = base_amounts[category]
            variance = base * 0.2  # 20% variance
            amount = base + random.uniform(-variance, variance)
            
            transactions.append({
                'date': month_date.isoformat(),
                'amount': amount,
                'category': category,
                'description': f'{category.title()} expense'
            })
    
    return transactions


def generate_test_income(months: int = 12) -> List[Dict[str, Any]]:
    """Generate test income data."""
    income = []
    base_date = datetime.now() - timedelta(days=months * 30)
    
    for month_offset in range(months):
        month_date = base_date + timedelta(days=month_offset * 30)
        income.append({
            'date': month_date.isoformat(),
            'amount': 5000,
            'category': 'income',
            'description': 'Salary'
        })
    
    return income


# ============================================================================
# Predictive Budgeting Tests
# ============================================================================

class TestPredictiveBudgetingEngine:
    """Test suite for PredictiveBudgetingEngine."""
    
    def test_engine_initialization(self):
        """Test engine initializes with default config."""
        engine = PredictiveBudgetingEngine()
        assert engine is not None
        assert engine.smoothing_alpha == 0.3
        assert engine.min_data_points == 3
    
    def test_engine_custom_config(self):
        """Test engine with custom configuration."""
        config = {
            'smoothing_alpha': 0.5,
            'confidence_level': 0.90,
            'min_data_points': 5
        }
        engine = PredictiveBudgetingEngine(config)
        assert engine.smoothing_alpha == 0.5
        assert engine.min_data_points == 5
    
    def test_train_on_historical_data(self):
        """Test training on historical transaction data."""
        engine = PredictiveBudgetingEngine()
        transactions = generate_test_transactions(12)
        
        result = run_async(engine.train_on_historical_data(transactions, 12))
        
        assert result['status'] == 'success'
        assert result['months_analyzed'] > 0
        assert result['categories_found'] > 0
        assert 'data_quality' in result
    
    def test_train_with_minimal_data(self):
        """Test training with minimal data."""
        engine = PredictiveBudgetingEngine()
        transactions = generate_test_transactions(2)  # Only 2 months
        
        result = run_async(engine.train_on_historical_data(transactions, 2))
        
        assert result['status'] == 'success'
        assert result['data_quality']['recommendation'] == 'collect_more_data'
    
    def test_predict_category_spending(self):
        """Test category spending prediction."""
        engine = PredictiveBudgetingEngine()
        transactions = generate_test_transactions(12)
        run_async(engine.train_on_historical_data(transactions, 12))
        
        predictions = run_async(engine.predict_category_spending('groceries', 3))
        
        assert len(predictions) == 3
        for pred in predictions:
            assert isinstance(pred, BudgetPrediction)
            assert pred.category == 'groceries'
            assert pred.predicted_amount > 0
            assert pred.lower_bound <= pred.predicted_amount
            assert pred.upper_bound >= pred.predicted_amount
            assert 0 <= pred.confidence <= 1
    
    def test_predict_all_categories(self):
        """Test prediction for all categories."""
        engine = PredictiveBudgetingEngine()
        transactions = generate_test_transactions(12)
        run_async(engine.train_on_historical_data(transactions, 12))
        
        all_predictions = run_async(engine.predict_all_categories(3))
        
        assert len(all_predictions) > 0
        for category, predictions in all_predictions.items():
            assert len(predictions) > 0
            assert all(isinstance(p, BudgetPrediction) for p in predictions)
    
    def test_predict_cash_flow(self):
        """Test cash flow projection."""
        engine = PredictiveBudgetingEngine()
        transactions = generate_test_transactions(12)
        income = generate_test_income(12)
        
        run_async(engine.train_on_historical_data(transactions, 12))
        expense_predictions = run_async(engine.predict_all_categories(3))
        
        projections = run_async(engine.predict_cash_flow(
            income_history=income,
            expense_predictions=expense_predictions,
            current_balance=1000,
            months_ahead=3
        ))
        
        assert len(projections) == 3
        for proj in projections:
            assert isinstance(proj, CashFlowPrediction)
            assert proj.predicted_income > 0
            assert proj.predicted_expenses >= 0
    
    def test_generate_budget_recommendations(self):
        """Test budget recommendation generation."""
        engine = PredictiveBudgetingEngine()
        transactions = generate_test_transactions(12)
        run_async(engine.train_on_historical_data(transactions, 12))
        
        predictions = run_async(engine.predict_all_categories(3))
        
        current_budget = {
            'groceries': 350,  # Under budget
            'dining': 300,    # Over budget
            'utilities': 150
        }
        
        recommendations = run_async(engine.generate_budget_recommendations(
            current_budget=current_budget,
            predictions=predictions,
            income_target=5000
        ))
        
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert isinstance(rec, BudgetRecommendation)
            assert rec.priority >= 1
            assert rec.reason != ''
    
    def test_predict_goal_progress(self):
        """Test financial goal progress prediction."""
        engine = PredictiveBudgetingEngine()
        
        result = run_async(engine.predict_goal_progress(
            goal_name="Emergency Fund",
            target_amount=10000,
            current_savings=3000,
            monthly_contribution=500,
            target_date=datetime.now() + timedelta(days=365),
            expected_return=0.05
        ))
        
        assert 'goal_name' in result
        assert 'projected_value' in result
        assert 'on_track' in result
        assert 'scenarios' in result
        assert 'optimistic' in result['scenarios']
        assert 'recommendations' in result
    
    def test_trend_determination(self):
        """Test trend determination logic."""
        engine = PredictiveBudgetingEngine()
        
        # Increasing trend
        increasing = [100, 120, 140, 160, 180]
        assert engine._determine_trend(increasing) == 'increasing'
        
        # Decreasing trend
        decreasing = [180, 160, 140, 120, 100]
        assert engine._determine_trend(decreasing) == 'decreasing'
        
        # Stable trend
        stable = [100, 102, 99, 101, 100]
        assert engine._determine_trend(stable) == 'stable'
    
    def test_exponential_smoothing(self):
        """Test exponential smoothing calculation."""
        engine = PredictiveBudgetingEngine({'smoothing_alpha': 0.3})
        
        history = [100, 110, 105, 115, 108]
        prediction = engine._exponential_smoothing(history)
        
        assert prediction > 0
        # Should be close to recent values
        assert abs(prediction - 108) < 20


# ============================================================================
# Investment Advice Coordinator Tests
# ============================================================================

class TestInvestmentAdviceCoordinator:
    """Test suite for InvestmentAdviceCoordinator."""
    
    def test_coordinator_initialization(self):
        """Test coordinator initializes correctly."""
        coordinator = InvestmentAdviceCoordinator()
        assert coordinator is not None
        assert coordinator.risk_free_rate == 0.03
        assert coordinator.rebalancing_threshold == 0.05
    
    def test_register_goal(self):
        """Test goal registration."""
        coordinator = InvestmentAdviceCoordinator()
        
        goal = InvestmentGoal(
            goal_id="retirement_001",
            name="Retirement Fund",
            target_amount=1000000,
            current_amount=50000,
            target_date=datetime.now() + timedelta(days=365 * 25),
            priority=GoalPriority.HIGH,
            risk_tolerance=RiskProfile.MODERATE,
            monthly_contribution=1000
        )
        
        goal_id = coordinator.register_goal(goal)
        
        assert goal_id == "retirement_001"
        assert len(coordinator.goals) == 1
    
    def test_goal_progress_calculation(self):
        """Test goal progress percentage calculation."""
        goal = InvestmentGoal(
            goal_id="test_001",
            name="Test Goal",
            target_amount=10000,
            current_amount=2500,
            target_date=datetime.now() + timedelta(days=365),
            priority=GoalPriority.MEDIUM,
            risk_tolerance=RiskProfile.MODERATE
        )
        
        assert goal.progress_percentage == 25.0
    
    def test_generate_portfolio_recommendation(self):
        """Test portfolio recommendation generation."""
        coordinator = InvestmentAdviceCoordinator()
        
        recommendation = run_async(coordinator.generate_portfolio_recommendation(
            risk_profile=RiskProfile.MODERATE,
            investment_horizon_years=10,
            total_investable=100000
        ))
        
        assert recommendation is not None
        assert len(recommendation.allocations) > 0
        assert recommendation.expected_annual_return > 0
        assert recommendation.expected_volatility > 0
        assert recommendation.sharpe_ratio > 0
        assert recommendation.confidence > 0
        
        # Verify allocations sum to 100%
        total_allocation = sum(a.percentage for a in recommendation.allocations)
        assert abs(total_allocation - 100) < 1  # Allow small rounding errors
    
    def test_portfolio_risk_profiles(self):
        """Test different risk profile allocations."""
        coordinator = InvestmentAdviceCoordinator()
        
        for risk_profile in RiskProfile:
            recommendation = run_async(coordinator.generate_portfolio_recommendation(
                risk_profile=risk_profile,
                investment_horizon_years=10,
                total_investable=100000
            ))
            
            assert recommendation is not None
            assert len(recommendation.allocations) > 0
            
            # More aggressive profiles should have higher expected returns
            # and higher volatility
    
    def test_set_current_portfolio(self):
        """Test setting current portfolio."""
        coordinator = InvestmentAdviceCoordinator()
        
        portfolio = {
            'us_stocks': 50000,
            'bonds': 30000,
            'cash': 20000
        }
        
        coordinator.set_current_portfolio(portfolio)
        
        assert len(coordinator.current_portfolio) == 3
        assert abs(sum(coordinator.current_portfolio.values()) - 1.0) < 0.01
    
    def test_check_rebalancing_needs(self):
        """Test rebalancing needs check."""
        coordinator = InvestmentAdviceCoordinator()
        
        # Set current portfolio
        coordinator.set_current_portfolio({
            'us_stocks': 70000,  # 70%
            'bonds': 10000,      # 10%
            'cash': 20000        # 20%
        })
        
        # Target allocation
        target = {
            AssetClass.US_STOCKS: 0.50,
            AssetClass.BONDS: 0.30,
            AssetClass.CASH: 0.20
        }
        
        actions = run_async(coordinator.check_rebalancing_needs(target, 100000))
        
        # Should have rebalancing actions
        assert len(actions) > 0
        
        # US stocks should be sold (overweight)
        us_action = next((a for a in actions if a.asset_class == AssetClass.US_STOCKS), None)
        if us_action:
            assert us_action.action == 'sell'
    
    def test_coordinate_goal_strategies(self):
        """Test coordinated goal strategy generation."""
        coordinator = InvestmentAdviceCoordinator()
        
        # Register multiple goals
        coordinator.register_goal(InvestmentGoal(
            goal_id="emergency_fund",
            name="Emergency Fund",
            target_amount=15000,
            current_amount=5000,
            target_date=datetime.now() + timedelta(days=365),
            priority=GoalPriority.CRITICAL,
            risk_tolerance=RiskProfile.CONSERVATIVE
        ))
        
        coordinator.register_goal(InvestmentGoal(
            goal_id="retirement",
            name="Retirement",
            target_amount=500000,
            current_amount=50000,
            target_date=datetime.now() + timedelta(days=365 * 20),
            priority=GoalPriority.HIGH,
            risk_tolerance=RiskProfile.MODERATE
        ))
        
        strategies = run_async(coordinator.coordinate_goal_strategies(2000))
        
        assert len(strategies) == 2
        for strategy in strategies:
            assert strategy.goal_id in ['emergency_fund', 'retirement']
            assert strategy.recommended_monthly_contribution >= 0
            assert strategy.portfolio_recommendation is not None
    
    def test_generate_investment_summary(self):
        """Test investment summary generation."""
        coordinator = InvestmentAdviceCoordinator()
        
        # Without goals
        empty_summary = run_async(coordinator.generate_investment_summary())
        assert empty_summary['status'] == 'no_goals'
        
        # With goals
        coordinator.register_goal(InvestmentGoal(
            goal_id="test",
            name="Test Goal",
            target_amount=10000,
            current_amount=2500,
            target_date=datetime.now() + timedelta(days=365),
            priority=GoalPriority.MEDIUM,
            risk_tolerance=RiskProfile.MODERATE
        ))
        
        summary = run_async(coordinator.generate_investment_summary())
        assert summary['status'] == 'active'
        assert 'summary' in summary
        assert summary['summary']['total_goals'] == 1
    
    def test_horizon_adjustment(self):
        """Test portfolio adjustment based on investment horizon."""
        coordinator = InvestmentAdviceCoordinator()
        
        # Short horizon should be more conservative
        short_horizon = run_async(coordinator.generate_portfolio_recommendation(
            risk_profile=RiskProfile.AGGRESSIVE,
            investment_horizon_years=2,
            total_investable=100000
        ))
        
        # Long horizon
        long_horizon = run_async(coordinator.generate_portfolio_recommendation(
            risk_profile=RiskProfile.AGGRESSIVE,
            investment_horizon_years=20,
            total_investable=100000
        ))
        
        # Short horizon should have more bonds/cash
        short_bonds = sum(a.percentage for a in short_horizon.allocations 
                         if a.asset_class in [AssetClass.BONDS, AssetClass.CASH])
        long_bonds = sum(a.percentage for a in long_horizon.allocations 
                        if a.asset_class in [AssetClass.BONDS, AssetClass.CASH])
        
        assert short_bonds >= long_bonds  # More conservative for shorter horizon


# ============================================================================
# Integration Tests
# ============================================================================

class TestFinancialModuleIntegration:
    """Integration tests for financial modules working together."""
    
    def test_budgeting_to_investment_workflow(self):
        """Test workflow from budgeting predictions to investment planning."""
        # Step 1: Generate budget predictions
        budget_engine = PredictiveBudgetingEngine()
        transactions = generate_test_transactions(12)
        run_async(budget_engine.train_on_historical_data(transactions, 12))
        
        predictions = run_async(budget_engine.predict_all_categories(3))
        
        # Step 2: Calculate available savings
        monthly_income = 5000
        predicted_expenses = sum(
            sum(p.predicted_amount for p in preds) / len(preds)
            for preds in predictions.values()
        )
        available_savings = monthly_income - predicted_expenses
        
        assert available_savings > 0
        
        # Step 3: Set up investment coordination
        coordinator = InvestmentAdviceCoordinator()
        coordinator.register_goal(InvestmentGoal(
            goal_id="savings_goal",
            name="Savings Goal",
            target_amount=available_savings * 12,  # 1 year savings
            current_amount=0,
            target_date=datetime.now() + timedelta(days=365),
            priority=GoalPriority.MEDIUM,
            risk_tolerance=RiskProfile.MODERATE,
            monthly_contribution=available_savings
        ))
        
        # Step 4: Get coordinated investment strategy
        strategies = run_async(coordinator.coordinate_goal_strategies(available_savings))
        
        assert len(strategies) == 1
        assert strategies[0].recommended_monthly_contribution > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
