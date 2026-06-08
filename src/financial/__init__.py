"""
Financial package initialization
"""

from .gnucash_patterns import (
    GnuCashDataAccess,
    FinancialAnalysisPatterns,
    Account,
    Transaction,
    Split
)

from .reasoning_engine import FinancialReasoningEngine

from .predictive_budgeting import (
    PredictiveBudgetingEngine,
    BudgetPrediction,
    CashFlowPrediction,
    BudgetRecommendation,
    PredictionModel
)

from .investment_advisor import (
    InvestmentAdviceCoordinator,
    InvestmentGoal,
    PortfolioRecommendation,
    RebalancingAction,
    GoalAllocationStrategy,
    AssetAllocation,
    RiskProfile,
    AssetClass,
    GoalPriority
)

__all__ = [
    # GnuCash integration
    'GnuCashDataAccess',
    'FinancialAnalysisPatterns', 
    'FinancialReasoningEngine',
    'Account',
    'Transaction',
    'Split',
    # Predictive budgeting
    'PredictiveBudgetingEngine',
    'BudgetPrediction',
    'CashFlowPrediction',
    'BudgetRecommendation',
    'PredictionModel',
    # Investment advice
    'InvestmentAdviceCoordinator',
    'InvestmentGoal',
    'PortfolioRecommendation',
    'RebalancingAction',
    'GoalAllocationStrategy',
    'AssetAllocation',
    'RiskProfile',
    'AssetClass',
    'GoalPriority'
]