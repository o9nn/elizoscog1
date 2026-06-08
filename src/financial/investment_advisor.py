"""
Investment Advice Coordination - Phase 6 Implementation

Implements coordinated investment advice features:
- Portfolio allocation recommendations
- Risk-adjusted investment suggestions
- Multi-objective goal coordination
- Rebalancing recommendations
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class RiskProfile(Enum):
    """Investor risk profiles"""
    CONSERVATIVE = "conservative"      # 80% bonds, 20% stocks
    MODERATE_CONSERVATIVE = "moderate_conservative"  # 60% bonds, 40% stocks
    MODERATE = "moderate"              # 40% bonds, 60% stocks
    MODERATE_AGGRESSIVE = "moderate_aggressive"  # 20% bonds, 80% stocks
    AGGRESSIVE = "aggressive"          # 10% bonds, 90% stocks


class AssetClass(Enum):
    """Investment asset classes"""
    US_STOCKS = "us_stocks"
    INTERNATIONAL_STOCKS = "international_stocks"
    EMERGING_MARKETS = "emerging_markets"
    BONDS = "bonds"
    REAL_ESTATE = "real_estate"
    COMMODITIES = "commodities"
    CASH = "cash"


class GoalPriority(Enum):
    """Financial goal priority levels"""
    CRITICAL = 1      # Emergency fund, debt payoff
    HIGH = 2          # Retirement, children's education
    MEDIUM = 3        # Home purchase, major purchases
    LOW = 4           # Vacation, lifestyle upgrades


@dataclass
class AssetAllocation:
    """Recommended asset allocation"""
    asset_class: AssetClass
    percentage: float
    expected_return: float
    risk_level: float  # Standard deviation
    reasoning: str


@dataclass
class InvestmentGoal:
    """Financial investment goal"""
    goal_id: str
    name: str
    target_amount: float
    current_amount: float
    target_date: datetime
    priority: GoalPriority
    risk_tolerance: RiskProfile
    monthly_contribution: float = 0
    
    @property
    def progress_percentage(self) -> float:
        if self.target_amount <= 0:
            return 100.0
        return min(100.0, (self.current_amount / self.target_amount) * 100)


@dataclass
class PortfolioRecommendation:
    """Complete portfolio recommendation"""
    allocations: List[AssetAllocation]
    expected_annual_return: float
    expected_volatility: float
    sharpe_ratio: float
    rebalancing_frequency: str
    rationale: str
    confidence: float


@dataclass
class RebalancingAction:
    """Portfolio rebalancing action"""
    asset_class: AssetClass
    action: str  # "buy" or "sell"
    current_percentage: float
    target_percentage: float
    amount_to_adjust: float
    priority: int
    reason: str


@dataclass
class GoalAllocationStrategy:
    """Coordinated allocation strategy across goals"""
    goal_id: str
    goal_name: str
    recommended_monthly_contribution: float
    portfolio_recommendation: PortfolioRecommendation
    expected_completion_date: datetime
    success_probability: float
    adjustments_needed: List[str]


class InvestmentAdviceCoordinator:
    """
    Coordinates investment advice across multiple financial goals,
    providing unified portfolio recommendations and rebalancing guidance.
    """
    
    # Asset class characteristics (expected returns and volatility)
    ASSET_CHARACTERISTICS = {
        AssetClass.US_STOCKS: {
            'expected_return': 0.10,
            'volatility': 0.16,
            'min_horizon_years': 5
        },
        AssetClass.INTERNATIONAL_STOCKS: {
            'expected_return': 0.09,
            'volatility': 0.18,
            'min_horizon_years': 5
        },
        AssetClass.EMERGING_MARKETS: {
            'expected_return': 0.11,
            'volatility': 0.24,
            'min_horizon_years': 7
        },
        AssetClass.BONDS: {
            'expected_return': 0.04,
            'volatility': 0.05,
            'min_horizon_years': 2
        },
        AssetClass.REAL_ESTATE: {
            'expected_return': 0.08,
            'volatility': 0.14,
            'min_horizon_years': 5
        },
        AssetClass.COMMODITIES: {
            'expected_return': 0.05,
            'volatility': 0.20,
            'min_horizon_years': 3
        },
        AssetClass.CASH: {
            'expected_return': 0.02,
            'volatility': 0.01,
            'min_horizon_years': 0
        }
    }
    
    # Risk profile to allocation mapping
    RISK_ALLOCATIONS = {
        RiskProfile.CONSERVATIVE: {
            AssetClass.BONDS: 0.50,
            AssetClass.US_STOCKS: 0.15,
            AssetClass.INTERNATIONAL_STOCKS: 0.05,
            AssetClass.REAL_ESTATE: 0.10,
            AssetClass.CASH: 0.20
        },
        RiskProfile.MODERATE_CONSERVATIVE: {
            AssetClass.BONDS: 0.40,
            AssetClass.US_STOCKS: 0.25,
            AssetClass.INTERNATIONAL_STOCKS: 0.10,
            AssetClass.REAL_ESTATE: 0.10,
            AssetClass.CASH: 0.15
        },
        RiskProfile.MODERATE: {
            AssetClass.BONDS: 0.30,
            AssetClass.US_STOCKS: 0.35,
            AssetClass.INTERNATIONAL_STOCKS: 0.15,
            AssetClass.REAL_ESTATE: 0.10,
            AssetClass.COMMODITIES: 0.05,
            AssetClass.CASH: 0.05
        },
        RiskProfile.MODERATE_AGGRESSIVE: {
            AssetClass.BONDS: 0.15,
            AssetClass.US_STOCKS: 0.40,
            AssetClass.INTERNATIONAL_STOCKS: 0.20,
            AssetClass.EMERGING_MARKETS: 0.10,
            AssetClass.REAL_ESTATE: 0.10,
            AssetClass.COMMODITIES: 0.05
        },
        RiskProfile.AGGRESSIVE: {
            AssetClass.BONDS: 0.05,
            AssetClass.US_STOCKS: 0.40,
            AssetClass.INTERNATIONAL_STOCKS: 0.25,
            AssetClass.EMERGING_MARKETS: 0.15,
            AssetClass.REAL_ESTATE: 0.10,
            AssetClass.COMMODITIES: 0.05
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.goals: Dict[str, InvestmentGoal] = {}
        self.current_portfolio: Dict[AssetClass, float] = {}
        self.risk_free_rate = self.config.get('risk_free_rate', 0.03)
        self.rebalancing_threshold = self.config.get('rebalancing_threshold', 0.05)
        
    def register_goal(self, goal: InvestmentGoal) -> str:
        """Register a financial goal for coordinated advice."""
        self.goals[goal.goal_id] = goal
        logger.info(f"Registered goal: {goal.name} with target ${goal.target_amount:,.2f}")
        return goal.goal_id
    
    def set_current_portfolio(self, portfolio: Dict[str, float]) -> None:
        """
        Set current portfolio allocation.
        
        Args:
            portfolio: Dict mapping asset class names to dollar amounts
        """
        total = sum(portfolio.values())
        if total > 0:
            self.current_portfolio = {
                AssetClass(k) if isinstance(k, str) else k: v / total
                for k, v in portfolio.items()
            }
        logger.info(f"Set current portfolio with {len(portfolio)} asset classes")
    
    async def generate_portfolio_recommendation(
        self,
        risk_profile: RiskProfile,
        investment_horizon_years: int,
        total_investable: float
    ) -> PortfolioRecommendation:
        """
        Generate portfolio recommendation based on risk profile and time horizon.
        
        Args:
            risk_profile: Investor's risk tolerance profile
            investment_horizon_years: Investment time horizon in years
            total_investable: Total amount available for investment
            
        Returns:
            PortfolioRecommendation with detailed allocations
        """
        logger.info(f"Generating portfolio for {risk_profile.value}, "
                   f"horizon={investment_horizon_years}yr, amount=${total_investable:,.2f}")
        
        # Get base allocation for risk profile
        base_allocation = self.RISK_ALLOCATIONS[risk_profile].copy()
        
        # Adjust for time horizon
        adjusted_allocation = self._adjust_for_horizon(
            base_allocation, investment_horizon_years
        )
        
        # Build allocation recommendations
        allocations = []
        total_expected_return = 0
        total_variance = 0
        
        for asset_class, percentage in adjusted_allocation.items():
            if percentage <= 0:
                continue
            
            characteristics = self.ASSET_CHARACTERISTICS[asset_class]
            expected_return = characteristics['expected_return']
            volatility = characteristics['volatility']
            
            allocations.append(AssetAllocation(
                asset_class=asset_class,
                percentage=round(percentage * 100, 1),
                expected_return=round(expected_return * 100, 2),
                risk_level=round(volatility * 100, 2),
                reasoning=self._get_allocation_reasoning(
                    asset_class, percentage, investment_horizon_years
                )
            ))
            
            total_expected_return += percentage * expected_return
            total_variance += (percentage * volatility) ** 2
        
        total_volatility = np.sqrt(total_variance)
        sharpe_ratio = (total_expected_return - self.risk_free_rate) / total_volatility \
            if total_volatility > 0 else 0
        
        # Determine rebalancing frequency based on risk profile
        rebalancing_freq = self._get_rebalancing_frequency(risk_profile)
        
        return PortfolioRecommendation(
            allocations=allocations,
            expected_annual_return=round(total_expected_return * 100, 2),
            expected_volatility=round(total_volatility * 100, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            rebalancing_frequency=rebalancing_freq,
            rationale=self._generate_portfolio_rationale(
                risk_profile, investment_horizon_years, total_expected_return
            ),
            confidence=self._calculate_recommendation_confidence(
                investment_horizon_years, len(allocations)
            )
        )
    
    def _adjust_for_horizon(
        self,
        allocation: Dict[AssetClass, float],
        horizon_years: int
    ) -> Dict[AssetClass, float]:
        """Adjust allocation based on investment horizon."""
        adjusted = allocation.copy()
        
        # Short horizon: shift to more conservative
        if horizon_years < 3:
            shift_amount = 0.15
            # Reduce stocks
            for asset in [AssetClass.US_STOCKS, AssetClass.INTERNATIONAL_STOCKS, 
                         AssetClass.EMERGING_MARKETS]:
                if asset in adjusted:
                    reduction = min(adjusted[asset], shift_amount / 3)
                    adjusted[asset] -= reduction
            # Increase bonds and cash
            adjusted[AssetClass.BONDS] = adjusted.get(AssetClass.BONDS, 0) + shift_amount * 0.7
            adjusted[AssetClass.CASH] = adjusted.get(AssetClass.CASH, 0) + shift_amount * 0.3
        
        # Long horizon: can be slightly more aggressive
        elif horizon_years > 15:
            shift_amount = 0.05
            # Reduce bonds
            if AssetClass.BONDS in adjusted:
                reduction = min(adjusted[AssetClass.BONDS], shift_amount)
                adjusted[AssetClass.BONDS] -= reduction
                # Add to stocks
                adjusted[AssetClass.US_STOCKS] = \
                    adjusted.get(AssetClass.US_STOCKS, 0) + reduction * 0.6
                adjusted[AssetClass.INTERNATIONAL_STOCKS] = \
                    adjusted.get(AssetClass.INTERNATIONAL_STOCKS, 0) + reduction * 0.4
        
        # Normalize to ensure sum equals 1
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    def _get_allocation_reasoning(
        self,
        asset_class: AssetClass,
        percentage: float,
        horizon: int
    ) -> str:
        """Generate reasoning for asset allocation."""
        characteristics = self.ASSET_CHARACTERISTICS[asset_class]
        
        reasons = {
            AssetClass.US_STOCKS: f"Core equity exposure for growth; "
                f"suitable for {horizon}+ year horizon",
            AssetClass.INTERNATIONAL_STOCKS: "Geographic diversification and "
                "exposure to global growth opportunities",
            AssetClass.EMERGING_MARKETS: "Higher growth potential with "
                "increased volatility; long-term growth driver",
            AssetClass.BONDS: "Stability and income; reduces portfolio volatility "
                "and provides downside protection",
            AssetClass.REAL_ESTATE: "Inflation hedge and income generation "
                "with moderate correlation to stocks",
            AssetClass.COMMODITIES: "Inflation protection and portfolio diversification; "
                "low correlation with stocks/bonds",
            AssetClass.CASH: "Liquidity and capital preservation; "
                "provides stability and opportunity fund"
        }
        
        return reasons.get(asset_class, "Portfolio diversification component")
    
    def _get_rebalancing_frequency(self, risk_profile: RiskProfile) -> str:
        """Determine appropriate rebalancing frequency."""
        frequencies = {
            RiskProfile.CONSERVATIVE: "annually",
            RiskProfile.MODERATE_CONSERVATIVE: "semi-annually",
            RiskProfile.MODERATE: "quarterly",
            RiskProfile.MODERATE_AGGRESSIVE: "quarterly",
            RiskProfile.AGGRESSIVE: "semi-annually"
        }
        return frequencies.get(risk_profile, "quarterly")
    
    def _generate_portfolio_rationale(
        self,
        risk_profile: RiskProfile,
        horizon: int,
        expected_return: float
    ) -> str:
        """Generate overall portfolio rationale."""
        return (
            f"This {risk_profile.value} portfolio is designed for a {horizon}-year "
            f"investment horizon with an expected annual return of {expected_return*100:.1f}%. "
            f"The allocation balances growth potential with risk management appropriate "
            f"for your stated risk tolerance."
        )
    
    def _calculate_recommendation_confidence(
        self,
        horizon: int,
        num_assets: int
    ) -> float:
        """Calculate confidence in the recommendation."""
        base_confidence = 0.7
        
        # Higher confidence for longer horizons (mean reversion)
        horizon_bonus = min(0.15, horizon * 0.01)
        
        # Higher confidence with more diversification
        diversification_bonus = min(0.1, num_assets * 0.02)
        
        return round(min(0.95, base_confidence + horizon_bonus + diversification_bonus), 2)
    
    async def check_rebalancing_needs(
        self,
        target_allocation: Dict[AssetClass, float],
        portfolio_value: float
    ) -> List[RebalancingAction]:
        """
        Check if portfolio needs rebalancing and generate action items.
        
        Args:
            target_allocation: Target allocation percentages
            portfolio_value: Current total portfolio value
            
        Returns:
            List of rebalancing actions needed
        """
        actions = []
        
        for asset_class, target_pct in target_allocation.items():
            current_pct = self.current_portfolio.get(asset_class, 0)
            deviation = abs(current_pct - target_pct)
            
            if deviation >= self.rebalancing_threshold:
                if current_pct > target_pct:
                    action = "sell"
                    amount = (current_pct - target_pct) * portfolio_value
                    reason = f"Overweight by {(deviation * 100):.1f}%"
                else:
                    action = "buy"
                    amount = (target_pct - current_pct) * portfolio_value
                    reason = f"Underweight by {(deviation * 100):.1f}%"
                
                actions.append(RebalancingAction(
                    asset_class=asset_class,
                    action=action,
                    current_percentage=round(current_pct * 100, 1),
                    target_percentage=round(target_pct * 100, 1),
                    amount_to_adjust=round(amount, 2),
                    priority=self._get_rebalancing_priority(asset_class, deviation),
                    reason=reason
                ))
        
        # Sort by priority
        actions.sort(key=lambda x: x.priority)
        
        return actions
    
    def _get_rebalancing_priority(
        self,
        asset_class: AssetClass,
        deviation: float
    ) -> int:
        """Calculate priority of rebalancing action."""
        # Higher deviation = higher priority
        base_priority = int((1 - deviation) * 10)
        
        # Adjust for asset class importance
        importance_adjustments = {
            AssetClass.BONDS: -1,  # Higher priority for bond adjustments
            AssetClass.CASH: -1,
            AssetClass.EMERGING_MARKETS: 1,  # Lower priority
            AssetClass.COMMODITIES: 1
        }
        
        adjustment = importance_adjustments.get(asset_class, 0)
        return max(1, min(10, base_priority + adjustment))
    
    async def coordinate_goal_strategies(
        self,
        available_monthly_savings: float
    ) -> List[GoalAllocationStrategy]:
        """
        Coordinate investment strategies across all registered goals.
        
        Args:
            available_monthly_savings: Total monthly amount available for all goals
            
        Returns:
            Coordinated strategies for each goal
        """
        if not self.goals:
            logger.warning("No goals registered for coordination")
            return []
        
        strategies = []
        
        # Sort goals by priority and urgency
        sorted_goals = sorted(
            self.goals.values(),
            key=lambda g: (g.priority.value, g.target_date)
        )
        
        remaining_savings = available_monthly_savings
        
        for goal in sorted_goals:
            # Calculate years to goal
            years_to_goal = max(0.5, (goal.target_date - datetime.now()).days / 365)
            
            # Determine appropriate risk profile based on time horizon and goal priority
            risk_profile = self._determine_goal_risk_profile(goal, years_to_goal)
            
            # Generate portfolio recommendation for this goal
            recommendation = await self.generate_portfolio_recommendation(
                risk_profile=risk_profile,
                investment_horizon_years=int(years_to_goal),
                total_investable=goal.current_amount
            )
            
            # Calculate required monthly contribution
            required_monthly = self._calculate_required_contribution(
                goal.target_amount,
                goal.current_amount,
                years_to_goal,
                recommendation.expected_annual_return / 100
            )
            
            # Allocate from remaining savings
            allocated_monthly = min(remaining_savings, required_monthly)
            remaining_savings -= allocated_monthly
            
            # Calculate success probability
            success_prob = self._calculate_success_probability(
                goal, allocated_monthly, recommendation.expected_annual_return / 100
            )
            
            # Determine adjustments needed
            adjustments = self._determine_needed_adjustments(
                goal, allocated_monthly, required_monthly, success_prob
            )
            
            # Calculate expected completion date
            expected_completion = self._estimate_completion_date(
                goal, allocated_monthly, recommendation.expected_annual_return / 100
            )
            
            strategies.append(GoalAllocationStrategy(
                goal_id=goal.goal_id,
                goal_name=goal.name,
                recommended_monthly_contribution=round(allocated_monthly, 2),
                portfolio_recommendation=recommendation,
                expected_completion_date=expected_completion,
                success_probability=round(success_prob, 2),
                adjustments_needed=adjustments
            ))
        
        return strategies
    
    def _determine_goal_risk_profile(
        self,
        goal: InvestmentGoal,
        years_to_goal: float
    ) -> RiskProfile:
        """Determine appropriate risk profile based on goal and timeline."""
        # Start with user's stated tolerance
        base_profile = goal.risk_tolerance
        
        # Adjust based on time horizon
        if years_to_goal < 2:
            # Short horizon: force conservative
            return RiskProfile.CONSERVATIVE
        elif years_to_goal < 5:
            # Medium horizon: cap at moderate
            if base_profile in [RiskProfile.AGGRESSIVE, RiskProfile.MODERATE_AGGRESSIVE]:
                return RiskProfile.MODERATE
        
        # Adjust based on goal priority
        if goal.priority == GoalPriority.CRITICAL:
            # Critical goals: reduce risk
            profile_order = list(RiskProfile)
            current_idx = profile_order.index(base_profile)
            return profile_order[max(0, current_idx - 1)]
        
        return base_profile
    
    def _calculate_required_contribution(
        self,
        target: float,
        current: float,
        years: float,
        annual_return: float
    ) -> float:
        """Calculate required monthly contribution to reach goal."""
        if years <= 0:
            return max(0, target - current)
        
        months = int(years * 12)
        monthly_rate = (1 + annual_return) ** (1/12) - 1
        
        # Future value of current savings
        fv_current = current * (1 + monthly_rate) ** months
        
        # Required additional amount
        remaining_needed = target - fv_current
        
        if remaining_needed <= 0:
            return 0
        
        # PMT formula for required monthly payment
        if monthly_rate > 0:
            pmt = remaining_needed * monthly_rate / ((1 + monthly_rate) ** months - 1)
        else:
            pmt = remaining_needed / months
        
        return max(0, pmt)
    
    def _calculate_success_probability(
        self,
        goal: InvestmentGoal,
        monthly_contribution: float,
        expected_return: float
    ) -> float:
        """Calculate probability of reaching the goal."""
        years = max(0.5, (goal.target_date - datetime.now()).days / 365)
        months = int(years * 12)
        monthly_rate = (1 + expected_return) ** (1/12) - 1
        
        # Projected future value
        fv = goal.current_amount
        for _ in range(months):
            fv = fv * (1 + monthly_rate) + monthly_contribution
        
        # Simple probability based on projected vs target
        ratio = fv / goal.target_amount if goal.target_amount > 0 else 1.0
        
        # Apply uncertainty factor based on time horizon and volatility
        # Base uncertainty (10%) increases by 2% per year to account for market volatility
        uncertainty = 0.1 + (years * 0.02)  # Longer = more uncertainty
        
        if ratio >= 1.0:
            # When projected value exceeds target:
            # - Base success probability (0.7) + bonus for surplus (up to 0.25)
            # - Capped at 0.95 to reflect inherent market uncertainty
            base_success_prob = 0.7
            surplus_bonus_factor = 0.2  # Max 20% bonus per 100% surplus
            max_probability = 0.95
            return min(max_probability, base_success_prob + (ratio - 1.0) * surplus_bonus_factor)
        else:
            # When projected value is below target:
            # - Scale probability by ratio (how close to target)
            # - Apply uncertainty discount
            # - Minimum 10% probability to avoid zero-probability scenarios
            min_probability = 0.1
            return max(min_probability, ratio * (1 - uncertainty))
    
    def _determine_needed_adjustments(
        self,
        goal: InvestmentGoal,
        allocated: float,
        required: float,
        success_prob: float
    ) -> List[str]:
        """Determine what adjustments are needed to improve goal success."""
        adjustments = []
        
        if allocated < required * 0.8:
            shortfall = required - allocated
            adjustments.append(
                f"Consider increasing monthly contribution by ${shortfall:.0f}"
            )
        
        if success_prob < 0.6:
            adjustments.append(
                "Consider extending the target date to increase success probability"
            )
        
        if success_prob < 0.4 and goal.priority != GoalPriority.CRITICAL:
            adjustments.append(
                "Consider reducing the target amount or prioritizing other goals"
            )
        
        if not adjustments and success_prob >= 0.8:
            adjustments.append("Goal is on track - continue current strategy")
        
        return adjustments
    
    def _estimate_completion_date(
        self,
        goal: InvestmentGoal,
        monthly_contribution: float,
        expected_return: float
    ) -> datetime:
        """Estimate when goal will be reached."""
        monthly_rate = (1 + expected_return) ** (1/12) - 1
        current = goal.current_amount
        target = goal.target_amount
        
        if current >= target:
            return datetime.now()
        
        months = 0
        max_months = 480  # 40 years max
        
        while current < target and months < max_months:
            current = current * (1 + monthly_rate) + monthly_contribution
            months += 1
        
        return datetime.now() + timedelta(days=months * 30)
    
    async def generate_investment_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive investment advice summary."""
        if not self.goals:
            return {
                'status': 'no_goals',
                'message': 'No investment goals registered. Add goals to receive coordinated advice.'
            }
        
        total_target = sum(g.target_amount for g in self.goals.values())
        total_current = sum(g.current_amount for g in self.goals.values())
        
        return {
            'status': 'active',
            'summary': {
                'total_goals': len(self.goals),
                'total_target_value': round(total_target, 2),
                'total_current_value': round(total_current, 2),
                'overall_progress': round((total_current / total_target) * 100, 1) 
                    if total_target > 0 else 0,
                'goals_by_priority': {
                    priority.name: len([g for g in self.goals.values() 
                                       if g.priority == priority])
                    for priority in GoalPriority
                }
            },
            'recommendations': [
                "Review and rebalance portfolio quarterly",
                "Increase savings rate during high-income months",
                "Consider tax-advantaged accounts for long-term goals"
            ],
            'generated_at': datetime.now().isoformat()
        }


# Export main classes
__all__ = [
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
