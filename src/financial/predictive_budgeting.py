"""
Predictive Budgeting Models - Phase 6 Implementation

Implements ML-based predictive budgeting models for:
- Monthly expense forecasting by category
- Budget variance prediction
- Cash flow projections
- Financial goal tracking predictions
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available; predictive models will use statistical fallback")


class PredictionModel(Enum):
    """Types of prediction models available"""
    LINEAR_REGRESSION = "linear_regression"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    MOVING_AVERAGE = "moving_average"
    SEASONAL = "seasonal"
    ENSEMBLE = "ensemble"


@dataclass
class BudgetPrediction:
    """Result of a budget prediction"""
    category: str
    period: str  # e.g., "2026-07", "2026-Q3"
    predicted_amount: float
    lower_bound: float
    upper_bound: float
    confidence: float
    model_used: str
    trend: str  # "increasing", "decreasing", "stable"
    variance_from_budget: Optional[float] = None
    factors: List[str] = field(default_factory=list)


@dataclass
class CashFlowPrediction:
    """Cash flow projection"""
    period: str
    predicted_income: float
    predicted_expenses: float
    net_cash_flow: float
    cumulative_balance: float
    confidence: float
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class BudgetRecommendation:
    """Budget adjustment recommendation"""
    category: str
    current_budget: float
    recommended_budget: float
    adjustment_percentage: float
    reason: str
    priority: int  # 1-10
    confidence: float


class PredictiveBudgetingEngine:
    """
    ML-based predictive budgeting engine that forecasts future expenses,
    identifies budget variances, and provides actionable recommendations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.smoothing_alpha = self.config.get('smoothing_alpha', 0.3)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.min_data_points = self.config.get('min_data_points', 3)
        self.forecast_months = self.config.get('forecast_months', 3)
        
        # Model storage
        self.trained_models: Dict[str, Any] = {}
        self.category_history: Dict[str, List[float]] = defaultdict(list)
        self.monthly_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Seasonal adjustment factors by category
        self.seasonal_factors: Dict[str, List[float]] = {}
        
    async def train_on_historical_data(
        self, 
        transactions: List[Dict[str, Any]],
        months_of_data: int = 12
    ) -> Dict[str, Any]:
        """
        Train predictive models on historical transaction data.
        
        Args:
            transactions: List of historical transactions with date, amount, category
            months_of_data: Number of months of data to use for training
            
        Returns:
            Training summary with model performance metrics
        """
        logger.info(f"Training predictive models on {len(transactions)} transactions")
        
        # Group transactions by month and category
        self.monthly_totals.clear()
        self.category_history.clear()
        
        cutoff_date = datetime.now() - timedelta(days=months_of_data * 30)
        
        for tx in transactions:
            try:
                tx_date = self._parse_date(tx.get('date', ''))
                if tx_date and tx_date >= cutoff_date:
                    month_key = tx_date.strftime('%Y-%m')
                    category = tx.get('category', 'uncategorized')
                    amount = abs(float(tx.get('amount', 0)))
                    
                    self.monthly_totals[month_key][category] += amount
            except Exception as e:
                logger.warning(f"Error processing transaction: {e}")
                continue
        
        # Build category histories (sorted by month)
        sorted_months = sorted(self.monthly_totals.keys())
        categories = set()
        for month_data in self.monthly_totals.values():
            categories.update(month_data.keys())
        
        for category in categories:
            self.category_history[category] = [
                self.monthly_totals[month].get(category, 0)
                for month in sorted_months
            ]
        
        # Calculate seasonal factors for each category
        self._calculate_seasonal_factors(sorted_months)
        
        # Train models if sklearn is available
        training_results = {}
        if SKLEARN_AVAILABLE and len(sorted_months) >= self.min_data_points:
            training_results = self._train_sklearn_models()
        
        return {
            'status': 'success',
            'months_analyzed': len(sorted_months),
            'categories_found': len(categories),
            'models_trained': len(self.trained_models),
            'training_results': training_results,
            'data_quality': self._assess_data_quality(sorted_months)
        }
    
    def _calculate_seasonal_factors(self, sorted_months: List[str]):
        """Calculate seasonal adjustment factors for each category."""
        for category, history in self.category_history.items():
            if len(history) < 12:
                # Not enough data for seasonal analysis
                self.seasonal_factors[category] = [1.0] * 12
                continue
            
            # Group by calendar month
            monthly_averages = defaultdict(list)
            for i, month_str in enumerate(sorted_months):
                if i < len(history):
                    month_num = int(month_str.split('-')[1])
                    monthly_averages[month_num].append(history[i])
            
            # Calculate seasonal factors as ratio to overall average
            overall_avg = np.mean(history) if history else 1.0
            if overall_avg == 0:
                overall_avg = 1.0
            
            factors = []
            for month in range(1, 13):
                if monthly_averages[month]:
                    month_avg = np.mean(monthly_averages[month])
                    factors.append(month_avg / overall_avg)
                else:
                    factors.append(1.0)
            
            self.seasonal_factors[category] = factors
    
    def _train_sklearn_models(self) -> Dict[str, Any]:
        """Train sklearn regression models for each category."""
        results = {}
        
        for category, history in self.category_history.items():
            if len(history) < self.min_data_points:
                continue
            
            try:
                X = np.arange(len(history)).reshape(-1, 1)
                y = np.array(history)
                
                # Train Ridge regression (regularized linear regression)
                model = Ridge(alpha=1.0)
                model.fit(X, y)
                
                # Calculate R² score
                predictions = model.predict(X)
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                self.trained_models[category] = {
                    'model': model,
                    'r2_score': r2_score,
                    'last_idx': len(history) - 1
                }
                
                results[category] = {
                    'r2_score': r2_score,
                    'data_points': len(history),
                    'trend': 'increasing' if model.coef_[0] > 0 else 'decreasing'
                }
                
            except Exception as e:
                logger.warning(f"Failed to train model for {category}: {e}")
                continue
        
        return results
    
    async def predict_category_spending(
        self,
        category: str,
        months_ahead: int = 3,
        include_seasonal: bool = True
    ) -> List[BudgetPrediction]:
        """
        Predict spending for a specific category.
        
        Args:
            category: Expense category to predict
            months_ahead: Number of months to forecast
            include_seasonal: Whether to apply seasonal adjustments
            
        Returns:
            List of BudgetPrediction objects for each forecasted month
        """
        predictions = []
        history = self.category_history.get(category, [])
        
        if len(history) < self.min_data_points:
            logger.warning(f"Insufficient data for {category}: {len(history)} points")
            return predictions
        
        current_date = datetime.now()
        
        for i in range(1, months_ahead + 1):
            future_date = current_date + timedelta(days=30 * i)
            period = future_date.strftime('%Y-%m')
            future_month = future_date.month
            
            # Get base prediction
            predicted_amount, confidence, model_used = self._get_prediction(
                category, len(history) + i - 1
            )
            
            # Apply seasonal adjustment
            if include_seasonal and category in self.seasonal_factors:
                seasonal_factor = self.seasonal_factors[category][future_month - 1]
                predicted_amount *= seasonal_factor
            
            # Calculate confidence interval
            std_dev = np.std(history) if len(history) > 1 else predicted_amount * 0.2
            z_score = 1.96  # 95% confidence
            lower_bound = max(0, predicted_amount - z_score * std_dev)
            upper_bound = predicted_amount + z_score * std_dev
            
            # Determine trend
            trend = self._determine_trend(history)
            
            predictions.append(BudgetPrediction(
                category=category,
                period=period,
                predicted_amount=round(predicted_amount, 2),
                lower_bound=round(lower_bound, 2),
                upper_bound=round(upper_bound, 2),
                confidence=round(confidence, 3),
                model_used=model_used,
                trend=trend,
                factors=['historical_trend', 'seasonal_adjustment' if include_seasonal else 'base_trend']
            ))
        
        return predictions
    
    def _get_prediction(
        self, 
        category: str, 
        future_idx: int
    ) -> Tuple[float, float, str]:
        """Get prediction using best available model."""
        history = self.category_history.get(category, [])
        
        # Try sklearn model first
        if SKLEARN_AVAILABLE and category in self.trained_models:
            model_info = self.trained_models[category]
            model = model_info['model']
            X_pred = np.array([[future_idx]])
            prediction = float(model.predict(X_pred)[0])
            confidence = min(0.95, model_info['r2_score'] + 0.3)
            return max(0, prediction), confidence, 'ridge_regression'
        
        # Fallback to exponential smoothing
        if history:
            prediction = self._exponential_smoothing(history, future_idx - len(history) + 1)
            confidence = 0.7
            return prediction, confidence, 'exponential_smoothing'
        
        return 0, 0.3, 'no_data'
    
    def _exponential_smoothing(
        self, 
        history: List[float], 
        steps_ahead: int = 1
    ) -> float:
        """Simple exponential smoothing forecast."""
        if not history:
            return 0
        
        alpha = self.smoothing_alpha
        smoothed = history[0]
        
        for value in history[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        # Forecast remains at last smoothed value
        return max(0, smoothed)
    
    def _determine_trend(self, history: List[float]) -> str:
        """Determine trend direction from history."""
        if len(history) < 3:
            return 'stable'
        
        # Compare recent average to older average
        recent = np.mean(history[-3:])
        older = np.mean(history[:-3]) if len(history) > 3 else history[0]
        
        if older == 0:
            return 'stable'
        
        change_pct = (recent - older) / older
        
        if change_pct > 0.1:
            return 'increasing'
        elif change_pct < -0.1:
            return 'decreasing'
        return 'stable'
    
    async def predict_all_categories(
        self,
        months_ahead: int = 3
    ) -> Dict[str, List[BudgetPrediction]]:
        """Predict spending for all categories with historical data."""
        all_predictions = {}
        
        for category in self.category_history.keys():
            predictions = await self.predict_category_spending(category, months_ahead)
            if predictions:
                all_predictions[category] = predictions
        
        return all_predictions
    
    async def predict_cash_flow(
        self,
        income_history: List[Dict[str, Any]],
        expense_predictions: Dict[str, List[BudgetPrediction]],
        current_balance: float = 0,
        months_ahead: int = 3
    ) -> List[CashFlowPrediction]:
        """
        Generate cash flow projections.
        
        Args:
            income_history: Historical income transactions
            expense_predictions: Predicted expenses by category
            current_balance: Current account balance
            months_ahead: Number of months to project
            
        Returns:
            List of CashFlowPrediction for each month
        """
        projections = []
        cumulative = current_balance
        
        # Calculate average monthly income
        monthly_income = defaultdict(float)
        for tx in income_history:
            try:
                tx_date = self._parse_date(tx.get('date', ''))
                if tx_date:
                    month_key = tx_date.strftime('%Y-%m')
                    monthly_income[month_key] += abs(float(tx.get('amount', 0)))
            except Exception:
                continue
        
        avg_income = np.mean(list(monthly_income.values())) if monthly_income else 0
        income_std = np.std(list(monthly_income.values())) if len(monthly_income) > 1 else avg_income * 0.2
        
        current_date = datetime.now()
        
        for i in range(1, months_ahead + 1):
            future_date = current_date + timedelta(days=30 * i)
            period = future_date.strftime('%Y-%m')
            
            # Predict income (simple average with variance)
            predicted_income = avg_income
            
            # Sum predicted expenses for this period
            predicted_expenses = 0
            for category, preds in expense_predictions.items():
                for pred in preds:
                    if pred.period == period:
                        predicted_expenses += pred.predicted_amount
            
            net_flow = predicted_income - predicted_expenses
            cumulative += net_flow
            
            # Assess risk factors
            risk_factors = []
            if net_flow < 0:
                risk_factors.append('negative_cash_flow')
            if cumulative < 0:
                risk_factors.append('projected_deficit')
            if income_std / avg_income > 0.3 if avg_income > 0 else False:
                risk_factors.append('income_volatility')
            
            # Calculate confidence based on data quality
            confidence = 0.8 if len(monthly_income) >= 6 else 0.6
            
            projections.append(CashFlowPrediction(
                period=period,
                predicted_income=round(predicted_income, 2),
                predicted_expenses=round(predicted_expenses, 2),
                net_cash_flow=round(net_flow, 2),
                cumulative_balance=round(cumulative, 2),
                confidence=confidence,
                risk_factors=risk_factors
            ))
        
        return projections
    
    async def generate_budget_recommendations(
        self,
        current_budget: Dict[str, float],
        predictions: Dict[str, List[BudgetPrediction]],
        income_target: float
    ) -> List[BudgetRecommendation]:
        """
        Generate budget adjustment recommendations.
        
        Args:
            current_budget: Current budget allocations by category
            predictions: Predicted spending by category
            income_target: Target monthly income/budget cap
            
        Returns:
            List of BudgetRecommendation objects
        """
        recommendations = []
        
        total_predicted = 0
        category_predictions = {}
        
        # Get average predicted spending per category
        for category, preds in predictions.items():
            if preds:
                avg_predicted = np.mean([p.predicted_amount for p in preds])
                category_predictions[category] = avg_predicted
                total_predicted += avg_predicted
        
        # Calculate budget utilization
        budget_utilization = total_predicted / income_target if income_target > 0 else 1.0
        
        for category, predicted in category_predictions.items():
            current = current_budget.get(category, 0)
            
            if current == 0:
                # No budget set, recommend based on prediction
                recommended = predicted * 1.1  # 10% buffer
                reason = f"No budget set; recommend based on predicted spending of ${predicted:.2f}"
                priority = 7
            elif predicted > current * 1.2:
                # Consistently over budget
                recommended = predicted * 1.1
                reason = f"Spending exceeds budget by {((predicted/current - 1) * 100):.0f}%"
                priority = 8
            elif predicted < current * 0.7:
                # Consistently under budget
                recommended = predicted * 1.15
                reason = f"Budget underutilized by {((1 - predicted/current) * 100):.0f}%"
                priority = 4
            else:
                # Budget is appropriate
                recommended = current
                reason = "Current budget aligns with spending patterns"
                priority = 2
            
            # Adjust for overall budget cap
            if budget_utilization > 1.0 and priority > 3:
                scale_factor = 0.9  # Reduce recommendations by 10%
                recommended *= scale_factor
                reason += "; adjusted for overall budget cap"
            
            if abs(recommended - current) / current > 0.05 if current > 0 else recommended > 0:
                recommendations.append(BudgetRecommendation(
                    category=category,
                    current_budget=round(current, 2),
                    recommended_budget=round(recommended, 2),
                    adjustment_percentage=round((recommended - current) / current * 100 if current > 0 else 100, 1),
                    reason=reason,
                    priority=priority,
                    confidence=0.75
                ))
        
        # Sort by priority (highest first)
        recommendations.sort(key=lambda x: -x.priority)
        
        return recommendations
    
    async def predict_goal_progress(
        self,
        goal_name: str,
        target_amount: float,
        current_savings: float,
        monthly_contribution: float,
        target_date: datetime,
        expected_return: float = 0.05
    ) -> Dict[str, Any]:
        """
        Predict progress toward a financial goal.
        
        Args:
            goal_name: Name of the financial goal
            target_amount: Target savings amount
            current_savings: Current amount saved
            monthly_contribution: Expected monthly contribution
            target_date: Target date to reach goal
            expected_return: Expected annual return on savings
            
        Returns:
            Goal progress prediction with scenarios
        """
        today = datetime.now()
        months_remaining = max(1, (target_date - today).days // 30)
        monthly_return = (1 + expected_return) ** (1/12) - 1
        
        # Calculate future value with compound interest
        future_value = current_savings
        for _ in range(months_remaining):
            future_value = future_value * (1 + monthly_return) + monthly_contribution
        
        shortfall = target_amount - future_value
        on_track = future_value >= target_amount
        
        # Calculate required monthly contribution to meet goal
        if shortfall > 0 and months_remaining > 0:
            # PMT formula: PMT = (PV * r - FV * r) / ((1+r)^n - 1)
            r = monthly_return
            n = months_remaining
            pv = current_savings
            fv = target_amount
            
            if r > 0:
                required_contribution = (fv - pv * (1 + r) ** n) * r / ((1 + r) ** n - 1)
            else:
                required_contribution = (fv - pv) / n
        else:
            required_contribution = monthly_contribution
        
        # Generate scenarios
        scenarios = {
            'optimistic': {
                'return_rate': expected_return + 0.02,
                'projected_value': self._calculate_fv(
                    current_savings, monthly_contribution, 
                    expected_return + 0.02, months_remaining
                )
            },
            'base': {
                'return_rate': expected_return,
                'projected_value': future_value
            },
            'pessimistic': {
                'return_rate': max(0, expected_return - 0.02),
                'projected_value': self._calculate_fv(
                    current_savings, monthly_contribution,
                    max(0, expected_return - 0.02), months_remaining
                )
            }
        }
        
        return {
            'goal_name': goal_name,
            'target_amount': target_amount,
            'current_savings': current_savings,
            'projected_value': round(future_value, 2),
            'shortfall': round(max(0, shortfall), 2),
            'surplus': round(max(0, -shortfall), 2),
            'on_track': on_track,
            'completion_probability': min(100, round(future_value / target_amount * 100, 1)),
            'months_remaining': months_remaining,
            'required_monthly_contribution': round(max(0, required_contribution), 2),
            'current_monthly_contribution': monthly_contribution,
            'scenarios': scenarios,
            'recommendations': self._generate_goal_recommendations(
                on_track, shortfall, required_contribution, monthly_contribution
            )
        }
    
    def _calculate_fv(
        self, 
        pv: float, 
        pmt: float, 
        annual_rate: float, 
        months: int
    ) -> float:
        """Calculate future value with monthly contributions."""
        monthly_rate = (1 + annual_rate) ** (1/12) - 1
        fv = pv
        for _ in range(months):
            fv = fv * (1 + monthly_rate) + pmt
        return round(fv, 2)
    
    def _generate_goal_recommendations(
        self,
        on_track: bool,
        shortfall: float,
        required: float,
        current: float
    ) -> List[str]:
        """Generate recommendations for reaching a financial goal."""
        recommendations = []
        
        if on_track:
            recommendations.append("You're on track to meet your goal!")
            if required < current * 0.8:
                recommendations.append(
                    f"Consider reducing contributions to ${required:.0f}/month "
                    "and redirecting savings to other goals."
                )
        else:
            increase_needed = required - current
            recommendations.append(
                f"Consider increasing monthly contributions by ${increase_needed:.0f} "
                f"to reach your goal."
            )
            if shortfall > 10000:
                recommendations.append(
                    "Review your timeline - extending the target date could reduce "
                    "required monthly contributions."
                )
            recommendations.append(
                "Look for opportunities to reduce discretionary spending "
                "to increase savings rate."
            )
        
        return recommendations
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        
        formats = [
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%d',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S%z',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.replace('Z', '+00:00').split('+')[0], fmt.split('%z')[0])
            except ValueError:
                continue
        
        return None
    
    def _assess_data_quality(self, months: List[str]) -> Dict[str, Any]:
        """Assess quality of training data."""
        total_transactions = sum(
            sum(cat_data.values()) 
            for cat_data in self.monthly_totals.values()
        )
        
        return {
            'months_covered': len(months),
            'categories_tracked': len(self.category_history),
            'total_data_points': total_transactions,
            'data_completeness': min(1.0, len(months) / 12),
            'recommendation': 'sufficient' if len(months) >= 6 else 'collect_more_data'
        }


# Export main classes
__all__ = [
    'PredictiveBudgetingEngine',
    'BudgetPrediction',
    'CashFlowPrediction',
    'BudgetRecommendation',
    'PredictionModel'
]
