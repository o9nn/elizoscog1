"""
Cognitive Financial Analysis - Phase 3 Implementation
Advanced pattern recognition, predictive modeling, and anomaly detection for financial behavior
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of financial patterns that can be detected"""
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"
    TRENDING = "trending"
    ANOMALOUS = "anomalous"
    HABITUAL = "habitual"


@dataclass
class FinancialPattern:
    """Represents a detected financial pattern"""
    pattern_type: PatternType
    category: str
    confidence: float
    description: str
    impact_score: float
    recommendations: List[str]
    detected_at: datetime


@dataclass
class PredictionResult:
    """Result of financial prediction model"""
    category: str
    predicted_amount: float
    confidence_interval: Tuple[float, float]
    confidence_score: float
    model_type: str
    factors: List[str]


class CognitiveFinancialAnalyzer:
    """Advanced cognitive financial analysis engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.historical_data = {}
        self.learned_patterns = []
        self.model_cache = {}
        
    async def analyze_financial_behavior(self, 
                                       transaction_history: List[Dict[str, Any]], 
                                       timeframe_days: int = 365) -> Dict[str, Any]:
        """
        Comprehensive cognitive analysis of financial behavior patterns
        """
        logger.info(f"Starting cognitive analysis of {len(transaction_history)} transactions")
        
        # Prepare transaction data
        processed_data = self._preprocess_transactions(transaction_history)
        
        # Pattern recognition
        patterns = await self._detect_patterns(processed_data, timeframe_days)
        
        # Behavioral insights
        behavioral_insights = await self._analyze_behavioral_patterns(processed_data)
        
        # Risk assessment
        risk_assessment = await self._assess_financial_risk(processed_data)
        
        # Cognitive recommendations
        cognitive_recommendations = await self._generate_cognitive_recommendations(
            patterns, behavioral_insights, risk_assessment
        )
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "data_overview": {
                "transaction_count": len(transaction_history),
                "analysis_period_days": timeframe_days,
                "categories_analyzed": len(set(t.get('category', 'unknown') for t in processed_data))
            },
            "detected_patterns": [self._pattern_to_dict(p) for p in patterns],
            "behavioral_insights": behavioral_insights,
            "risk_assessment": risk_assessment,
            "cognitive_recommendations": cognitive_recommendations,
            "confidence_score": self._calculate_overall_confidence(patterns, processed_data)
        }
    
    async def predict_future_expenses(self, 
                                    transaction_history: List[Dict[str, Any]], 
                                    prediction_months: int = 3) -> Dict[str, List[PredictionResult]]:
        """
        Generate predictive models for future expense forecasting
        """
        logger.info(f"Generating {prediction_months}-month expense predictions")
        
        processed_data = self._preprocess_transactions(transaction_history)
        categories = self._get_expense_categories(processed_data)
        
        predictions = {}
        
        for category in categories:
            category_data = [t for t in processed_data if t.get('category') == category]
            if len(category_data) < 3:  # Need minimum data for prediction
                continue
                
            # Multiple prediction models
            trend_prediction = await self._trend_based_prediction(category_data, prediction_months)
            seasonal_prediction = await self._seasonal_prediction(category_data, prediction_months)
            ml_prediction = await self._ml_based_prediction(category_data, prediction_months)
            
            # Ensemble prediction combining multiple models
            ensemble_prediction = await self._ensemble_prediction([
                trend_prediction, seasonal_prediction, ml_prediction
            ], category)
            
            predictions[category] = [ensemble_prediction]
        
        return predictions
    
    async def detect_anomalies(self, 
                             transaction_history: List[Dict[str, Any]], 
                             sensitivity: float = 0.8) -> Dict[str, Any]:
        """
        Advanced anomaly detection using statistical and cognitive methods
        """
        logger.info(f"Running anomaly detection with sensitivity {sensitivity}")
        
        processed_data = self._preprocess_transactions(transaction_history)
        anomalies = {
            "statistical_anomalies": [],
            "behavioral_anomalies": [],
            "pattern_anomalies": [],
            "temporal_anomalies": []
        }
        
        # Statistical anomaly detection
        statistical_anomalies = await self._statistical_anomaly_detection(processed_data, sensitivity)
        anomalies["statistical_anomalies"] = statistical_anomalies
        
        # Behavioral anomaly detection
        behavioral_anomalies = await self._behavioral_anomaly_detection(processed_data)
        anomalies["behavioral_anomalies"] = behavioral_anomalies
        
        # Pattern-based anomaly detection
        pattern_anomalies = await self._pattern_anomaly_detection(processed_data)
        anomalies["pattern_anomalies"] = pattern_anomalies
        
        # Temporal anomaly detection
        temporal_anomalies = await self._temporal_anomaly_detection(processed_data)
        anomalies["temporal_anomalies"] = temporal_anomalies
        
        # Calculate overall anomaly score
        total_anomalies = sum(len(anomaly_list) for anomaly_list in anomalies.values())
        anomaly_score = min(total_anomalies / len(processed_data) * 100, 100) if processed_data else 0
        
        return {
            "anomaly_summary": {
                "total_anomalies": total_anomalies,
                "anomaly_score": anomaly_score,
                "severity_distribution": self._categorize_anomaly_severity(anomalies)
            },
            "detected_anomalies": anomalies,
            "analysis_metadata": {
                "sensitivity_threshold": sensitivity,
                "analysis_timestamp": datetime.now().isoformat(),
                "transactions_analyzed": len(processed_data)
            }
        }
    
    async def assess_financial_risk(self, 
                                  transaction_history: List[Dict[str, Any]], 
                                  income_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Comprehensive financial risk assessment framework
        """
        logger.info("Conducting comprehensive financial risk assessment")
        
        processed_transactions = self._preprocess_transactions(transaction_history)
        
        # Risk factors analysis
        liquidity_risk = await self._assess_liquidity_risk(processed_transactions, income_data)
        volatility_risk = await self._assess_volatility_risk(processed_transactions)
        category_concentration_risk = await self._assess_concentration_risk(processed_transactions)
        behavioral_risk = await self._assess_behavioral_risk(processed_transactions)
        
        # Overall risk score calculation
        risk_scores = {
            "liquidity_risk": liquidity_risk["score"],
            "volatility_risk": volatility_risk["score"],
            "concentration_risk": category_concentration_risk["score"],
            "behavioral_risk": behavioral_risk["score"]
        }
        
        overall_risk_score = sum(risk_scores.values()) / len(risk_scores)
        risk_level = self._categorize_risk_level(overall_risk_score)
        
        return {
            "overall_assessment": {
                "risk_score": overall_risk_score,
                "risk_level": risk_level,
                "primary_concerns": self._identify_primary_risk_concerns(risk_scores)
            },
            "detailed_analysis": {
                "liquidity_analysis": liquidity_risk,
                "volatility_analysis": volatility_risk,
                "concentration_analysis": category_concentration_risk,
                "behavioral_analysis": behavioral_risk
            },
            "risk_mitigation_strategies": await self._generate_risk_mitigation_strategies(risk_scores),
            "assessment_timestamp": datetime.now().isoformat()
        }
    
    # Helper methods for data processing and analysis
    
    def _preprocess_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess and clean transaction data"""
        processed = []
        for transaction in transactions:
            # Ensure required fields exist
            processed_transaction = {
                'amount': abs(float(transaction.get('amount', 0))),
                'date': transaction.get('date', datetime.now().isoformat()),
                'category': transaction.get('category', 'uncategorized'),
                'description': transaction.get('description', ''),
                'account': transaction.get('account', 'unknown')
            }
            
            # Skip zero-amount transactions
            if processed_transaction['amount'] > 0:
                processed.append(processed_transaction)
        
        return sorted(processed, key=lambda x: x['date'])
    
    async def _detect_patterns(self, data: List[Dict[str, Any]], timeframe_days: int) -> List[FinancialPattern]:
        """Detect various financial patterns in transaction data"""
        patterns = []
        
        # Seasonal pattern detection
        seasonal_patterns = await self._detect_seasonal_patterns(data)
        patterns.extend(seasonal_patterns)
        
        # Spending habit patterns
        habit_patterns = await self._detect_habit_patterns(data)
        patterns.extend(habit_patterns)
        
        # Trending patterns
        trend_patterns = await self._detect_trend_patterns(data)
        patterns.extend(trend_patterns)
        
        return patterns
    
    async def _detect_seasonal_patterns(self, data: List[Dict[str, Any]]) -> List[FinancialPattern]:
        """Detect seasonal spending patterns"""
        patterns = []
        
        # Group transactions by month
        monthly_spending = {}
        for transaction in data:
            try:
                date_obj = datetime.fromisoformat(transaction['date'].replace('Z', '+00:00'))
                month = date_obj.month
                category = transaction['category']
                
                if month not in monthly_spending:
                    monthly_spending[month] = {}
                if category not in monthly_spending[month]:
                    monthly_spending[month][category] = 0
                
                monthly_spending[month][category] += transaction['amount']
            except Exception as e:
                logger.warning(f"Error processing date {transaction['date']}: {e}")
                continue
        
        # Analyze for seasonal patterns
        for category in set(t['category'] for t in data):
            monthly_amounts = []
            for month in range(1, 13):
                amount = monthly_spending.get(month, {}).get(category, 0)
                monthly_amounts.append(amount)
            
            if max(monthly_amounts) > 0:
                # Simple seasonal detection: check if certain months are consistently higher
                avg_amount = sum(monthly_amounts) / 12
                seasonal_months = [i+1 for i, amount in enumerate(monthly_amounts) if amount > avg_amount * 1.5]
                
                if len(seasonal_months) >= 2:
                    patterns.append(FinancialPattern(
                        pattern_type=PatternType.SEASONAL,
                        category=category,
                        confidence=0.7,
                        description=f"Seasonal spending pattern detected in {category} during months {seasonal_months}",
                        impact_score=max(monthly_amounts) - avg_amount,
                        recommendations=[
                            f"Budget extra for {category} in months {seasonal_months}",
                            f"Consider setting aside funds in advance for seasonal {category} expenses"
                        ],
                        detected_at=datetime.now()
                    ))
        
        return patterns
    
    async def _detect_habit_patterns(self, data: List[Dict[str, Any]]) -> List[FinancialPattern]:
        """Detect habitual spending patterns"""
        patterns = []
        
        # Group by category and analyze frequency
        category_frequency = {}
        for transaction in data:
            category = transaction['category']
            if category not in category_frequency:
                category_frequency[category] = []
            category_frequency[category].append(transaction)
        
        # Detect high-frequency categories (habits)
        for category, transactions in category_frequency.items():
            if len(transactions) >= 10:  # Frequent transactions
                avg_amount = sum(t['amount'] for t in transactions) / len(transactions)
                consistency_score = 1.0 - (np.std([t['amount'] for t in transactions]) / avg_amount if avg_amount > 0 else 1)
                
                if consistency_score > 0.7:  # Consistent amounts indicate habit
                    patterns.append(FinancialPattern(
                        pattern_type=PatternType.HABITUAL,
                        category=category,
                        confidence=consistency_score,
                        description=f"Habitual spending pattern in {category}: {len(transactions)} transactions averaging ${avg_amount:.2f}",
                        impact_score=len(transactions) * avg_amount,
                        recommendations=[
                            f"Track {category} spending closely as it's a habitual expense",
                            f"Consider automating budgeting for {category} given its consistency"
                        ],
                        detected_at=datetime.now()
                    ))
        
        return patterns
    
    async def _detect_trend_patterns(self, data: List[Dict[str, Any]]) -> List[FinancialPattern]:
        """Detect trending patterns in spending"""
        patterns = []
        
        # Group by category and month for trend analysis
        monthly_category_spending = {}
        for transaction in data:
            try:
                date_obj = datetime.fromisoformat(transaction['date'].replace('Z', '+00:00'))
                month_key = f"{date_obj.year}-{date_obj.month:02d}"
                category = transaction['category']
                
                if category not in monthly_category_spending:
                    monthly_category_spending[category] = {}
                if month_key not in monthly_category_spending[category]:
                    monthly_category_spending[category][month_key] = 0
                
                monthly_category_spending[category][month_key] += transaction['amount']
            except Exception as e:
                logger.warning(f"Error processing trend data: {e}")
                continue
        
        # Analyze trends for each category
        for category, monthly_data in monthly_category_spending.items():
            if len(monthly_data) >= 3:  # Need at least 3 months of data
                sorted_months = sorted(monthly_data.keys())
                amounts = [monthly_data[month] for month in sorted_months]
                
                # Simple trend detection using linear regression slope
                if len(amounts) > 1:
                    x = list(range(len(amounts)))
                    slope = np.polyfit(x, amounts, 1)[0] if len(amounts) >= 2 else 0
                    
                    # Significant trend detection
                    avg_amount = sum(amounts) / len(amounts)
                    trend_strength = abs(slope) / avg_amount if avg_amount > 0 else 0
                    
                    if trend_strength > 0.1:  # 10% trend threshold
                        trend_direction = "increasing" if slope > 0 else "decreasing"
                        patterns.append(FinancialPattern(
                            pattern_type=PatternType.TRENDING,
                            category=category,
                            confidence=min(trend_strength, 1.0),
                            description=f"{trend_direction.capitalize()} trend in {category} spending: {trend_strength*100:.1f}% monthly change",
                            impact_score=abs(slope) * 12,  # Annual impact
                            recommendations=[
                                f"Monitor {category} spending trend carefully",
                                f"Adjust budget for {category} to account for {trend_direction} trend"
                            ],
                            detected_at=datetime.now()
                        ))
        
        return patterns
    
    # Additional helper methods would be implemented here...
    
    def _pattern_to_dict(self, pattern: FinancialPattern) -> Dict[str, Any]:
        """Convert FinancialPattern to dictionary"""
        return {
            "type": pattern.pattern_type.value,
            "category": pattern.category,
            "confidence": pattern.confidence,
            "description": pattern.description,
            "impact_score": pattern.impact_score,
            "recommendations": pattern.recommendations,
            "detected_at": pattern.detected_at.isoformat()
        }
    
    def _calculate_overall_confidence(self, patterns: List[FinancialPattern], data: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for the analysis"""
        if not patterns or not data:
            return 0.0
        
        # Confidence based on data volume and pattern consistency
        data_confidence = min(len(data) / 100, 1.0)  # More data = higher confidence
        pattern_confidence = sum(p.confidence for p in patterns) / len(patterns)
        
        return (data_confidence + pattern_confidence) / 2
    
    # Placeholder implementations for additional methods
    async def _analyze_behavioral_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Placeholder for behavioral pattern analysis"""
        return {"behavioral_score": 0.7, "insights": ["Behavioral analysis pending implementation"]}
    
    async def _assess_financial_risk(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Placeholder for risk assessment"""
        return {"risk_level": "medium", "risk_factors": ["Analysis pending"]}
    
    async def _generate_cognitive_recommendations(self, patterns, insights, risk) -> List[str]:
        """Generate cognitive recommendations based on analysis"""
        recommendations = []
        
        # Pattern-based recommendations
        for pattern in patterns:
            recommendations.extend(pattern.recommendations)
        
        # General cognitive recommendations
        recommendations.extend([
            "Review spending patterns monthly for optimal financial health",
            "Consider automating savings based on detected spending habits",
            "Monitor for anomalies in regular transaction patterns"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _get_expense_categories(self, data: List[Dict[str, Any]]) -> List[str]:
        """Get unique expense categories from data"""
        return list(set(t['category'] for t in data if t['category'] != 'income'))
    
    # Placeholder prediction methods
    async def _trend_based_prediction(self, data: List[Dict[str, Any]], months: int) -> PredictionResult:
        """Trend-based prediction model"""
        avg_amount = sum(t['amount'] for t in data) / len(data) if data else 0
        return PredictionResult(
            category=data[0]['category'] if data else 'unknown',
            predicted_amount=avg_amount,
            confidence_interval=(avg_amount * 0.8, avg_amount * 1.2),
            confidence_score=0.6,
            model_type="trend_based",
            factors=["historical_trend", "seasonal_adjustment"]
        )
    
    async def _seasonal_prediction(self, data: List[Dict[str, Any]], months: int) -> PredictionResult:
        """Seasonal prediction model"""
        avg_amount = sum(t['amount'] for t in data) / len(data) if data else 0
        return PredictionResult(
            category=data[0]['category'] if data else 'unknown',
            predicted_amount=avg_amount * 1.1,  # Slight seasonal adjustment
            confidence_interval=(avg_amount * 0.7, avg_amount * 1.4),
            confidence_score=0.7,
            model_type="seasonal",
            factors=["seasonal_patterns", "historical_data"]
        )
    
    async def _ml_based_prediction(self, data: List[Dict[str, Any]], months: int) -> PredictionResult:
        """Machine learning based prediction"""
        avg_amount = sum(t['amount'] for t in data) / len(data) if data else 0
        return PredictionResult(
            category=data[0]['category'] if data else 'unknown',
            predicted_amount=avg_amount * 0.95,
            confidence_interval=(avg_amount * 0.75, avg_amount * 1.15),
            confidence_score=0.8,
            model_type="ml_ensemble",
            factors=["feature_importance", "model_consensus"]
        )
    
    async def _ensemble_prediction(self, predictions: List[PredictionResult], category: str) -> PredictionResult:
        """Combine multiple predictions into ensemble result"""
        if not predictions:
            return PredictionResult(category, 0, (0, 0), 0, "ensemble", [])
        
        # Weighted average based on confidence scores
        total_weight = sum(p.confidence_score for p in predictions)
        if total_weight == 0:
            total_weight = len(predictions)
        
        weighted_amount = sum(p.predicted_amount * p.confidence_score for p in predictions) / total_weight
        avg_confidence = sum(p.confidence_score for p in predictions) / len(predictions)
        
        return PredictionResult(
            category=category,
            predicted_amount=weighted_amount,
            confidence_interval=(weighted_amount * 0.8, weighted_amount * 1.2),
            confidence_score=avg_confidence,
            model_type="ensemble",
            factors=["multi_model_consensus", "weighted_average"]
        )
    
    async def _statistical_anomaly_detection(self, data: List[Dict[str, Any]], sensitivity: float) -> List[Dict[str, Any]]:
        """Statistical anomaly detection using Z-score and IQR methods."""
        anomalies = []
        if not data:
            return anomalies

        amounts = np.array([t['amount'] for t in data])
        mean_amount = float(np.mean(amounts))
        std_amount = float(np.std(amounts))

        # IQR-based outlier bounds (robust to non-normal distributions)
        q1 = float(np.percentile(amounts, 25))
        q3 = float(np.percentile(amounts, 75))
        iqr = q3 - q1
        iqr_multiplier = 1.5 / sensitivity  # tighter with higher sensitivity
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        # Z-score threshold (adjustable via sensitivity)
        z_threshold = 3.0 / sensitivity

        for transaction in data:
            amount = transaction['amount']
            flags = []

            # Z-score check
            if std_amount > 0:
                z_score = abs(amount - mean_amount) / std_amount
                if z_score > z_threshold:
                    flags.append(('z_score', z_score))

            # IQR check
            if amount < lower_bound or amount > upper_bound:
                flags.append(('iqr_outlier', amount))

            if flags:
                methods = ', '.join(f[0] for f in flags)
                z_val = next((f[1] for f in flags if f[0] == 'z_score'), 0.0)
                severity = "high" if len(flags) >= 2 else "medium"
                anomalies.append({
                    "transaction": transaction,
                    "anomaly_type": "statistical_outlier",
                    "severity": severity,
                    "z_score": round(z_val, 2),
                    "iqr_bounds": (round(lower_bound, 2), round(upper_bound, 2)),
                    "detection_methods": methods,
                    "explanation": (
                        f"Amount ${amount:.2f} flagged by [{methods}]. "
                        f"Normal range: IQR [{lower_bound:.2f}, {upper_bound:.2f}]"
                    )
                })

        return anomalies

    async def _behavioral_anomaly_detection(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect behavioural anomalies: unusual category or new merchant."""
        if len(data) < 10:
            return []
        anomalies = []
        category_amounts: defaultdict = defaultdict(list)
        for t in data:
            category_amounts[t['category']].append(t['amount'])

        for t in data:
            cat = t['category']
            amounts_for_cat = category_amounts[cat]
            if len(amounts_for_cat) < 3:
                continue
            mean_cat = np.mean(amounts_for_cat)
            std_cat = np.std(amounts_for_cat)
            if std_cat > 0:
                z = abs(t['amount'] - mean_cat) / std_cat
                if z > 2.5:
                    anomalies.append({
                        "transaction": t,
                        "anomaly_type": "category_behaviour",
                        "severity": "medium",
                        "z_score": round(float(z), 2),
                        "explanation": (
                            f"Unusual {cat} amount ${t['amount']:.2f} "
                            f"(category mean ${mean_cat:.2f})"
                        )
                    })
        return anomalies

    async def _temporal_anomaly_detection(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect transactions outside normal time patterns (weekend spikes, etc.)."""
        if len(data) < 5:
            return []

        anomalies = []
        # Build daily totals
        daily_totals: defaultdict = defaultdict(float)
        for t in data:
            raw_date = t.get('date', '')[:10]
            daily_totals[raw_date] += t['amount']

        if len(daily_totals) < 3:
            return []

        daily_values = np.array(list(daily_totals.values()))
        mean_daily = float(np.mean(daily_values))
        std_daily = float(np.std(daily_values))

        for day, total in daily_totals.items():
            if std_daily > 0:
                z = abs(total - mean_daily) / std_daily
                if z > 2.0:
                    anomalies.append({
                        "date": day,
                        "anomaly_type": "daily_spending_spike",
                        "severity": "low",
                        "daily_total": round(total, 2),
                        "z_score": round(float(z), 2),
                        "explanation": (
                            f"Daily spend of ${total:.2f} on {day} "
                            f"is {z:.1f}σ above normal (mean ${mean_daily:.2f})"
                        )
                    })
        return anomalies

    async def _pattern_anomaly_detection(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect transactions that break established category-amount patterns."""
        if len(data) < 15:
            return []

        anomalies = []
        # Look for unusual category-description combinations (new merchant in known category)
        cat_descriptions: defaultdict = defaultdict(set)
        for t in data:
            cat_descriptions[t['category']].add(t['description'].lower()[:30])

        # Flag new descriptions in categories that have established patterns
        seen: defaultdict = defaultdict(set)
        for t in data:
            cat = t['category']
            desc = t['description'].lower()[:30]
            known = cat_descriptions[cat]
            if desc not in seen[cat] and len(known) >= 5:
                seen[cat].add(desc)
                # Only flag if the category has a well-established history
                anomalies.append({
                    "transaction": t,
                    "anomaly_type": "new_pattern",
                    "severity": "low",
                    "explanation": (
                        f"New transaction description in {cat} category: '{desc}'"
                    )
                })
                if len(anomalies) >= 5:
                    break  # Limit noise

        return anomalies

    def _categorize_anomaly_severity(self, anomalies: Dict[str, List]) -> Dict[str, int]:
        """Categorize anomalies by severity"""
        severity_count = {"high": 0, "medium": 0, "low": 0}
        
        for anomaly_list in anomalies.values():
            for anomaly in anomaly_list:
                severity = anomaly.get('severity', 'low')
                severity_count[severity] = severity_count.get(severity, 0) + 1
        
        return severity_count
    
    # Risk assessment methods (placeholders)
    async def _assess_liquidity_risk(self, data: List[Dict[str, Any]], income: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Assess liquidity risk based on income-to-expense ratio, emergency fund adequacy,
        and cash flow patterns.
        
        Returns a score between 0 (lowest risk) and 1 (highest risk).
        """
        if not data:
            return {"score": 0.5, "analysis": "Insufficient data for liquidity analysis", "metrics": {}}
        
        # Calculate total expenses
        total_expenses = sum(t['amount'] for t in data)
        monthly_avg_expenses = total_expenses / max(1, len(set(
            t['date'][:7] if isinstance(t['date'], str) else t['date'].strftime('%Y-%m') 
            for t in data
        )))
        
        # Calculate income if provided
        total_income = 0
        if income:
            total_income = sum(abs(t.get('amount', 0)) for t in income)
        else:
            # Estimate income from transactions (deposits/positive flows)
            # In a real scenario, we'd identify income transactions properly
            total_income = monthly_avg_expenses * 1.2  # Conservative estimate
        
        monthly_income = total_income / max(1, len(set(
            t['date'][:7] if isinstance(t.get('date', ''), str) else t.get('date', datetime.now()).strftime('%Y-%m')
            for t in (income or data)
        )))
        
        # Liquidity metrics
        expense_to_income_ratio = monthly_avg_expenses / max(monthly_income, 1)
        savings_rate = max(0, (monthly_income - monthly_avg_expenses) / max(monthly_income, 1))
        emergency_fund_months = savings_rate * 12 if savings_rate > 0 else 0
        
        # Calculate cash flow variability (high variability = higher risk)
        monthly_expenses = defaultdict(float)
        for t in data:
            month_key = t['date'][:7] if isinstance(t['date'], str) else t['date'].strftime('%Y-%m')
            monthly_expenses[month_key] += t['amount']
        
        expense_values = list(monthly_expenses.values())
        if len(expense_values) > 1:
            expense_std = np.std(expense_values)
            expense_mean = np.mean(expense_values)
            cash_flow_variability = expense_std / max(expense_mean, 1)
        else:
            cash_flow_variability = 0.3  # Default moderate variability
        
        # Calculate liquidity risk score
        # High expense-to-income ratio increases risk
        ratio_risk = min(expense_to_income_ratio, 1.5) / 1.5  # Cap at 1.5x
        # Low savings rate increases risk
        savings_risk = 1 - min(savings_rate, 0.3) / 0.3  # Target 30% savings
        # Low emergency fund increases risk
        emergency_risk = 1 - min(emergency_fund_months, 6) / 6  # Target 6 months
        # High variability increases risk
        variability_risk = min(cash_flow_variability, 1)
        
        # Weighted liquidity risk score
        liquidity_score = (
            ratio_risk * 0.35 +
            savings_risk * 0.25 +
            emergency_risk * 0.25 +
            variability_risk * 0.15
        )
        
        # Generate analysis summary
        concerns = []
        if expense_to_income_ratio > 0.9:
            concerns.append("Expense-to-income ratio is dangerously high")
        if savings_rate < 0.1:
            concerns.append("Savings rate below recommended minimum")
        if emergency_fund_months < 3:
            concerns.append("Emergency fund below 3-month threshold")
        if cash_flow_variability > 0.5:
            concerns.append("High cash flow variability indicates instability")
        
        return {
            "score": round(liquidity_score, 3),
            "analysis": "Liquidity risk based on income coverage, savings rate, and cash flow stability",
            "metrics": {
                "expense_to_income_ratio": round(expense_to_income_ratio, 3),
                "savings_rate": round(savings_rate, 3),
                "emergency_fund_months": round(emergency_fund_months, 1),
                "cash_flow_variability": round(cash_flow_variability, 3),
                "monthly_avg_expenses": round(monthly_avg_expenses, 2),
                "monthly_income_estimate": round(monthly_income, 2)
            },
            "concerns": concerns,
            "risk_level": self._categorize_risk_level(liquidity_score)
        }
    
    async def _assess_volatility_risk(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess spending volatility risk based on transaction amount variance,
        frequency patterns, and category-level volatility.
        
        Returns a score between 0 (stable) and 1 (highly volatile).
        """
        if not data or len(data) < 5:
            return {"score": 0.5, "analysis": "Insufficient data for volatility analysis", "metrics": {}}
        
        amounts = [t['amount'] for t in data]
        
        # Calculate basic volatility metrics
        mean_amount = np.mean(amounts)
        std_amount = np.std(amounts)
        cv = std_amount / max(mean_amount, 1)  # Coefficient of variation
        
        # Calculate interquartile range volatility
        q1, q3 = np.percentile(amounts, [25, 75])
        iqr = q3 - q1
        iqr_ratio = iqr / max(mean_amount, 1)
        
        # Calculate day-to-day volatility
        sorted_data = sorted(data, key=lambda x: x['date'])
        daily_totals = defaultdict(float)
        for t in sorted_data:
            day_key = t['date'][:10] if isinstance(t['date'], str) else t['date'].strftime('%Y-%m-%d')
            daily_totals[day_key] += t['amount']
        
        daily_values = list(daily_totals.values())
        if len(daily_values) > 1:
            daily_cv = np.std(daily_values) / max(np.mean(daily_values), 1)
        else:
            daily_cv = 0.3
        
        # Calculate category-level volatility
        category_volatility = {}
        category_data = defaultdict(list)
        for t in data:
            category_data[t['category']].append(t['amount'])
        
        cat_volatility_scores = []
        for category, cat_amounts in category_data.items():
            if len(cat_amounts) >= 3:
                cat_cv = np.std(cat_amounts) / max(np.mean(cat_amounts), 1)
                category_volatility[category] = round(cat_cv, 3)
                cat_volatility_scores.append(cat_cv)
        
        avg_category_volatility = np.mean(cat_volatility_scores) if cat_volatility_scores else 0.3
        
        # Calculate extreme transaction frequency (outliers)
        z_scores = np.abs((np.array(amounts) - mean_amount) / max(std_amount, 1))
        extreme_ratio = np.sum(z_scores > 2) / len(amounts)
        
        # Calculate overall volatility score
        # Normalize CV to 0-1 scale (CV > 1 is highly volatile)
        cv_risk = min(cv, 2) / 2
        iqr_risk = min(iqr_ratio, 2) / 2
        daily_risk = min(daily_cv, 2) / 2
        category_risk = min(avg_category_volatility, 1.5) / 1.5
        extreme_risk = min(extreme_ratio * 5, 1)  # Scale up extreme transaction impact
        
        volatility_score = (
            cv_risk * 0.25 +
            iqr_risk * 0.15 +
            daily_risk * 0.25 +
            category_risk * 0.20 +
            extreme_risk * 0.15
        )
        
        # Generate concerns
        concerns = []
        if cv > 1:
            concerns.append("High overall spending variance detected")
        if daily_cv > 0.8:
            concerns.append("Significant day-to-day spending fluctuations")
        if extreme_ratio > 0.1:
            concerns.append("Frequent extreme transactions (outliers)")
        
        # Find most volatile categories
        volatile_categories = [
            cat for cat, vol in category_volatility.items() if vol > 0.8
        ]
        if volatile_categories:
            concerns.append(f"High volatility in: {', '.join(volatile_categories[:3])}")
        
        return {
            "score": round(volatility_score, 3),
            "analysis": "Spending volatility based on amount variance, daily patterns, and category stability",
            "metrics": {
                "coefficient_of_variation": round(cv, 3),
                "iqr_ratio": round(iqr_ratio, 3),
                "daily_volatility": round(daily_cv, 3),
                "extreme_transaction_ratio": round(extreme_ratio, 3),
                "mean_transaction_amount": round(mean_amount, 2),
                "std_transaction_amount": round(std_amount, 2)
            },
            "category_volatility": category_volatility,
            "concerns": concerns,
            "risk_level": self._categorize_risk_level(volatility_score)
        }
    
    async def _assess_concentration_risk(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess category concentration risk using Herfindahl-Hirschman Index (HHI)
        and top-category dominance metrics.
        
        Returns a score between 0 (well-diversified) and 1 (highly concentrated).
        """
        if not data:
            return {"score": 0.5, "analysis": "Insufficient data for concentration analysis", "metrics": {}}
        
        # Calculate spending by category
        category_spending = defaultdict(float)
        for t in data:
            category_spending[t['category']] += t['amount']
        
        total_spending = sum(category_spending.values())
        if total_spending == 0:
            return {"score": 0.5, "analysis": "No spending to analyze", "metrics": {}}
        
        # Calculate market share (proportion) for each category
        category_shares = {
            cat: amount / total_spending 
            for cat, amount in category_spending.items()
        }
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        # HHI = sum of squared market shares
        # Range: 1/n (perfect distribution) to 1 (complete concentration)
        hhi = sum(share ** 2 for share in category_shares.values())
        
        # Normalized HHI (0 to 1 scale)
        n_categories = len(category_shares)
        min_hhi = 1 / n_categories if n_categories > 0 else 1
        normalized_hhi = (hhi - min_hhi) / (1 - min_hhi) if n_categories > 1 else 0
        
        # Top category concentration (CR1, CR3, CR5)
        sorted_shares = sorted(category_shares.values(), reverse=True)
        cr1 = sorted_shares[0] if len(sorted_shares) >= 1 else 0
        cr3 = sum(sorted_shares[:3]) if len(sorted_shares) >= 3 else sum(sorted_shares)
        cr5 = sum(sorted_shares[:5]) if len(sorted_shares) >= 5 else sum(sorted_shares)
        
        # Category diversity score (entropy-based)
        entropy = -sum(
            share * np.log2(share) if share > 0 else 0 
            for share in category_shares.values()
        )
        max_entropy = np.log2(n_categories) if n_categories > 1 else 1
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0
        
        # Calculate concentration risk score
        hhi_risk = normalized_hhi
        cr1_risk = cr1  # Single category dominance
        diversity_risk = 1 - diversity_score
        
        concentration_score = (
            hhi_risk * 0.4 +
            cr1_risk * 0.35 +
            diversity_risk * 0.25
        )
        
        # Identify dominant categories
        dominant_categories = [
            {"category": cat, "share": round(share, 3), "amount": round(category_spending[cat], 2)}
            for cat, share in sorted(category_shares.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        # Generate concerns
        concerns = []
        if cr1 > 0.5:
            top_cat = max(category_shares, key=category_shares.get)
            concerns.append(f"Over 50% spending concentrated in '{top_cat}'")
        if normalized_hhi > 0.6:
            concerns.append("High spending concentration (HHI indicates oligopoly-like pattern)")
        if n_categories < 5:
            concerns.append(f"Limited category diversity (only {n_categories} categories)")
        if diversity_score < 0.5:
            concerns.append("Low spending diversity - consider broader expense distribution")
        
        return {
            "score": round(concentration_score, 3),
            "analysis": "Concentration risk based on HHI, category dominance, and diversity metrics",
            "metrics": {
                "hhi": round(hhi, 4),
                "normalized_hhi": round(normalized_hhi, 3),
                "cr1_top_category": round(cr1, 3),
                "cr3_top_three": round(cr3, 3),
                "cr5_top_five": round(cr5, 3),
                "diversity_score": round(diversity_score, 3),
                "num_categories": n_categories,
                "total_spending": round(total_spending, 2)
            },
            "dominant_categories": dominant_categories,
            "concerns": concerns,
            "risk_level": self._categorize_risk_level(concentration_score)
        }
    
    async def _assess_behavioral_risk(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess behavioral spending risk based on impulse patterns, timing anomalies,
        weekend/late-night spending, and spending acceleration trends.
        
        Returns a score between 0 (disciplined) and 1 (high-risk behavior).
        """
        if not data or len(data) < 10:
            return {"score": 0.5, "analysis": "Insufficient data for behavioral analysis", "metrics": {}}
        
        # Parse dates and extract behavioral features
        transactions_with_dt = []
        for t in data:
            try:
                if isinstance(t['date'], str):
                    dt = datetime.fromisoformat(t['date'].replace('Z', '+00:00').split('+')[0])
                else:
                    dt = t['date']
                transactions_with_dt.append({**t, 'datetime': dt})
            except (ValueError, TypeError):
                continue
        
        if len(transactions_with_dt) < 10:
            return {"score": 0.5, "analysis": "Unable to parse sufficient transaction dates", "metrics": {}}
        
        # 1. Impulse spending detection (small frequent transactions)
        amounts = [t['amount'] for t in transactions_with_dt]
        median_amount = np.median(amounts)
        small_threshold = median_amount * 0.3
        small_transactions = [t for t in transactions_with_dt if t['amount'] < small_threshold]
        small_transaction_ratio = len(small_transactions) / len(transactions_with_dt)
        
        # Check for clustering of small transactions (impulse bursts)
        sorted_small = sorted(small_transactions, key=lambda x: x['datetime'])
        impulse_bursts = 0
        for i in range(len(sorted_small) - 2):
            time_diff1 = (sorted_small[i+1]['datetime'] - sorted_small[i]['datetime']).total_seconds() / 3600
            time_diff2 = (sorted_small[i+2]['datetime'] - sorted_small[i+1]['datetime']).total_seconds() / 3600
            if time_diff1 < 24 and time_diff2 < 24:  # 3 small transactions within 24 hours
                impulse_bursts += 1
        
        impulse_burst_ratio = impulse_bursts / max(len(small_transactions) - 2, 1) if small_transactions else 0
        
        # 2. Weekend spending pattern (potentially discretionary)
        weekend_spending = sum(
            t['amount'] for t in transactions_with_dt 
            if t['datetime'].weekday() >= 5  # Saturday = 5, Sunday = 6
        )
        total_spending = sum(t['amount'] for t in transactions_with_dt)
        weekend_ratio = weekend_spending / max(total_spending, 1)
        # Expected weekend ratio is ~28.6% (2/7 days), higher indicates discretionary spending
        weekend_excess = max(0, weekend_ratio - 0.286) / 0.714  # Normalize to 0-1
        
        # 3. Late-night spending (potentially impulsive)
        if all('datetime' in t for t in transactions_with_dt):
            late_night_transactions = [
                t for t in transactions_with_dt 
                if hasattr(t['datetime'], 'hour') and (t['datetime'].hour >= 22 or t['datetime'].hour < 6)
            ]
            late_night_ratio = len(late_night_transactions) / len(transactions_with_dt)
            late_night_amount_ratio = sum(t['amount'] for t in late_night_transactions) / max(total_spending, 1)
        else:
            late_night_ratio = 0
            late_night_amount_ratio = 0
        
        # 4. Spending acceleration (month-over-month growth)
        monthly_spending = defaultdict(float)
        for t in transactions_with_dt:
            month_key = t['datetime'].strftime('%Y-%m')
            monthly_spending[month_key] += t['amount']
        
        sorted_months = sorted(monthly_spending.keys())
        if len(sorted_months) >= 3:
            monthly_values = [monthly_spending[m] for m in sorted_months]
            # Calculate average month-over-month growth rate
            growth_rates = []
            for i in range(1, len(monthly_values)):
                if monthly_values[i-1] > 0:
                    growth_rates.append((monthly_values[i] - monthly_values[i-1]) / monthly_values[i-1])
            avg_growth_rate = np.mean(growth_rates) if growth_rates else 0
            spending_acceleration = max(0, min(avg_growth_rate, 0.5)) / 0.5  # Cap at 50% growth
        else:
            spending_acceleration = 0
        
        # 5. Transaction frequency irregularity
        daily_counts = defaultdict(int)
        for t in transactions_with_dt:
            day_key = t['datetime'].strftime('%Y-%m-%d')
            daily_counts[day_key] += 1
        
        count_values = list(daily_counts.values())
        if len(count_values) > 1:
            frequency_cv = np.std(count_values) / max(np.mean(count_values), 1)
        else:
            frequency_cv = 0.3
        
        # Calculate behavioral risk score
        impulse_risk = small_transaction_ratio * 0.5 + impulse_burst_ratio * 0.5
        timing_risk = weekend_excess * 0.4 + late_night_ratio * 0.3 + late_night_amount_ratio * 0.3
        acceleration_risk = spending_acceleration
        irregularity_risk = min(frequency_cv, 1.5) / 1.5
        
        behavioral_score = (
            impulse_risk * 0.30 +
            timing_risk * 0.25 +
            acceleration_risk * 0.25 +
            irregularity_risk * 0.20
        )
        
        # Generate concerns
        concerns = []
        if small_transaction_ratio > 0.4:
            concerns.append("High frequency of small transactions suggests impulse spending")
        if impulse_burst_ratio > 0.3:
            concerns.append("Detected clusters of impulse purchases")
        if weekend_excess > 0.3:
            concerns.append("Weekend spending significantly exceeds weekday patterns")
        if late_night_ratio > 0.1:
            concerns.append("Notable late-night transaction activity")
        if spending_acceleration > 0.5:
            concerns.append("Spending is accelerating month-over-month")
        if frequency_cv > 1:
            concerns.append("Highly irregular transaction frequency patterns")
        
        return {
            "score": round(behavioral_score, 3),
            "analysis": "Behavioral risk based on impulse patterns, timing, and spending trends",
            "metrics": {
                "small_transaction_ratio": round(small_transaction_ratio, 3),
                "impulse_burst_ratio": round(impulse_burst_ratio, 3),
                "weekend_spending_ratio": round(weekend_ratio, 3),
                "weekend_excess": round(weekend_excess, 3),
                "late_night_transaction_ratio": round(late_night_ratio, 3),
                "spending_acceleration": round(spending_acceleration, 3),
                "frequency_irregularity": round(frequency_cv, 3)
            },
            "behavioral_patterns": {
                "impulse_tendency": "high" if impulse_risk > 0.6 else "moderate" if impulse_risk > 0.3 else "low",
                "timing_discipline": "poor" if timing_risk > 0.5 else "moderate" if timing_risk > 0.25 else "good",
                "spending_trend": "accelerating" if spending_acceleration > 0.3 else "stable" if spending_acceleration < 0.1 else "growing"
            },
            "concerns": concerns,
            "risk_level": self._categorize_risk_level(behavioral_score)
        }
    
    def _categorize_risk_level(self, score: float) -> str:
        """Categorize overall risk level"""
        if score < 0.3:
            return "low"
        elif score < 0.7:
            return "medium"
        else:
            return "high"
    
    def _identify_primary_risk_concerns(self, risk_scores: Dict[str, float]) -> List[str]:
        """Identify primary risk concerns"""
        concerns = []
        threshold = 0.6
        
        for risk_type, score in risk_scores.items():
            if score > threshold:
                concerns.append(risk_type.replace('_', ' ').title())
        
        return concerns if concerns else ["No major risk concerns identified"]
    
    async def _generate_risk_mitigation_strategies(self, risk_scores: Dict[str, float]) -> List[str]:
        """Generate risk mitigation strategies"""
        strategies = []
        
        if risk_scores.get('liquidity_risk', 0) > 0.5:
            strategies.append("Build emergency fund to improve liquidity")
        
        if risk_scores.get('volatility_risk', 0) > 0.5:
            strategies.append("Establish more consistent spending patterns")
        
        if risk_scores.get('concentration_risk', 0) > 0.5:
            strategies.append("Diversify spending across categories")
        
        if risk_scores.get('behavioral_risk', 0) > 0.5:
            strategies.append("Implement spending awareness and control mechanisms")
        
        return strategies if strategies else ["Continue current financial practices"]