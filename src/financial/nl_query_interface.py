"""
Natural Language Query Interface for Financial Data
Allows users to query financial information using natural language
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of financial queries"""
    BALANCE = "balance"
    SPENDING = "spending"
    INCOME = "income"
    COMPARISON = "comparison"
    TREND = "trend"
    CATEGORY = "category"
    SEARCH = "search"
    SUMMARY = "summary"
    PREDICTION = "prediction"


@dataclass
class QueryIntent:
    """Parsed intent from natural language query"""
    query_type: QueryType
    entities: Dict[str, Any]
    timeframe: Optional[Tuple[datetime, datetime]]
    confidence: float
    parameters: Dict[str, Any]


class NaturalLanguageQueryEngine:
    """
    Natural language interface for querying financial data
    """
    
    def __init__(self, gnucash_access=None, memory_manager=None):
        self.gnucash_access = gnucash_access
        self.memory_manager = memory_manager
        self.query_patterns = self._initialize_patterns()
        self.entity_extractors = self._initialize_entity_extractors()
        
    def _initialize_patterns(self) -> Dict[QueryType, List[str]]:
        """Initialize query patterns for intent recognition"""
        return {
            QueryType.BALANCE: [
                r'what.*balance', r'how much.*have', r'current balance',
                r'account balance', r'balance.*account'
            ],
            QueryType.SPENDING: [
                r'how much.*spend', r'spent.*on', r'spending.*', 
                r'expenses.*', r'what.*cost'
            ],
            QueryType.INCOME: [
                r'income', r'earn', r'revenue', r'salary', r'paycheck'
            ],
            QueryType.COMPARISON: [
                r'compare', r'versus', r'vs', r'difference between',
                r'more than', r'less than'
            ],
            QueryType.TREND: [
                r'trend', r'over time', r'pattern', r'increasing',
                r'decreasing', r'growing'
            ],
            QueryType.CATEGORY: [
                r'category', r'type of', r'categorize', r'breakdown'
            ],
            QueryType.SEARCH: [
                r'find', r'search', r'look for', r'show.*transaction'
            ],
            QueryType.SUMMARY: [
                r'summary', r'overview', r'report', r'total'
            ],
            QueryType.PREDICTION: [
                r'predict', r'forecast', r'expect', r'will.*spend',
                r'future', r'next month'
            ]
        }
    
    def _initialize_entity_extractors(self) -> Dict[str, Any]:
        """Initialize entity extraction patterns"""
        return {
            'amount': r'\$?\d+(?:,\d{3})*(?:\.\d{2})?',
            'category': r'(?:groceries|dining|transportation|utilities|healthcare|entertainment|shopping|travel)',
            'timeframe': {
                'today': (lambda: (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0), datetime.now())),
                'yesterday': (lambda: (datetime.now() - timedelta(days=1), datetime.now() - timedelta(days=1))),
                'this week': (lambda: (datetime.now() - timedelta(days=datetime.now().weekday()), datetime.now())),
                'this month': (lambda: (datetime.now().replace(day=1), datetime.now())),
                'last month': (lambda: ((datetime.now().replace(day=1) - timedelta(days=1)).replace(day=1), datetime.now().replace(day=1) - timedelta(days=1))),
                'this year': (lambda: (datetime.now().replace(month=1, day=1), datetime.now())),
                'last year': (lambda: ((datetime.now().replace(month=1, day=1) - timedelta(days=1)).replace(month=1, day=1), datetime.now().replace(month=1, day=1) - timedelta(days=1)))
            },
            'account': r'(?:checking|savings|credit card|account)'
        }
    
    async def parse_query(self, query: str) -> QueryIntent:
        """
        Parse natural language query into structured intent
        
        Args:
            query: Natural language query string
            
        Returns:
            QueryIntent with parsed information
        """
        query_lower = query.lower()
        
        # Detect query type
        query_type = await self._detect_query_type(query_lower)
        
        # Extract entities
        entities = await self._extract_entities(query_lower)
        
        # Extract timeframe
        timeframe = await self._extract_timeframe(query_lower)
        
        # Extract additional parameters
        parameters = await self._extract_parameters(query_lower, query_type)
        
        # Calculate confidence
        confidence = self._calculate_intent_confidence(query_type, entities, timeframe)
        
        return QueryIntent(
            query_type=query_type,
            entities=entities,
            timeframe=timeframe,
            confidence=confidence,
            parameters=parameters
        )
    
    async def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query"""
        scores = {}
        
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query):
                    score += 1
            scores[query_type] = score
        
        if max(scores.values()) == 0:
            return QueryType.SEARCH  # Default
        
        return max(scores, key=scores.get)
    
    async def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from query"""
        entities = {}
        
        # Extract amounts
        amount_pattern = self.entity_extractors['amount']
        amounts = re.findall(amount_pattern, query)
        if amounts:
            entities['amounts'] = [float(a.replace('$', '').replace(',', '')) for a in amounts]
        
        # Extract categories
        category_pattern = self.entity_extractors['category']
        categories = re.findall(category_pattern, query)
        if categories:
            entities['categories'] = list(set(categories))
        
        # Extract accounts
        account_pattern = self.entity_extractors['account']
        accounts = re.findall(account_pattern, query)
        if accounts:
            entities['accounts'] = list(set(accounts))
        
        return entities
    
    async def _extract_timeframe(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """Extract timeframe from query"""
        timeframe_patterns = self.entity_extractors['timeframe']
        
        for phrase, generator in timeframe_patterns.items():
            if phrase in query:
                return generator()
        
        # Check for specific date patterns
        # Format: "in January", "in 2023", "from X to Y"
        
        return None  # No timeframe specified
    
    async def _extract_parameters(self, query: str, query_type: QueryType) -> Dict[str, Any]:
        """Extract additional parameters based on query type"""
        parameters = {}
        
        if query_type == QueryType.COMPARISON:
            # Extract comparison operator
            if 'more than' in query or 'greater than' in query:
                parameters['operator'] = '>'
            elif 'less than' in query or 'fewer than' in query:
                parameters['operator'] = '<'
            else:
                parameters['operator'] = '=='
        
        elif query_type == QueryType.TREND:
            # Extract trend direction
            if 'increasing' in query or 'growing' in query:
                parameters['direction'] = 'up'
            elif 'decreasing' in query or 'declining' in query:
                parameters['direction'] = 'down'
        
        elif query_type == QueryType.SUMMARY:
            # Extract aggregation type
            if 'total' in query or 'sum' in query:
                parameters['aggregation'] = 'sum'
            elif 'average' in query or 'mean' in query:
                parameters['aggregation'] = 'average'
            elif 'count' in query or 'number' in query:
                parameters['aggregation'] = 'count'
        
        return parameters
    
    def _calculate_intent_confidence(self, query_type: QueryType, 
                                     entities: Dict, timeframe: Optional[Tuple]) -> float:
        """Calculate confidence in parsed intent"""
        confidence = 0.5  # Base confidence
        
        # Boost for recognized entities
        if entities:
            confidence += 0.2
        
        # Boost for timeframe
        if timeframe:
            confidence += 0.2
        
        # Boost for specific query types
        if query_type in [QueryType.BALANCE, QueryType.SPENDING, QueryType.INCOME]:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def execute_query(self, query: str) -> Dict[str, Any]:
        """
        Execute natural language query and return results
        
        Args:
            query: Natural language query
            
        Returns:
            Query results with data and metadata
        """
        # Parse query
        intent = await self.parse_query(query)
        
        if intent.confidence < 0.5:
            return {
                'success': False,
                'error': 'Could not understand query',
                'suggestion': 'Try rephrasing your question',
                'confidence': intent.confidence
            }
        
        # Execute based on query type
        try:
            if intent.query_type == QueryType.BALANCE:
                result = await self._execute_balance_query(intent)
            elif intent.query_type == QueryType.SPENDING:
                result = await self._execute_spending_query(intent)
            elif intent.query_type == QueryType.INCOME:
                result = await self._execute_income_query(intent)
            elif intent.query_type == QueryType.COMPARISON:
                result = await self._execute_comparison_query(intent)
            elif intent.query_type == QueryType.TREND:
                result = await self._execute_trend_query(intent)
            elif intent.query_type == QueryType.CATEGORY:
                result = await self._execute_category_query(intent)
            elif intent.query_type == QueryType.SEARCH:
                result = await self._execute_search_query(intent)
            elif intent.query_type == QueryType.SUMMARY:
                result = await self._execute_summary_query(intent)
            elif intent.query_type == QueryType.PREDICTION:
                result = await self._execute_prediction_query(intent)
            else:
                result = {'error': 'Unknown query type'}
            
            return {
                'success': True,
                'query_type': intent.query_type.value,
                'result': result,
                'confidence': intent.confidence,
                'entities': intent.entities,
                'timeframe': intent.timeframe
            }
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {
                'success': False,
                'error': str(e),
                'confidence': intent.confidence
            }
    
    async def _execute_balance_query(self, intent: QueryIntent) -> Dict[str, Any]:
        """Execute balance query against GnuCash data."""
        if self.gnucash_access and self.gnucash_access.initialized:
            accounts = await self.gnucash_access.get_accounts()
            result_accounts = []
            total = 0.0
            for acct in accounts:
                if acct['account_type'] in ('BANK', 'ASSET', 'CASH', 'CREDIT'):
                    balance = float(
                        await self.gnucash_access.get_account_balance(acct['guid'])
                    )
                    result_accounts.append({'name': acct['name'], 'balance': balance})
                    total += balance
            return {
                'accounts': result_accounts,
                'total_balance': round(total, 2),
                'currency': 'USD'
            }
        # Mock fallback
        return {
            'accounts': [
                {'name': 'Checking', 'balance': 5234.56},
                {'name': 'Savings', 'balance': 12500.00}
            ],
            'total_balance': 17734.56,
            'currency': 'USD',
            'note': 'mock data — no GnuCash connection'
        }

    async def _execute_spending_query(self, intent: QueryIntent) -> Dict[str, Any]:
        """Execute spending query against GnuCash data."""
        if self.gnucash_access and self.gnucash_access.initialized:
            from datetime import date as _date
            if intent.timeframe:
                start = _date.fromisoformat(intent.timeframe[0].date().isoformat())
                end = _date.fromisoformat(intent.timeframe[1].date().isoformat())
            else:
                from datetime import timedelta as _td
                end = _date.today()
                start = _date(end.year, end.month, 1)

            requested_categories = set(intent.entities.get('categories', []))
            spending = await self.gnucash_access.get_spending_by_category(start, end)
            if requested_categories:
                spending = {
                    k: v for k, v in spending.items()
                    if k.lower() in requested_categories
                }
            return {
                'total_spending': round(sum(float(v) for v in spending.values()), 2),
                'category_breakdown': {k: round(float(v), 2) for k, v in spending.items()},
                'timeframe': f"{start} to {end}",
                'currency': 'USD'
            }
        # Mock fallback
        return {
            'total_spending': 1450.32,
            'category_breakdown': {
                'groceries': 450.00,
                'dining': 300.00,
                'transportation': 200.00,
                'utilities': 500.32
            },
            'timeframe': 'this month',
            'currency': 'USD',
            'note': 'mock data — no GnuCash connection'
        }

    async def _execute_income_query(self, intent: QueryIntent) -> Dict[str, Any]:
        """Execute income query against GnuCash data."""
        if self.gnucash_access and self.gnucash_access.initialized:
            from datetime import date as _date
            if intent.timeframe:
                start = _date.fromisoformat(intent.timeframe[0].date().isoformat())
                end = _date.fromisoformat(intent.timeframe[1].date().isoformat())
            else:
                from datetime import timedelta as _td
                end = _date.today()
                start = _date(end.year, end.month, 1)
            income = await self.gnucash_access.get_income_by_category(start, end)
            return {
                'total_income': round(sum(float(v) for v in income.values()), 2),
                'sources': [{'name': k, 'amount': round(float(v), 2)} for k, v in income.items()],
                'timeframe': f"{start} to {end}",
                'currency': 'USD'
            }
        return {
            'total_income': 5000.00,
            'sources': [
                {'name': 'Salary', 'amount': 4500.00},
                {'name': 'Freelance', 'amount': 500.00}
            ],
            'timeframe': 'this month',
            'currency': 'USD',
            'note': 'mock data — no GnuCash connection'
        }

    async def _execute_search_query(self, intent: QueryIntent) -> Dict[str, Any]:
        """Execute search query against GnuCash data."""
        search_term = intent.entities.get('categories', [''])[0] if intent.entities.get('categories') else ''
        if not search_term:
            # Try extracting from raw query via memory
            search_term = intent.parameters.get('search_term', '')

        if self.gnucash_access and self.gnucash_access.initialized and search_term:
            rows = await self.gnucash_access.search_transactions(search_term)
            transactions = [
                {
                    'date': r['post_date'],
                    'description': r['description'],
                    'amount': float(r['amount'])
                }
                for r in rows
            ]
            return {'transactions': transactions, 'count': len(transactions)}

        return {
            'transactions': [
                {'date': '2024-01-15', 'description': 'Grocery Store', 'amount': 45.23},
                {'date': '2024-01-12', 'description': 'Gas Station', 'amount': 40.00}
            ],
            'count': 2,
            'note': 'mock data — no GnuCash connection'
        }

    async def _execute_summary_query(self, intent: QueryIntent) -> Dict[str, Any]:
        """Execute summary query against GnuCash data."""
        if self.gnucash_access and self.gnucash_access.initialized:
            from datetime import date as _date
            if intent.timeframe:
                start = _date.fromisoformat(intent.timeframe[0].date().isoformat())
                end = _date.fromisoformat(intent.timeframe[1].date().isoformat())
            else:
                from datetime import timedelta as _td
                end = _date.today()
                start = _date(end.year, end.month, 1)
            spending = await self.gnucash_access.get_spending_by_category(start, end)
            income = await self.gnucash_access.get_income_by_category(start, end)
            total_expenses = sum(float(v) for v in spending.values())
            total_income = sum(float(v) for v in income.values())
            txns = await self.gnucash_access.get_transactions(
                start_date=start, end_date=end, limit=500
            )
            top_cats = sorted(spending.items(), key=lambda x: x[1], reverse=True)[:3]
            return {
                'period': f"{start} to {end}",
                'total_income': round(total_income, 2),
                'total_expenses': round(total_expenses, 2),
                'net': round(total_income - total_expenses, 2),
                'transactions': len(txns),
                'top_categories': [c for c, _ in top_cats]
            }
        return {
            'period': 'this month',
            'total_income': 5000.00,
            'total_expenses': 1450.32,
            'net': 3549.68,
            'transactions': 45,
            'top_categories': ['groceries', 'dining', 'transportation'],
            'note': 'mock data — no GnuCash connection'
        }
    
    async def get_query_suggestions(self, partial_query: str) -> List[str]:
        """Get query suggestions based on partial input"""
        suggestions = [
            "How much did I spend on groceries this month?",
            "What's my current checking account balance?",
            "Show me all transactions over $100",
            "Compare my spending this month to last month",
            "What's my income for this year?",
            "Predict my expenses for next month",
            "Show spending trends over time",
            "Summarize my finances for this month"
        ]
        
        # Filter based on partial query
        if partial_query:
            query_lower = partial_query.lower()
            suggestions = [s for s in suggestions if query_lower in s.lower()]
        
        return suggestions[:5]
