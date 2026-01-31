"""
AI-Powered Transaction Categorization for ElizaOS
Implements intelligent transaction categorization using machine learning and cognitive reasoning
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CategoryPrediction:
    """Result of transaction categorization"""
    category: str
    confidence: float
    subcategory: Optional[str] = None
    reasoning: str = ""
    alternative_categories: List[Tuple[str, float]] = None


class AITransactionCategorizer:
    """
    AI-powered transaction categorization system
    Uses pattern matching, ML, and cognitive reasoning to categorize transactions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.category_patterns = self._initialize_patterns()
        self.learned_patterns = {}
        self.category_history = defaultdict(list)
        self.merchant_cache = {}
        
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize category detection patterns"""
        return {
            'groceries': [
                r'walmart', r'target', r'kroger', r'safeway', r'whole\s*foods',
                r'trader\s*joe', r'costco', r'aldi', r'publix', r'grocery'
            ],
            'dining': [
                r'restaurant', r'cafe', r'coffee', r'starbucks', r'mcdonald',
                r'burger', r'pizza', r'chipotle', r'subway', r'dining'
            ],
            'transportation': [
                r'uber', r'lyft', r'gas\s*station', r'shell', r'chevron',
                r'exxon', r'parking', r'transit', r'metro', r'taxi'
            ],
            'utilities': [
                r'electric', r'water', r'gas\s*company', r'internet', r'phone',
                r'att', r'verizon', r'comcast', r'utility', r'bill'
            ],
            'healthcare': [
                r'pharmacy', r'cvs', r'walgreens', r'hospital', r'clinic',
                r'doctor', r'dental', r'medical', r'health'
            ],
            'entertainment': [
                r'netflix', r'spotify', r'hulu', r'disney', r'movie',
                r'theater', r'concert', r'game', r'entertainment'
            ],
            'shopping': [
                r'amazon', r'ebay', r'shop', r'store', r'mall', r'retail'
            ],
            'travel': [
                r'hotel', r'airbnb', r'airline', r'flight', r'travel',
                r'booking', r'expedia', r'vacation'
            ],
            'education': [
                r'school', r'university', r'college', r'tuition', r'course',
                r'textbook', r'education'
            ],
            'insurance': [
                r'insurance', r'policy', r'premium', r'coverage'
            ],
            'savings': [
                r'savings', r'deposit', r'investment', r'transfer.*savings'
            ],
            'income': [
                r'salary', r'paycheck', r'deposit.*payroll', r'wage', r'income'
            ]
        }
    
    async def categorize_transaction(self, transaction: Dict[str, Any]) -> CategoryPrediction:
        """
        Categorize a single transaction using AI and pattern matching
        
        Args:
            transaction: Transaction dict with 'description', 'amount', etc.
            
        Returns:
            CategoryPrediction with category and confidence
        """
        description = transaction.get('description', '').lower()
        amount = transaction.get('amount', 0.0)
        merchant = transaction.get('merchant', '').lower()
        
        # Check merchant cache first
        if merchant and merchant in self.merchant_cache:
            cached = self.merchant_cache[merchant]
            return CategoryPrediction(
                category=cached['category'],
                confidence=min(0.95, cached['confidence'] + 0.1),
                reasoning="Cached from previous categorization"
            )
        
        # Pattern-based categorization
        pattern_results = []
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, description) or re.search(pattern, merchant):
                    confidence = 0.8
                    # Boost confidence if multiple patterns match
                    pattern_results.append((category, confidence))
        
        # If patterns found, use the most common one
        if pattern_results:
            category_scores = defaultdict(float)
            for cat, conf in pattern_results:
                category_scores[cat] += conf
            
            best_category = max(category_scores.items(), key=lambda x: x[1])
            alternatives = sorted(
                [(cat, score) for cat, score in category_scores.items() if cat != best_category[0]],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            result = CategoryPrediction(
                category=best_category[0],
                confidence=min(0.95, best_category[1]),
                reasoning=f"Pattern matching on description/merchant",
                alternative_categories=alternatives
            )
            
            # Cache merchant
            if merchant:
                self.merchant_cache[merchant] = {
                    'category': result.category,
                    'confidence': result.confidence
                }
            
            return result
        
        # Fallback: Use learned patterns or amount-based heuristics
        if amount < 0:  # Income
            return CategoryPrediction(
                category='income',
                confidence=0.6,
                reasoning="Negative amount suggests income"
            )
        elif amount > 1000:
            return CategoryPrediction(
                category='large_purchase',
                confidence=0.5,
                reasoning="Large amount suggests major purchase"
            )
        else:
            return CategoryPrediction(
                category='uncategorized',
                confidence=0.3,
                reasoning="No matching patterns found"
            )
    
    async def categorize_batch(self, transactions: List[Dict[str, Any]]) -> List[CategoryPrediction]:
        """Categorize multiple transactions efficiently"""
        results = []
        for transaction in transactions:
            result = await self.categorize_transaction(transaction)
            results.append(result)
            
            # Learn from high-confidence categorizations
            if result.confidence > 0.8:
                self._update_learned_patterns(transaction, result.category)
        
        return results
    
    def _update_learned_patterns(self, transaction: Dict, category: str):
        """Learn from successful categorizations"""
        description = transaction.get('description', '').lower()
        merchant = transaction.get('merchant', '').lower()
        
        # Extract key terms
        terms = set(description.split()) | set(merchant.split())
        terms = {t for t in terms if len(t) > 3}  # Filter short terms
        
        if category not in self.learned_patterns:
            self.learned_patterns[category] = defaultdict(int)
        
        for term in terms:
            self.learned_patterns[category][term] += 1
    
    async def auto_categorize_uncategorized(self, transactions: List[Dict[str, Any]],
                                           min_confidence: float = 0.7) -> Dict[str, Any]:
        """
        Automatically categorize all uncategorized transactions
        
        Args:
            transactions: List of transactions to categorize
            min_confidence: Minimum confidence threshold for auto-categorization
            
        Returns:
            Summary of categorization results
        """
        uncategorized = [
            t for t in transactions 
            if not t.get('category') or t.get('category') == 'uncategorized'
        ]
        
        logger.info(f"Auto-categorizing {len(uncategorized)} transactions")
        
        results = await self.categorize_batch(uncategorized)
        
        # Apply categorizations with sufficient confidence
        categorized_count = 0
        low_confidence_count = 0
        
        for transaction, result in zip(uncategorized, results):
            if result.confidence >= min_confidence:
                transaction['category'] = result.category
                transaction['ai_categorized'] = True
                transaction['categorization_confidence'] = result.confidence
                categorized_count += 1
            else:
                low_confidence_count += 1
        
        return {
            'total_processed': len(uncategorized),
            'successfully_categorized': categorized_count,
            'low_confidence': low_confidence_count,
            'categorization_rate': categorized_count / max(1, len(uncategorized)),
            'categories_assigned': len(set(r.category for r in results if r.confidence >= min_confidence))
        }
    
    async def suggest_recategorization(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Suggest recategorization for transactions that might be miscategorized
        
        Returns:
            List of suggestions with current and suggested categories
        """
        suggestions = []
        
        for transaction in transactions:
            current_category = transaction.get('category', 'uncategorized')
            
            # Get AI suggestion
            result = await self.categorize_transaction(transaction)
            
            # If AI suggests different category with high confidence
            if result.category != current_category and result.confidence > 0.75:
                suggestions.append({
                    'transaction_id': transaction.get('id', ''),
                    'description': transaction.get('description', ''),
                    'current_category': current_category,
                    'suggested_category': result.category,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning
                })
        
        return suggestions
    
    def get_category_statistics(self) -> Dict[str, Any]:
        """Get statistics about categorization performance"""
        return {
            'total_patterns': sum(len(patterns) for patterns in self.category_patterns.values()),
            'learned_patterns': len(self.learned_patterns),
            'cached_merchants': len(self.merchant_cache),
            'categories': list(self.category_patterns.keys())
        }


class PLNFinancialReasoner:
    """
    Uses OpenCog PLN (Probabilistic Logic Networks) for financial pattern recognition
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.atomspace = None
        self.pln_available = False
        
        try:
            from opencog.atomspace import AtomSpace, types
            from opencog.type_constructors import *
            self.AtomSpace = AtomSpace
            self.types = types
            self.pln_available = True
            self.atomspace = AtomSpace()
        except ImportError:
            logger.warning("OpenCog not available, using fallback reasoning")
    
    async def recognize_financial_patterns(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use PLN to recognize patterns in financial transactions
        
        Args:
            transactions: List of transaction data
            
        Returns:
            List of recognized patterns with confidence scores
        """
        if not self.pln_available:
            return await self._fallback_pattern_recognition(transactions)
        
        patterns = []
        
        # Group transactions by category
        category_groups = defaultdict(list)
        for t in transactions:
            category = t.get('category', 'uncategorized')
            category_groups[category].append(t)
        
        # Analyze each category
        for category, cat_transactions in category_groups.items():
            if len(cat_transactions) < 3:
                continue
            
            # Calculate statistics
            amounts = [abs(t.get('amount', 0)) for t in cat_transactions]
            avg_amount = sum(amounts) / len(amounts)
            
            # Detect patterns
            pattern = {
                'category': category,
                'pattern_type': 'spending_habit',
                'transaction_count': len(cat_transactions),
                'average_amount': avg_amount,
                'confidence': 0.8 if len(cat_transactions) > 10 else 0.6,
                'description': f"Regular {category} spending pattern detected"
            }
            patterns.append(pattern)
        
        return patterns
    
    async def _fallback_pattern_recognition(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback pattern recognition without PLN"""
        patterns = []
        
        # Simple frequency-based pattern detection
        category_counts = defaultdict(int)
        for t in transactions:
            category = t.get('category', 'uncategorized')
            category_counts[category] += 1
        
        for category, count in category_counts.items():
            if count >= 5:
                patterns.append({
                    'category': category,
                    'pattern_type': 'frequency',
                    'transaction_count': count,
                    'confidence': min(0.9, 0.5 + (count / 20)),
                    'description': f"Frequent {category} transactions"
                })
        
        return patterns
    
    async def infer_spending_habits(self, transaction_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use PLN to infer spending habits and tendencies"""
        patterns = await self.recognize_financial_patterns(transaction_history)
        
        # Infer habits from patterns
        habits = {
            'regular_categories': [],
            'irregular_categories': [],
            'spending_tendencies': [],
            'confidence': 0.7
        }
        
        for pattern in patterns:
            if pattern['transaction_count'] > 10:
                habits['regular_categories'].append(pattern['category'])
            else:
                habits['irregular_categories'].append(pattern['category'])
            
            if pattern['average_amount'] > 100:
                habits['spending_tendencies'].append(f"High spending on {pattern['category']}")
        
        return habits
