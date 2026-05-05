"""
AI-Powered Transaction Categorization for ElizaOS

Two-stage pipeline:
1. Fast regex pattern matching (always available).
2. scikit-learn TF-IDF + LogisticRegression classifier that is trained on-the-fly
   from high-confidence pattern results whenever enough labelled examples have
   been collected.  Falls back gracefully if scikit-learn is not installed.
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Minimum labelled samples per class before training the ML model
_MIN_SAMPLES_FOR_ML = 5


@dataclass
class CategoryPrediction:
    """Result of transaction categorization"""
    category: str
    confidence: float
    subcategory: Optional[str] = None
    reasoning: str = ""
    alternative_categories: List[Tuple[str, float]] = field(default_factory=list)


class AITransactionCategorizer:
    """
    AI-powered transaction categorization system.

    Stage 1 – regex pattern matching (zero dependencies).
    Stage 2 – scikit-learn TF-IDF + LogisticRegression, trained automatically
               once enough labelled examples accumulate.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.category_patterns = self._initialize_patterns()
        self.merchant_cache: Dict[str, Dict] = {}
        self.category_history: defaultdict = defaultdict(list)

        # ML model components
        self._ml_pipeline: Optional[Any] = None  # sklearn Pipeline
        self._ml_texts: List[str] = []
        self._ml_labels: List[str] = []
        self._ml_trained = False
        
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
        """Categorize a single transaction.

        1. Check merchant cache.
        2. Try ML model (if trained).
        3. Fall back to regex patterns.
        4. Final heuristic on amount sign.
        """
        description = transaction.get('description', '').lower()
        amount = transaction.get('amount', 0.0)
        merchant = transaction.get('merchant', '').lower()

        # 1. Merchant cache
        if merchant and merchant in self.merchant_cache:
            cached = self.merchant_cache[merchant]
            return CategoryPrediction(
                category=cached['category'],
                confidence=min(0.95, cached['confidence'] + 0.1),
                reasoning="Cached from previous categorization"
            )

        combined_text = f"{description} {merchant}".strip()

        # 2. ML model prediction
        if self._ml_trained and self._ml_pipeline is not None and combined_text:
            try:
                proba = self._ml_pipeline.predict_proba([combined_text])[0]
                classes = self._ml_pipeline.classes_
                best_idx = int(proba.argmax())
                best_category = classes[best_idx]
                best_confidence = float(proba[best_idx])
                if best_confidence > 0.5:
                    alternatives = [
                        (classes[i], float(proba[i]))
                        for i in range(len(classes))
                        if i != best_idx and proba[i] > 0.05
                    ]
                    result = CategoryPrediction(
                        category=best_category,
                        confidence=best_confidence,
                        reasoning="ML classifier (TF-IDF + LogisticRegression)",
                        alternative_categories=sorted(alternatives, key=lambda x: -x[1])[:3]
                    )
                    if merchant:
                        self.merchant_cache[merchant] = {
                            'category': result.category,
                            'confidence': result.confidence
                        }
                    return result
            except Exception as exc:
                logger.debug(f"ML prediction failed, using patterns: {exc}")

        # 3. Regex pattern matching
        pattern_results = []
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, description) or re.search(pattern, merchant):
                    pattern_results.append((category, 0.8))

        if pattern_results:
            category_scores: defaultdict = defaultdict(float)
            for cat, conf in pattern_results:
                category_scores[cat] += conf

            best_category, best_score = max(category_scores.items(), key=lambda x: x[1])
            alternatives = sorted(
                [(cat, score) for cat, score in category_scores.items() if cat != best_category],
                key=lambda x: x[1],
                reverse=True
            )[:3]

            result = CategoryPrediction(
                category=best_category,
                confidence=min(0.95, best_score),
                reasoning="Regex pattern matching on description/merchant",
                alternative_categories=alternatives
            )
            if merchant:
                self.merchant_cache[merchant] = {
                    'category': result.category,
                    'confidence': result.confidence
                }
            # Collect labelled sample for ML training
            self._add_training_sample(combined_text, best_category)
            return result

        # 4. Amount-based heuristic
        if amount < 0:
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
        return CategoryPrediction(
            category='uncategorized',
            confidence=0.3,
            reasoning="No matching patterns found"
        )
    
    async def categorize_batch(self, transactions: List[Dict[str, Any]]) -> List[CategoryPrediction]:
        """Categorize multiple transactions, then retrain ML model if enough data."""
        results = []
        for transaction in transactions:
            result = await self.categorize_transaction(transaction)
            results.append(result)

        # Retrain after a batch
        self._maybe_train_ml()
        return results

    # ------------------------------------------------------------------
    # ML training helpers
    # ------------------------------------------------------------------

    def _add_training_sample(self, text: str, label: str) -> None:
        """Collect a labelled sample for future ML training."""
        if text.strip():
            self._ml_texts.append(text)
            self._ml_labels.append(label)

    def _maybe_train_ml(self) -> None:
        """Train (or retrain) the ML model if sufficient data is available."""
        if not _SKLEARN_AVAILABLE:
            return
        # Check that each class has at least _MIN_SAMPLES_FOR_ML examples
        label_counts: defaultdict = defaultdict(int)
        for label in self._ml_labels:
            label_counts[label] += 1
        qualifying = sum(1 for c in label_counts.values() if c >= _MIN_SAMPLES_FOR_ML)
        if qualifying < 2:
            return

        # Filter to only classes with enough samples
        qualified_classes = {c for c, n in label_counts.items() if n >= _MIN_SAMPLES_FOR_ML}
        filtered = [
            (t, l) for t, l in zip(self._ml_texts, self._ml_labels)
            if l in qualified_classes
        ]
        if len(filtered) < 10:
            return

        texts, labels = zip(*filtered)
        try:
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    analyzer='word',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=5000
                )),
                ('clf', LogisticRegression(max_iter=500, C=1.0))
            ])
            pipeline.fit(list(texts), list(labels))
            self._ml_pipeline = pipeline
            self._ml_trained = True
            logger.info(
                f"ML categorizer trained on {len(texts)} samples "
                f"({len(set(labels))} categories)"
            )
        except Exception as exc:
            logger.warning(f"ML training failed: {exc}")

    def train_from_labelled_data(self, labelled: List[Tuple[str, str]]) -> bool:
        """Seed the ML model from externally supplied (text, category) pairs.

        Args:
            labelled: list of (transaction_text, category) tuples.

        Returns:
            True if the model was trained, False if scikit-learn is unavailable
            or there is insufficient data.
        """
        if not _SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available — ML categorization disabled")
            return False
        for text, label in labelled:
            self._add_training_sample(text, label)
        self._maybe_train_ml()
        return self._ml_trained

    def _update_learned_patterns(self, transaction: Dict, category: str):
        """Keep for backward compatibility — now delegates to ML training."""
        description = transaction.get('description', '').lower()
        merchant = transaction.get('merchant', '').lower()
        combined = f"{description} {merchant}".strip()
        self._add_training_sample(combined, category)
    
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
        """Get statistics about categorization performance."""
        return {
            'total_patterns': sum(len(p) for p in self.category_patterns.values()),
            'ml_available': _SKLEARN_AVAILABLE,
            'ml_trained': self._ml_trained,
            'ml_training_samples': len(self._ml_texts),
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
