"""
Comprehensive Test Suite for Modular Strategy Engine & Historical Data Replay

This test suite covers unit tests, regression tests, performance tests,
and risk management validation for the trading strategy framework.
"""

import unittest
import asyncio
import json
import tempfile
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from advanced_applications.strategy_engine import (
    ModularStrategyEngine, HistoricalDataReplay, BacktestPortfolio,
    MeanReversionStrategy, MomentumStrategy, PairsTradingStrategy, RiskParityStrategy,
    HistoricalDataPoint, TradingSignal, SignalStrength, StrategyPerformanceMetrics,
    StrategyType
)

from advanced_applications.ggml_strategies import (
    GGMLIntegration, MLBasedTradingStrategy, HypergraphPatternAnalyzer,
    MLModelType, MLFeatureSet, MLModelPrediction, HypergraphNode
)

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


class TestHistoricalDataReplay(unittest.TestCase):
    """Test historical data replay functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.replay = HistoricalDataReplay()
        self.symbols = ['AAPL', 'MSFT', 'GOOGL']
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 12, 31)
    
    async def test_load_historical_data(self):
        """Test loading historical data"""
        data = await self.replay.load_historical_data(
            self.symbols, self.start_date, self.end_date
        )
        
        # Verify data structure
        self.assertEqual(len(data), len(self.symbols))
        
        for symbol in self.symbols:
            self.assertIn(symbol, data)
            symbol_data = data[symbol]
            self.assertGreater(len(symbol_data), 0)
            
            # Verify data point structure
            first_point = symbol_data[0]
            self.assertEqual(first_point.symbol, symbol)
            self.assertIsInstance(first_point.timestamp, datetime)
            self.assertGreater(first_point.close_price, 0)
            self.assertGreater(first_point.volume, 0)
    
    async def test_data_replay_stream(self):
        """Test data replay streaming"""
        # Load test data
        await self.replay.load_historical_data(
            ['AAPL'], self.start_date, self.start_date + timedelta(days=10)
        )
        
        received_data = []
        
        async def capture_callback(current_data):
            received_data.append(current_data)
        
        # Replay with high speed
        await self.replay.replay_data_stream(['AAPL'], capture_callback, speed_multiplier=100)
        
        # Verify replay worked
        self.assertGreater(len(received_data), 5)
        self.assertIn('AAPL', received_data[0])
    
    def test_get_data_slice(self):
        """Test data slice functionality"""
        # Create mock data
        test_data = [
            HistoricalDataPoint(
                timestamp=datetime(2023, 1, i+1),
                symbol='TEST',
                open_price=100.0,
                high_price=105.0,
                low_price=95.0,
                close_price=100.0 + i,
                volume=1000000
            )
            for i in range(10)
        ]
        
        self.replay.data_cache['TEST'] = test_data
        
        # Test slice
        slice_data = self.replay.get_data_slice('TEST', 2, 5)
        self.assertEqual(len(slice_data), 3)
        self.assertEqual(slice_data[0].close_price, 102.0)
        self.assertEqual(slice_data[-1].close_price, 104.0)


class TestModularStrategyEngine(unittest.TestCase):
    """Test modular strategy engine functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = ModularStrategyEngine()
        self.symbols = ['SPY', 'QQQ']
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 6, 30)
    
    def test_strategy_registration(self):
        """Test strategy registration and management"""
        initial_count = len(self.engine.strategies)
        
        # Register custom strategy
        custom_strategy = MeanReversionStrategy(lookback_period=15)
        self.engine.register_strategy(custom_strategy)
        
        self.assertEqual(len(self.engine.strategies), initial_count + 1)
        self.assertIn(custom_strategy.strategy_id, self.engine.strategies)
        
        # Unregister strategy
        self.engine.unregister_strategy(custom_strategy.strategy_id)
        self.assertEqual(len(self.engine.strategies), initial_count)
        self.assertNotIn(custom_strategy.strategy_id, self.engine.strategies)
    
    def test_list_strategies(self):
        """Test strategy listing"""
        strategies = self.engine.list_strategies()
        
        self.assertIsInstance(strategies, list)
        self.assertGreater(len(strategies), 0)
        
        # Verify strategy info structure
        first_strategy = strategies[0]
        required_keys = ['strategy_id', 'strategy_name', 'strategy_type', 'parameters']
        for key in required_keys:
            self.assertIn(key, first_strategy)
    
    async def test_backtest_strategy(self):
        """Test strategy backtesting"""
        # Use a built-in strategy
        strategy_id = list(self.engine.strategies.keys())[0]
        
        # Run backtest
        metrics = await self.engine.backtest_strategy(
            strategy_id, self.symbols, self.start_date, self.end_date
        )
        
        # Verify metrics structure
        self.assertIsInstance(metrics, StrategyPerformanceMetrics)
        self.assertEqual(metrics.strategy_id, strategy_id)
        self.assertIsInstance(metrics.total_return, float)
        self.assertIsInstance(metrics.sharpe_ratio, float)
        self.assertIsInstance(metrics.max_drawdown, float)
        self.assertGreaterEqual(metrics.total_trades, 0)
    
    async def test_strategy_comparison(self):
        """Test strategy comparison functionality"""
        strategy_ids = list(self.engine.strategies.keys())[:2]  # Test with first 2 strategies
        
        results = await self.engine.run_strategy_comparison(
            strategy_ids, self.symbols, self.start_date, self.end_date
        )
        
        # Verify results
        self.assertEqual(len(results), len(strategy_ids))
        
        for strategy_id in strategy_ids:
            self.assertIn(strategy_id, results)
            metrics = results[strategy_id]
            self.assertIsInstance(metrics, StrategyPerformanceMetrics)
            self.assertEqual(metrics.strategy_id, strategy_id)


class TestTradingStrategies(unittest.TestCase):
    """Test individual trading strategies"""
    
    def setUp(self):
        """Set up test data"""
        self.test_data = self._create_test_data()
    
    def _create_test_data(self) -> List[HistoricalDataPoint]:
        """Create test historical data"""
        data = []
        base_price = 100.0
        
        for i in range(60):  # 60 days of data
            # Create trending price pattern
            price = base_price + (i * 0.5) + (5 * (i % 10 - 5) / 5)  # Trend + noise
            
            data.append(HistoricalDataPoint(
                timestamp=datetime(2023, 1, 1) + timedelta(days=i),
                symbol='TEST',
                open_price=price - 0.5,
                high_price=price + 1.0,
                low_price=price - 1.0,
                close_price=price,
                volume=1000000 + (i * 10000)
            ))
        
        return data
    
    async def test_mean_reversion_strategy(self):
        """Test mean reversion strategy"""
        strategy = MeanReversionStrategy(lookback_period=20, std_dev_threshold=1.5)
        
        signals = await strategy.generate_signals(self.test_data)
        
        # Verify signals
        self.assertIsInstance(signals, list)
        
        if signals:  # May not always generate signals
            for signal in signals:
                self.assertIsInstance(signal, TradingSignal)
                self.assertIn(signal.signal_type, ['BUY', 'SELL', 'HOLD'])
                self.assertIsInstance(signal.confidence, float)
                self.assertGreaterEqual(signal.confidence, 0.0)
                self.assertLessEqual(signal.confidence, 1.0)
    
    async def test_momentum_strategy(self):
        """Test momentum strategy"""
        strategy = MomentumStrategy(momentum_period=10, min_momentum=0.02)
        
        signals = await strategy.generate_signals(self.test_data)
        
        self.assertIsInstance(signals, list)
        
        if signals:
            for signal in signals:
                self.assertIsInstance(signal, TradingSignal)
                self.assertEqual(signal.strategy_id, strategy.strategy_id)
                self.assertIsNotNone(signal.reasoning)
    
    async def test_pairs_trading_strategy(self):
        """Test pairs trading strategy"""
        # Create correlated data for two symbols
        correlated_data = []
        
        for i in range(60):
            price1 = 100 + i * 0.3 + (2 * (i % 5 - 2))  # Base trend + noise
            price2 = 80 + i * 0.25 + (1.5 * (i % 5 - 2))  # Correlated trend
            
            # Add data for both symbols
            for symbol, price in [('TEST1', price1), ('TEST2', price2)]:
                correlated_data.append(HistoricalDataPoint(
                    timestamp=datetime(2023, 1, 1) + timedelta(days=i),
                    symbol=symbol,
                    open_price=price - 0.5,
                    high_price=price + 1.0,
                    low_price=price - 1.0,
                    close_price=price,
                    volume=500000
                ))
        
        strategy = PairsTradingStrategy(correlation_threshold=0.7, spread_threshold=1.5)
        signals = await strategy.generate_signals(correlated_data)
        
        self.assertIsInstance(signals, list)
        # Pairs trading should generate signals for both symbols or none
    
    async def test_risk_parity_strategy(self):
        """Test risk parity strategy"""
        # Create multi-symbol data
        multi_symbol_data = []
        symbols = ['STOCK1', 'STOCK2', 'STOCK3']
        
        for i in range(60):
            for j, symbol in enumerate(symbols):
                # Different volatilities for each symbol
                volatility = 0.01 * (j + 1)
                price = 100 + (i * 0.2) + (volatility * 50 * (i % 7 - 3))
                
                multi_symbol_data.append(HistoricalDataPoint(
                    timestamp=datetime(2023, 1, 1) + timedelta(days=i),
                    symbol=symbol,
                    open_price=price - 0.5,
                    high_price=price + 1.0,
                    low_price=price - 1.0,
                    close_price=price,
                    volume=200000
                ))
        
        strategy = RiskParityStrategy(rebalance_threshold=0.05)
        signals = await strategy.generate_signals(multi_symbol_data)
        
        self.assertIsInstance(signals, list)
        
        if signals:
            # Should generate signals for rebalancing
            symbols_with_signals = {signal.symbol for signal in signals}
            self.assertLessEqual(len(symbols_with_signals), len(symbols))


class TestBacktestPortfolio(unittest.TestCase):
    """Test portfolio management for backtesting"""
    
    def setUp(self):
        """Set up portfolio"""
        self.portfolio = BacktestPortfolio(initial_capital=100000.0, transaction_cost=0.001)
        self.test_signal = TradingSignal(
            symbol='TEST',
            signal_type='BUY',
            strength=SignalStrength.MODERATE,
            confidence=0.8,
            timestamp=datetime.now(),
            strategy_id='test_strategy',
            entry_price=100.0
        )
        self.test_data = [HistoricalDataPoint(
            timestamp=datetime.now(),
            symbol='TEST',
            open_price=99.0,
            high_price=101.0,
            low_price=98.0,
            close_price=100.0,
            volume=1000000
        )]
    
    async def test_buy_execution(self):
        """Test buy order execution"""
        initial_cash = self.portfolio.cash
        
        trade = await self.portfolio.execute_signal(self.test_signal, self.test_data)
        
        # Verify trade execution
        if trade:  # May not execute if insufficient funds or other constraints
            self.assertEqual(trade['type'], 'BUY')
            self.assertEqual(trade['symbol'], 'TEST')
            self.assertGreater(trade['shares'], 0)
            self.assertLess(self.portfolio.cash, initial_cash)  # Cash should decrease
            
            # Verify position was created
            self.assertIn('TEST', self.portfolio.positions)
            self.assertGreater(self.portfolio.positions['TEST']['shares'], 0)
    
    async def test_sell_execution(self):
        """Test sell order execution"""
        # First buy some shares
        await self.portfolio.execute_signal(self.test_signal, self.test_data)
        
        # Create sell signal
        sell_signal = TradingSignal(
            symbol='TEST',
            signal_type='SELL',
            strength=SignalStrength.MODERATE,
            confidence=0.8,
            timestamp=datetime.now(),
            strategy_id='test_strategy',
            entry_price=105.0
        )
        
        # Update test data for selling
        sell_data = [HistoricalDataPoint(
            timestamp=datetime.now(),
            symbol='TEST',
            open_price=104.0,
            high_price=106.0,
            low_price=103.0,
            close_price=105.0,
            volume=1000000
        )]
        
        if self.portfolio.positions.get('TEST', {}).get('shares', 0) > 0:
            initial_cash = self.portfolio.cash
            trade = await self.portfolio.execute_signal(sell_signal, sell_data)
            
            if trade:
                self.assertEqual(trade['type'], 'SELL')
                self.assertGreater(self.portfolio.cash, initial_cash)  # Cash should increase
                self.assertIn('pnl', trade)  # Should have P&L calculation
    
    def test_portfolio_value_update(self):
        """Test portfolio value updates"""
        initial_value = self.portfolio.total_value
        
        # Update with test data (should maintain value if no positions)
        self.portfolio.update_portfolio_value(self.test_data)
        
        self.assertEqual(self.portfolio.total_value, initial_value)


class TestGGMLIntegration(unittest.TestCase):
    """Test GGML integration functionality"""
    
    def setUp(self):
        """Set up GGML integration"""
        self.ggml = GGMLIntegration()
        self.test_data = self._create_test_data()
    
    def _create_test_data(self) -> List[HistoricalDataPoint]:
        """Create test data for GGML testing"""
        data = []
        for i in range(100):  # More data for ML features
            price = 100 + (i * 0.1) + (3 * ((i % 20) - 10) / 10)  # Price with pattern
            
            data.append(HistoricalDataPoint(
                timestamp=datetime(2023, 1, 1) + timedelta(days=i),
                symbol='MLTEST',
                open_price=price - 0.5,
                high_price=price + 1.5,
                low_price=price - 1.5,
                close_price=price,
                volume=800000 + (i * 5000)
            ))
        
        return data
    
    async def test_feature_extraction(self):
        """Test ML feature extraction"""
        features = await self.ggml.extract_features(self.test_data, 'MLTEST')
        
        # Verify feature structure
        self.assertIsInstance(features, MLFeatureSet)
        self.assertEqual(features.symbol, 'MLTEST')
        
        # Check all feature categories
        self.assertIsInstance(features.price_features, dict)
        self.assertIsInstance(features.technical_features, dict)
        self.assertIsInstance(features.volume_features, dict)
        self.assertIsInstance(features.sentiment_features, dict)
        self.assertIsInstance(features.macro_features, dict)
        self.assertIsInstance(features.hypergraph_features, dict)
        
        # Verify key price features
        self.assertIn('close', features.price_features)
        self.assertIn('return_1d', features.price_features)
        self.assertIn('volatility_20d', features.price_features)
        
        # Verify technical features
        self.assertIn('rsi', features.technical_features)
        self.assertIn('sma_20', features.technical_features)
    
    async def test_model_prediction(self):
        """Test ML model predictions"""
        features = await self.ggml.extract_features(self.test_data, 'MLTEST')
        
        # Test each model type
        for model_id in self.ggml.models.keys():
            prediction = await self.ggml.predict_with_model(model_id, features)
            
            # Verify prediction structure
            self.assertIsInstance(prediction, MLModelPrediction)
            self.assertEqual(prediction.model_id, model_id)
            self.assertEqual(prediction.symbol, 'MLTEST')
            self.assertIsInstance(prediction.prediction, float)
            self.assertIsInstance(prediction.confidence, float)
            self.assertGreaterEqual(prediction.confidence, 0.0)
            self.assertLessEqual(prediction.confidence, 1.0)
            
            # Verify explanation and feature importance
            self.assertIsInstance(prediction.explanation, dict)
            self.assertIsInstance(prediction.feature_importance, dict)


class TestHypergraphPatternAnalyzer(unittest.TestCase):
    """Test hypergraph pattern analysis"""
    
    def setUp(self):
        """Set up hypergraph analyzer"""
        self.analyzer = HypergraphPatternAnalyzer()
        self.test_data = self._create_pattern_data()
    
    def _create_pattern_data(self) -> List[HistoricalDataPoint]:
        """Create data with identifiable patterns"""
        data = []
        for i in range(50):
            # Create cyclical pattern with trend
            cycle = 5 * ((i % 10) / 10) * 2 - 1  # -1 to 1 cycle
            trend = i * 0.2
            price = 100 + trend + cycle
            
            data.append(HistoricalDataPoint(
                timestamp=datetime(2023, 1, 1) + timedelta(days=i),
                symbol='PATTERN',
                open_price=price - 0.3,
                high_price=price + 0.8,
                low_price=price - 0.8,
                close_price=price,
                volume=600000 + abs(int(cycle * 100000))  # Volume correlated with pattern
            ))
        
        return data
    
    async def test_hypergraph_feature_extraction(self):
        """Test hypergraph feature extraction"""
        features = await self.analyzer.extract_hypergraph_features(self.test_data, 'PATTERN')
        
        # Verify all expected features are present
        expected_features = [
            'pattern_strength', 'node_connectivity', 'node_centrality',
            'clustering_coefficient', 'network_density', 'pattern_consistency',
            'information_flow', 'structural_stability'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
            self.assertGreaterEqual(features[feature], 0.0)
            self.assertLessEqual(features[feature], 1.0)
    
    def test_node_creation(self):
        """Test hypergraph node creation"""
        # Create nodes
        self.analyzer._create_price_nodes(self.test_data, 'PATTERN')
        self.analyzer._create_volume_nodes(self.test_data, 'PATTERN')
        self.analyzer._create_pattern_nodes(self.test_data, 'PATTERN')
        
        # Verify nodes were created
        pattern_nodes = [n for n in self.analyzer.nodes.keys() if 'PATTERN' in n]
        self.assertGreater(len(pattern_nodes), 0)
        
        # Verify node structure
        if pattern_nodes:
            first_node = self.analyzer.nodes[pattern_nodes[0]]
            self.assertIsInstance(first_node, HypergraphNode)
            self.assertIn(first_node.node_type, ['price', 'volume', 'pattern'])


class TestMLBasedTradingStrategy(unittest.TestCase):
    """Test ML-based trading strategy"""
    
    def setUp(self):
        """Set up ML strategy"""
        model_ids = ['lstm_price_predictor', 'transformer_pattern']
        self.ml_strategy = MLBasedTradingStrategy(model_ids, ensemble_method='weighted_average')
        self.test_data = self._create_ml_test_data()
    
    def _create_ml_test_data(self) -> List[HistoricalDataPoint]:
        """Create test data for ML strategy"""
        data = []
        for i in range(80):
            # Create data with ML-detectable patterns
            base_price = 150.0
            trend = i * 0.15
            momentum = 3 * ((i % 15) - 7.5) / 7.5
            noise = ((i * 17) % 7 - 3) * 0.5
            
            price = base_price + trend + momentum + noise
            
            data.append(HistoricalDataPoint(
                timestamp=datetime(2023, 1, 1) + timedelta(days=i),
                symbol='MLSTOCK',
                open_price=price - 1.0,
                high_price=price + 2.0,
                low_price=price - 2.0,
                close_price=price,
                volume=1200000 + int(abs(momentum) * 200000)
            ))
        
        return data
    
    async def test_ml_signal_generation(self):
        """Test ML-based signal generation"""
        signals = await self.ml_strategy.generate_signals(self.test_data)
        
        # Verify signals
        self.assertIsInstance(signals, list)
        
        if signals:  # May not generate signals if confidence too low
            for signal in signals:
                self.assertIsInstance(signal, TradingSignal)
                self.assertEqual(signal.strategy_id, self.ml_strategy.strategy_id)
                self.assertIn(signal.signal_type, ['BUY', 'SELL'])
                
                # Verify ML-specific metadata
                self.assertIn('metadata', signal.__dict__)
                if 'metadata' in signal.__dict__ and signal.metadata:
                    self.assertIn('ensemble_prediction', signal.metadata)
                    self.assertIn('model_contributions', signal.metadata)
    
    def test_strategy_properties(self):
        """Test strategy properties"""
        self.assertEqual(self.ml_strategy.strategy_type, StrategyType.MACHINE_LEARNING)
        self.assertIn('model_ids', self.ml_strategy.parameters)
        self.assertIn('ensemble_method', self.ml_strategy.parameters)
        self.assertGreater(self.ml_strategy.get_required_data_history(), 50)


class TestPerformanceAndRisk(unittest.TestCase):
    """Test performance and risk management"""
    
    def setUp(self):
        """Set up performance testing"""
        self.engine = ModularStrategyEngine()
        self.start_time = None
    
    async def test_strategy_performance_timing(self):
        """Test strategy performance under load"""
        import time
        
        symbols = ['SPY', 'QQQ', 'IWM', 'VTI']
        strategy_id = list(self.engine.strategies.keys())[0]
        
        start_time = time.time()
        
        # Run backtest
        await self.engine.backtest_strategy(
            strategy_id, symbols, 
            datetime(2023, 1, 1), datetime(2023, 3, 31)
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(execution_time, 30.0, "Strategy backtest took too long")
    
    async def test_risk_metrics_calculation(self):
        """Test risk metrics calculation"""
        strategy_id = list(self.engine.strategies.keys())[0]
        
        metrics = await self.engine.backtest_strategy(
            strategy_id, ['SPY'], 
            datetime(2023, 1, 1), datetime(2023, 6, 30)
        )
        
        # Verify risk metrics are within reasonable bounds
        self.assertGreaterEqual(metrics.max_drawdown, 0.0)
        self.assertLessEqual(metrics.max_drawdown, 1.0)
        
        self.assertGreaterEqual(metrics.win_rate, 0.0)
        self.assertLessEqual(metrics.win_rate, 1.0)
        
        # Volatility should be positive
        self.assertGreaterEqual(metrics.volatility, 0.0)
        
        # Sharpe ratio should be reasonable (between -10 and 10)
        self.assertGreater(metrics.sharpe_ratio, -10.0)
        self.assertLess(metrics.sharpe_ratio, 10.0)
    
    async def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple backtests
        for i in range(3):
            strategy_id = list(self.engine.strategies.keys())[i % len(self.engine.strategies)]
            await self.engine.backtest_strategy(
                strategy_id, ['SPY'], 
                datetime(2023, 1, 1), datetime(2023, 2, 28)
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        self.assertLess(memory_increase, 100.0, f"Memory usage increased by {memory_increase:.1f}MB")


class TestRegressionSuite(unittest.TestCase):
    """Regression tests to ensure existing functionality doesn't break"""
    
    def setUp(self):
        """Set up regression tests"""
        self.engine = ModularStrategyEngine()
    
    async def test_all_builtin_strategies_work(self):
        """Test that all built-in strategies can generate signals"""
        test_data = self._create_regression_data()
        
        for strategy_id, strategy in self.engine.strategies.items():
            try:
                signals = await strategy.generate_signals(test_data)
                self.assertIsInstance(signals, list, f"Strategy {strategy_id} should return list of signals")
                
                # If signals generated, verify structure
                for signal in signals:
                    self.assertIsInstance(signal, TradingSignal)
                    self.assertIn(signal.signal_type, ['BUY', 'SELL', 'HOLD'])
                    
            except Exception as e:
                self.fail(f"Strategy {strategy_id} failed to generate signals: {e}")
    
    def _create_regression_data(self) -> List[HistoricalDataPoint]:
        """Create standard regression test data"""
        data = []
        symbols = ['TEST1', 'TEST2', 'TEST3']  # Multiple symbols for pairs trading
        
        for i in range(100):
            for symbol in symbols:
                base_price = 100.0 + (symbols.index(symbol) * 20)  # Different base prices
                trend = i * 0.1
                volatility = 2 * ((i % 10) - 5) / 5
                
                price = base_price + trend + volatility
                
                data.append(HistoricalDataPoint(
                    timestamp=datetime(2023, 1, 1) + timedelta(days=i),
                    symbol=symbol,
                    open_price=price - 0.5,
                    high_price=price + 1.0,
                    low_price=price - 1.0,
                    close_price=price,
                    volume=1000000 + (i * 10000)
                ))
        
        return data
    
    async def test_backtest_consistency(self):
        """Test that backtests produce consistent results"""
        strategy_id = list(self.engine.strategies.keys())[0]
        symbols = ['SPY']
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 31)
        
        # Run backtest twice
        metrics1 = await self.engine.backtest_strategy(strategy_id, symbols, start_date, end_date)
        metrics2 = await self.engine.backtest_strategy(strategy_id, symbols, start_date, end_date)
        
        # Results should be identical (deterministic)
        self.assertAlmostEqual(metrics1.total_return, metrics2.total_return, places=6)
        self.assertAlmostEqual(metrics1.sharpe_ratio, metrics2.sharpe_ratio, places=6)
        self.assertEqual(metrics1.total_trades, metrics2.total_trades)


async def run_async_tests():
    """Run all async tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestHistoricalDataReplay,
        TestModularStrategyEngine,
        TestTradingStrategies,
        TestBacktestPortfolio,
        TestGGMLIntegration,
        TestHypergraphPatternAnalyzer,
        TestMLBasedTradingStrategy,
        TestPerformanceAndRisk,
        TestRegressionSuite
    ]
    
    # Add async test methods
    async_test_methods = [
        'test_load_historical_data',
        'test_data_replay_stream',
        'test_backtest_strategy',
        'test_strategy_comparison',
        'test_mean_reversion_strategy',
        'test_momentum_strategy',
        'test_pairs_trading_strategy',
        'test_risk_parity_strategy',
        'test_buy_execution',
        'test_sell_execution',
        'test_feature_extraction',
        'test_model_prediction',
        'test_hypergraph_feature_extraction',
        'test_ml_signal_generation',
        'test_strategy_performance_timing',
        'test_risk_metrics_calculation',
        'test_memory_usage',
        'test_all_builtin_strategies_work',
        'test_backtest_consistency'
    ]
    
    # Run async tests
    results = {'passed': 0, 'failed': 0, 'errors': []}
    
    for test_class in test_classes:
        test_instance = test_class()
        test_instance.setUp()
        
        for method_name in dir(test_instance):
            if method_name.startswith('test_') and method_name in async_test_methods:
                try:
                    method = getattr(test_instance, method_name)
                    await method()
                    results['passed'] += 1
                    print(f"✓ {test_class.__name__}.{method_name}")
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"{test_class.__name__}.{method_name}: {str(e)}")
                    print(f"✗ {test_class.__name__}.{method_name}: {str(e)}")
    
    return results


def run_sync_tests():
    """Run synchronous tests"""
    # Create test suite for sync tests
    suite = unittest.TestSuite()
    
    # Add sync test methods
    sync_test_classes = [
        TestHistoricalDataReplay,
        TestModularStrategyEngine,
        TestTradingStrategies,
        TestBacktestPortfolio,
        TestGGMLIntegration,
        TestHypergraphPatternAnalyzer,
        TestMLBasedTradingStrategy,
        TestPerformanceAndRisk,
        TestRegressionSuite
    ]
    
    sync_test_methods = [
        'test_get_data_slice',
        'test_strategy_registration',
        'test_list_strategies',
        'test_portfolio_value_update',
        'test_node_creation',
        'test_strategy_properties'
    ]
    
    results = {'passed': 0, 'failed': 0, 'errors': []}
    
    for test_class in sync_test_classes:
        test_instance = test_class()
        test_instance.setUp()
        
        for method_name in dir(test_instance):
            if method_name.startswith('test_') and method_name in sync_test_methods:
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    results['passed'] += 1
                    print(f"✓ {test_class.__name__}.{method_name}")
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"{test_class.__name__}.{method_name}: {str(e)}")
                    print(f"✗ {test_class.__name__}.{method_name}: {str(e)}")
    
    return results


def main():
    """Main test runner"""
    print("🧪 Running Comprehensive Strategy Engine Test Suite")
    print("=" * 60)
    
    # Run synchronous tests
    print("\n📋 Running Synchronous Tests...")
    sync_results = run_sync_tests()
    
    # Run asynchronous tests
    print("\n⚡ Running Asynchronous Tests...")
    async_results = asyncio.run(run_async_tests())
    
    # Combine results
    total_passed = sync_results['passed'] + async_results['passed']
    total_failed = sync_results['failed'] + async_results['failed']
    all_errors = sync_results['errors'] + async_results['errors']
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"📈 Success Rate: {total_passed/(total_passed + total_failed)*100:.1f}%")
    
    if all_errors:
        print("\n❌ FAILED TESTS:")
        for error in all_errors:
            print(f"  - {error}")
    
    print("\n🎯 Test Categories Covered:")
    print("  • Historical Data Replay ✓")
    print("  • Modular Strategy Engine ✓") 
    print("  • Individual Trading Strategies ✓")
    print("  • Portfolio Management ✓")
    print("  • GGML ML Integration ✓")
    print("  • Hypergraph Pattern Analysis ✓")
    print("  • Performance & Risk Management ✓")
    print("  • Regression Testing ✓")
    
    return total_failed == 0


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)