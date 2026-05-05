#!/usr/bin/env python3
"""
Phase 4 & 5 Integration Test for ElizaOS-OpenCog-GnuCash Framework
Tests Phase 4 optimization features and Phase 5 advanced applications
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from optimization.performance_optimization import PerformanceProfiler, CachingStrategy, DistributedProcessingEngine
from optimization.production_readiness import MonitoringSystem, BackupManager, DeploymentAutomation
from advanced_applications.intelligent_financial_advisory import IntelligentFinancialAdvisor, FinancialGoal, RiskTolerance
from advanced_applications.market_analysis_integration import MarketAnalysisEngine, MarketSentiment
from datetime import datetime, timedelta


async def test_performance_optimization():
    """Test Phase 4 performance optimization features"""
    print("Testing Phase 4 Performance Optimization...")
    
    # Test Performance Profiler
    profiler = PerformanceProfiler()
    
    async def sample_operation():
        await asyncio.sleep(0.05)  # Simulate work
        return "operation_complete"
    
    result = await profiler.profile_operation("test_operation", sample_operation)
    assert result == "operation_complete"
    
    report = profiler.get_performance_report()
    assert report['total_operations'] >= 1
    assert 'test_operation' in report['operation_statistics']
    
    # Test Caching Strategy
    cache = CachingStrategy()
    
    # Test cache set/get
    await cache.set("test_key", {"data": "test_value"}, "test_category")
    cached_value = await cache.get("test_key", "test_category")
    assert cached_value == {"data": "test_value"}
    
    cache_stats = cache.get_cache_stats()
    assert cache_stats['test_category']['hits'] >= 1
    
    # Test Distributed Processing
    processor = DistributedProcessingEngine()
    
    tasks = [
        {'data_type': 'transactions', 'count': 100},
        {'data_type': 'accounts', 'count': 50},
        {'data_type': 'budgets', 'count': 25}
    ]
    
    results = await processor.process_distributed('financial_analysis', tasks, 'thread')
    assert len(results) == 3
    assert all('result' in result for result in results if 'error' not in result)
    
    await processor.shutdown()
    
    print("✓ Phase 4 Performance Optimization tests passed")


async def test_production_readiness():
    """Test Phase 4 production readiness features"""
    print("Testing Phase 4 Production Readiness...")
    
    # Test Monitoring System
    monitoring = MonitoringSystem()
    
    # Start monitoring briefly
    monitoring_task = asyncio.create_task(monitoring.start_monitoring())
    
    # Let it run for a short time
    await asyncio.sleep(2)
    
    # Stop monitoring
    await monitoring.stop_monitoring()
    
    # Check system status
    status = monitoring.get_system_status()
    assert 'overall_status' in status
    assert 'health_checks' in status
    
    # Test Backup Manager
    backup_manager = BackupManager()
    
    backup_result = await backup_manager.create_full_backup()
    assert backup_result['status'] == 'completed'
    assert 'backup_id' in backup_result
    
    backups = backup_manager.list_backups()
    assert len(backups) >= 1
    
    # Test cleanup (without actually deleting recent backups)
    await backup_manager.cleanup_old_backups()
    
    # Test Deployment Automation
    deployment = DeploymentAutomation()
    
    deployment_result = await deployment.deploy('development', 'test_version')
    assert deployment_result['status'] == 'completed'
    assert deployment_result['environment'] == 'development'
    
    deployment_status = deployment.get_deployment_status(deployment_result['deployment_id'])
    assert deployment_status['status'] == 'completed'
    
    print("✓ Phase 4 Production Readiness tests passed")


async def test_intelligent_financial_advisory():
    """Test Phase 5 intelligent financial advisory features"""
    print("Testing Phase 5 Intelligent Financial Advisory...")
    
    advisor = IntelligentFinancialAdvisor()
    
    # Create client profile
    client_data = {
        'age': 35,
        'annual_income': 100000,
        'net_worth': 150000,
        'total_debt': 30000,
        'monthly_expenses': 5000,
        'dependents': 1,
        'investment_experience': 'intermediate',
        'goals': [
            {
                'name': 'Retirement',
                'target_amount': 1000000,
                'target_date': '2060-01-01',
                'priority': 10,
                'category': 'retirement',
                'current_savings': 50000,
                'monthly_contribution': 1000
            },
            {
                'name': 'House Down Payment',
                'target_amount': 80000,
                'target_date': '2030-01-01',
                'priority': 8,
                'category': 'real_estate',
                'current_savings': 20000,
                'monthly_contribution': 800
            }
        ],
        'risk_questionnaire': {
            'market_volatility_comfort': 7
        }
    }
    
    profile = await advisor.create_client_profile('test_client_001', client_data)
    assert profile['client_id'] == 'test_client_001'
    assert profile['risk_tolerance'] == RiskTolerance.AGGRESSIVE
    assert len(profile['goals']) == 2
    
    # Generate investment recommendations
    recommendations = await advisor.generate_investment_recommendations('test_client_001')
    assert len(recommendations) > 0
    assert all(rec.allocation_percentage > 0 for rec in recommendations)
    assert sum(rec.allocation_percentage for rec in recommendations) <= 101  # Allow for rounding
    
    # Generate tax optimization strategies
    tax_strategies = await advisor.generate_tax_optimization_strategies('test_client_001')
    assert len(tax_strategies) > 0
    assert all(strategy.estimated_savings > 0 for strategy in tax_strategies)
    
    # Generate retirement analysis
    retirement_analysis = await advisor.generate_retirement_analysis('test_client_001')
    assert 'retirement_goal' in retirement_analysis
    assert 'analysis' in retirement_analysis
    assert 'recommendations' in retirement_analysis
    
    print("✓ Phase 5 Intelligent Financial Advisory tests passed")


async def test_market_analysis_integration():
    """Test Phase 5 market analysis integration features"""
    print("Testing Phase 5 Market Analysis Integration...")
    
    market_engine = MarketAnalysisEngine()
    
    # Test real-time market data
    symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT']
    market_data = await market_engine.get_real_time_market_data(symbols)
    
    assert len(market_data) == len(symbols)
    assert all(data.symbol in symbols for data in market_data.values())
    assert all(data.price > 0 for data in market_data.values())
    
    # Test market sentiment analysis
    sentiment_analysis = await market_engine.analyze_market_sentiment()
    assert 'overall_sentiment' in sentiment_analysis
    assert sentiment_analysis['overall_sentiment'] in MarketSentiment
    assert 'component_sentiments' in sentiment_analysis
    assert 'confidence' in sentiment_analysis
    
    # Test portfolio optimization
    optimization = await market_engine.optimize_portfolio(
        assets=['SPY', 'QQQ', 'IWM'], 
        objective='max_sharpe'
    )
    
    assert optimization.objective == 'max_sharpe'
    assert len(optimization.weights) == 3
    assert abs(sum(optimization.weights.values()) - 1.0) < 0.01  # Should sum to ~1
    assert optimization.sharpe_ratio is not None
    
    # Test trading signals
    trading_signals = await market_engine.generate_trading_signals(
        symbols=['AAPL', 'MSFT'], 
        strategy='mean_reversion'
    )
    
    assert len(trading_signals) == 2
    assert 'AAPL' in trading_signals
    assert 'MSFT' in trading_signals
    
    # Test strategy backtest
    backtest_result = await market_engine.backtest_strategy(
        'mean_reversion',
        ['SPY', 'QQQ'],
        datetime.now() - timedelta(days=365),
        datetime.now()
    )
    
    assert 'total_return' in backtest_result
    assert 'sharpe_ratio' in backtest_result
    assert 'max_drawdown' in backtest_result
    
    # Test market summary
    market_summary = market_engine.get_market_summary()
    assert 'market_overview' in market_summary
    assert 'total_symbols_tracked' in market_summary['market_overview']
    
    print("✓ Phase 5 Market Analysis Integration tests passed")


async def test_integrated_workflow():
    """Test integrated workflow across Phase 4 & 5 components"""
    print("Testing Integrated Phase 4 & 5 Workflow...")
    
    # Initialize all components
    profiler = PerformanceProfiler()
    advisor = IntelligentFinancialAdvisor()
    market_engine = MarketAnalysisEngine()
    
    # Profile the complete workflow
    async def complete_financial_analysis():
        # 1. Create client profile
        client_data = {
            'age': 40,
            'annual_income': 120000,
            'net_worth': 200000,
            'investment_experience': 'experienced',
            'goals': [{
                'name': 'Retirement',
                'target_amount': 1500000,
                'target_date': '2055-01-01',
                'category': 'retirement',
                'current_savings': 75000,
                'monthly_contribution': 1200
            }]
        }
        
        profile = await advisor.create_client_profile('integrated_client', client_data)
        
        # 2. Get market data and sentiment
        market_data = await market_engine.get_real_time_market_data(['SPY', 'QQQ', 'BND'])
        sentiment = await market_engine.analyze_market_sentiment()
        
        # 3. Generate recommendations based on market conditions
        recommendations = await advisor.generate_investment_recommendations('integrated_client')
        
        # 4. Optimize portfolio
        portfolio_opt = await market_engine.optimize_portfolio(['SPY', 'QQQ', 'BND'])
        
        return {
            'profile': profile,
            'market_data': market_data,
            'sentiment': sentiment,
            'recommendations': recommendations,
            'portfolio_optimization': portfolio_opt
        }
    
    # Profile the complete workflow
    result = await profiler.profile_operation(
        "complete_financial_analysis",
        complete_financial_analysis
    )
    
    # Verify results
    assert 'profile' in result
    assert 'market_data' in result
    assert 'sentiment' in result
    assert 'recommendations' in result
    assert 'portfolio_optimization' in result
    
    # Check performance
    performance_report = profiler.get_performance_report()
    workflow_stats = performance_report['operation_statistics']['complete_financial_analysis']
    
    print(f"  ✓ Integrated workflow completed in {workflow_stats['average_duration']:.3f}s")
    print(f"  ✓ Generated {len(result['recommendations'])} investment recommendations")
    print(f"  ✓ Market sentiment: {result['sentiment']['overall_sentiment'].value}")
    print(f"  ✓ Portfolio Sharpe ratio: {result['portfolio_optimization'].sharpe_ratio:.2f}")
    
    print("✓ Integrated Phase 4 & 5 Workflow tests passed")


async def main():
    """Run all Phase 4 & 5 tests"""
    print("=== Phase 4 & 5 ElizaOS-OpenCog-GnuCash Integration Tests ===\n")
    
    try:
        await test_performance_optimization()
        print()
        
        await test_production_readiness()
        print()
        
        await test_intelligent_financial_advisory()
        print()
        
        await test_market_analysis_integration()
        print()
        
        await test_integrated_workflow()
        print()
        
        print("=== All Phase 4 & 5 Tests Passed! ===")
        print("✓ Performance Optimization - Profiling, caching, distributed processing")
        print("✓ Production Readiness - Monitoring, backup, deployment automation")
        print("✓ Intelligent Financial Advisory - Personalized recommendations and analysis")
        print("✓ Market Analysis Integration - Real-time data, sentiment, portfolio optimization")
        print("✓ Integrated Workflow - Seamless coordination across all advanced components")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)