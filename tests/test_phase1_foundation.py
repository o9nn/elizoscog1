#!/usr/bin/env python3
"""
Comprehensive Testing Framework for ElizaOS-OpenCog-GnuCash Integration
Phase 1: Foundation testing infrastructure
"""

import asyncio
import unittest
import logging
import tempfile
import os
from datetime import date, timedelta
from pathlib import Path

# Import core components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.atomspace_bindings import AtomSpaceCore, FinancialAtomSpace
from src.core.elizaos_plugin_architecture import (
    OpenCogAgentPlugin, FinancialCognitivePlugin, ElizaOSPluginManager
)
from src.core.gnucash_access import GnuCashDataAccess, FinancialPatternAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAtomSpaceCore(unittest.IsolatedAsyncioTestCase):
    """Test AtomSpace core functionality"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        self.atomspace = AtomSpaceCore()
        await self.atomspace.initialize()
    
    async def test_atomspace_initialization(self):
        """Test AtomSpace initialization"""
        self.assertTrue(self.atomspace.initialized)
        self.assertEqual(self.atomspace.get_atom_count(), 0)
        self.assertEqual(self.atomspace.get_link_count(), 0)
    
    async def test_atom_creation(self):
        """Test creating atoms"""
        atom_id = self.atomspace.create_atom('ConceptNode', 'TestConcept')
        self.assertIsInstance(atom_id, int)
        self.assertEqual(self.atomspace.get_atom_count(), 1)
        
        atom = self.atomspace.get_atom(atom_id)
        self.assertIsNotNone(atom)
        self.assertEqual(atom['name'], 'TestConcept')
        self.assertEqual(atom['type'], 'ConceptNode')
    
    async def test_link_creation(self):
        """Test creating links between atoms"""
        atom1 = self.atomspace.create_atom('ConceptNode', 'Concept1')
        atom2 = self.atomspace.create_atom('ConceptNode', 'Concept2')
        
        link_id = self.atomspace.create_link('ListLink', [atom1, atom2])
        self.assertEqual(self.atomspace.get_link_count(), 1)
        
        link = self.atomspace.get_link(link_id)
        self.assertIsNotNone(link)
        self.assertEqual(link['type'], 'ListLink')
        self.assertEqual(link['outgoing'], [atom1, atom2])
    
    async def test_query_functionality(self):
        """Test querying atoms"""
        # Create test atoms
        concept1 = self.atomspace.create_atom('ConceptNode', 'Financial')
        concept2 = self.atomspace.create_atom('ConceptNode', 'Transaction')
        predicate = self.atomspace.create_atom('PredicateNode', 'HasType')
        
        # Test query by type
        concepts = self.atomspace.query_by_type('ConceptNode')
        self.assertEqual(len(concepts), 2)
        
        predicates = self.atomspace.query_by_type('PredicateNode')
        self.assertEqual(len(predicates), 1)
        
        # Test query by name
        financial_atoms = self.atomspace.query_by_name('Financial')
        self.assertEqual(len(financial_atoms), 1)
        self.assertEqual(financial_atoms[0]['name'], 'Financial')


class TestFinancialAtomSpace(unittest.IsolatedAsyncioTestCase):
    """Test financial-specific AtomSpace functionality"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        self.financial_atomspace = FinancialAtomSpace()
        await self.financial_atomspace.initialize()
    
    async def test_financial_atomspace_initialization(self):
        """Test financial AtomSpace initialization"""
        self.assertTrue(self.financial_atomspace.initialized)
        self.assertIn('account', self.financial_atomspace.financial_concepts)
        self.assertIn('transaction', self.financial_atomspace.financial_concepts)
    
    async def test_account_creation(self):
        """Test creating financial account atoms"""
        account_id = self.financial_atomspace.create_account_atom(
            'TestChecking', 'BANK', 1000.0
        )
        
        self.assertIsInstance(account_id, int)
        self.assertIn('TestChecking', self.financial_atomspace.account_atoms)
        
        balance = self.financial_atomspace.get_account_balance('TestChecking')
        self.assertEqual(balance, 1000.0)
    
    async def test_transaction_creation(self):
        """Test creating transaction links"""
        # Create accounts
        self.financial_atomspace.create_account_atom('Checking', 'BANK', 1500.0)
        self.financial_atomspace.create_account_atom('Groceries', 'EXPENSE', 0.0)
        
        # Create transaction
        tx_link = self.financial_atomspace.create_transaction_link(
            'Checking', 'Groceries', 85.50, 'Weekly shopping'
        )
        
        self.assertIsInstance(tx_link, int)
        
        # Query transactions for account
        transactions = self.financial_atomspace.query_transactions_by_account('Checking')
        self.assertGreater(len(transactions), 0)


class TestGnuCashDataAccess(unittest.IsolatedAsyncioTestCase):
    """Test GnuCash data access functionality"""
    
    async def asyncSetUp(self):
        """Set up test environment with temporary database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_gnucash.sqlite')
        self.data_access = GnuCashDataAccess(self.db_path)
        await self.data_access.initialize()
    
    async def asyncTearDown(self):
        """Clean up test environment"""
        await self.data_access.close()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    async def test_database_initialization(self):
        """Test database initialization"""
        self.assertTrue(self.data_access.initialized)
        self.assertIsNotNone(self.data_access.connection)
    
    async def test_get_accounts(self):
        """Test retrieving accounts"""
        accounts = await self.data_access.get_accounts()
        self.assertGreater(len(accounts), 0)
        
        # Test filtering by account type
        bank_accounts = await self.data_access.get_accounts('BANK')
        self.assertGreater(len(bank_accounts), 0)
        
        for account in bank_accounts:
            self.assertEqual(account['account_type'], 'BANK')
    
    async def test_get_account_balance(self):
        """Test getting account balances"""
        accounts = await self.data_access.get_accounts('BANK')
        if accounts:
            account_guid = accounts[0]['guid']
            balance = await self.data_access.get_account_balance(account_guid)
            self.assertIsInstance(balance, type(balance))  # Decimal type
    
    async def test_get_transactions(self):
        """Test retrieving transactions"""
        transactions = await self.data_access.get_transactions(limit=10)
        self.assertIsInstance(transactions, list)
        
        if transactions:
            tx = transactions[0]
            required_fields = ['transaction_guid', 'description', 'post_date', 'amount']
            for field in required_fields:
                self.assertIn(field, tx)
    
    async def test_spending_analysis(self):
        """Test spending by category analysis"""
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        
        spending = await self.data_access.get_spending_by_category(start_date, end_date)
        self.assertIsInstance(spending, dict)


class TestElizaOSPlugins(unittest.IsolatedAsyncioTestCase):
    """Test ElizaOS plugin architecture"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        self.plugin_manager = ElizaOSPluginManager()
        
        # Create plugin configurations
        self.opencog_config = {
            'atomspace': {'host': 'localhost', 'port': 17001}
        }
        
        temp_dir = tempfile.mkdtemp()
        self.financial_config = {
            'gnucash': {'database_path': os.path.join(temp_dir, 'test.sqlite')}
        }
    
    async def test_opencog_agent_plugin(self):
        """Test OpenCog agent plugin"""
        plugin = OpenCogAgentPlugin(self.opencog_config)
        
        # Test registration
        success = await self.plugin_manager.register_plugin(plugin)
        self.assertTrue(success)
        
        # Test initialization
        success = await self.plugin_manager.enable_plugin('opencog_agent')
        self.assertTrue(success)
        self.assertTrue(plugin.enabled)
        
        # Test message processing
        test_message = {
            'content': 'What are my recent financial transactions?',
            'type': 'text',
            'context': {'type': 'financial'}
        }
        
        result = await plugin.process_message(test_message)
        self.assertIn('reasoning_results', result)
        self.assertIn('atomspace_stats', result)
    
    async def test_financial_cognitive_plugin(self):
        """Test financial cognitive plugin"""
        plugin = FinancialCognitivePlugin(self.financial_config)
        
        # Test registration and initialization
        await self.plugin_manager.register_plugin(plugin)
        success = await self.plugin_manager.enable_plugin('financial_cognitive')
        self.assertTrue(success)
        
        # Test balance query
        balance_message = {
            'content': 'What is my account balance?',
            'type': 'text'
        }
        
        result = await plugin.process_message(balance_message)
        self.assertEqual(result['query_type'], 'balance')
        self.assertIn('response', result)
        self.assertIn('cognitive_analysis', result)
        
        # Test spending query
        spending_message = {
            'content': 'How much did I spend on groceries?',
            'type': 'text'
        }
        
        result = await plugin.process_message(spending_message)
        self.assertEqual(result['query_type'], 'spending')
    
    async def test_plugin_manager_workflow(self):
        """Test complete plugin manager workflow"""
        # Register plugins
        opencog_plugin = OpenCogAgentPlugin(self.opencog_config)
        financial_plugin = FinancialCognitivePlugin(self.financial_config)
        
        await self.plugin_manager.register_plugin(opencog_plugin)
        await self.plugin_manager.register_plugin(financial_plugin)
        
        # Enable plugins
        await self.plugin_manager.enable_plugin('opencog_agent')
        await self.plugin_manager.enable_plugin('financial_cognitive')
        
        # Process message through all plugins
        test_message = {
            'content': 'Analyze my spending patterns and provide insights',
            'type': 'text',
            'context': {'type': 'financial'}
        }
        
        results = await self.plugin_manager.process_message_through_plugins(test_message)
        self.assertEqual(len(results), 2)  # Results from both plugins
        
        # Verify each plugin processed the message
        plugin_names = [result['plugin'] for result in results if 'plugin' in result]
        self.assertIn('opencog_agent', plugin_names)
        self.assertIn('financial_cognitive', plugin_names)


class TestFinancialPatternAnalyzer(unittest.IsolatedAsyncioTestCase):
    """Test financial pattern analysis"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, 'test_analysis.sqlite')
        
        self.data_access = GnuCashDataAccess(db_path)
        await self.data_access.initialize()
        
        self.analyzer = FinancialPatternAnalyzer(self.data_access)
    
    async def asyncTearDown(self):
        """Clean up test environment"""
        await self.data_access.close()
    
    async def test_spending_trends_analysis(self):
        """Test spending trends analysis"""
        trends = await self.analyzer.analyze_spending_trends(months=3)
        
        required_keys = [
            'period', 'total_spending', 'category_breakdown',
            'category_percentages', 'top_categories', 'analysis_insights'
        ]
        
        for key in required_keys:
            self.assertIn(key, trends)
        
        self.assertIsInstance(trends['analysis_insights'], list)
        self.assertIsInstance(trends['top_categories'], list)
    
    async def test_anomaly_detection(self):
        """Test anomaly detection"""
        # Get a bank account for testing
        accounts = await self.data_access.get_accounts('BANK')
        if accounts:
            account_guid = accounts[0]['guid']
            anomalies = await self.analyzer.detect_anomalies(account_guid, days=30)
            self.assertIsInstance(anomalies, list)


class IntegrationTestSuite:
    """Main integration test suite runner"""
    
    def __init__(self):
        self.test_suites = [
            TestAtomSpaceCore,
            TestFinancialAtomSpace,
            TestGnuCashDataAccess,
            TestElizaOSPlugins,
            TestFinancialPatternAnalyzer
        ]
    
    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("🧪 Starting comprehensive integration test suite...")
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for test_class in self.test_suites:
            logger.info(f"\n--- Running {test_class.__name__} ---")
            
            # Create test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            
            # Run tests
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            total_tests += result.testsRun
            passed_tests += result.testsRun - len(result.failures) - len(result.errors)
            failed_tests += len(result.failures) + len(result.errors)
            
            if result.failures:
                logger.error(f"Failures in {test_class.__name__}:")
                for failure in result.failures:
                    logger.error(f"  - {failure[0]}: {failure[1]}")
            
            if result.errors:
                logger.error(f"Errors in {test_class.__name__}:")
                for error in result.errors:
                    logger.error(f"  - {error[0]}: {error[1]}")
        
        # Print summary
        logger.info(f"\n🏁 Integration Test Summary:")
        logger.info(f"   Total tests run: {total_tests}")
        logger.info(f"   Tests passed: {passed_tests}")
        logger.info(f"   Tests failed: {failed_tests}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        logger.info(f"   Success rate: {success_rate:.1f}%")
        
        if failed_tests == 0:
            logger.info("✅ All integration tests passed!")
            return True
        else:
            logger.error("❌ Some integration tests failed!")
            return False


async def run_basic_integration_demo():
    """Run a basic integration demonstration"""
    logger.info("🚀 Running basic integration demo...")
    
    try:
        # Initialize core components
        atomspace = AtomSpaceCore()
        await atomspace.initialize()
        
        financial_atomspace = FinancialAtomSpace()
        await financial_atomspace.initialize()
        
        # Create sample financial data
        checking_account = financial_atomspace.create_account_atom(
            "DemoChecking", "BANK", 2500.00
        )
        
        transaction = financial_atomspace.create_transaction_link(
            "DemoChecking", "Groceries", 125.75, "Weekly grocery shopping"
        )
        
        # Test plugin system
        plugin_manager = ElizaOSPluginManager()
        
        # Register and enable OpenCog plugin
        opencog_plugin = OpenCogAgentPlugin({
            'atomspace': {'host': 'localhost', 'port': 17001}
        })
        await plugin_manager.register_plugin(opencog_plugin)
        await plugin_manager.enable_plugin('opencog_agent')
        
        # Process test message
        test_message = {
            'content': 'Analyze my financial data and provide insights',
            'type': 'text',
            'context': {'type': 'financial'}
        }
        
        results = await plugin_manager.process_message_through_plugins(test_message)
        
        logger.info("✅ Basic integration demo completed successfully!")
        logger.info(f"   AtomSpace atoms: {atomspace.get_atom_count()}")
        logger.info(f"   Financial atoms: {financial_atomspace.get_atom_count()}")
        logger.info(f"   Plugin results: {len(results)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration demo failed: {e}")
        return False


if __name__ == "__main__":
    """Run tests when executed directly"""
    async def main():
        # Run basic demo first
        demo_success = await run_basic_integration_demo()
        
        if demo_success:
            # Run comprehensive test suite
            test_suite = IntegrationTestSuite()
            test_success = await test_suite.run_all_tests()
            
            if test_success:
                logger.info("🎉 All integration tests and demos completed successfully!")
                exit(0)
            else:
                logger.error("💥 Some tests failed!")
                exit(1)
        else:
            logger.error("💥 Basic integration demo failed!")
            exit(1)
    
    # Run the async main function
    asyncio.run(main())