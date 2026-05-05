#!/usr/bin/env python3
"""
Complete ElizaOS-OpenCog-GnuCash Integration Framework Demo
Demonstrates the full cognitive-financial intelligence system
"""

import asyncio
import logging
from pathlib import Path
import sys

# Import the master integration framework
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from integration.master_integration import HybridCognitiveFinancialFramework

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def comprehensive_integration_demo():
    """Run comprehensive integration framework demonstration"""
    
    logger.info("🌟 ElizaOS-OpenCog-GnuCash Integration Framework")
    logger.info("🧠💰 Comprehensive Cognitive-Financial Intelligence Demo")
    logger.info("=" * 70)
    
    # Configuration for the framework
    config = {
        'atomspace': {
            'host': 'localhost',
            'port': 17001
        },
        'financial_atomspace': {
            'enable_reasoning': True
        },
        'gnucash': {
            'database_path': 'data/demo_financial.sqlite'
        },
        'opencog_plugin': {
            'reasoning_agents': ['pattern_recognition', 'logic_reasoning', 'memory']
        },
        'financial_plugin': {
            'enable_cognitive_analysis': True
        }
    }
    
    try:
        # Initialize the complete framework
        framework = HybridCognitiveFinancialFramework(config)
        
        logger.info("🚀 Initializing complete integration framework...")
        success = await framework.initialize()
        
        if not success:
            logger.error("❌ Framework initialization failed")
            return False
        
        # Get system status
        status = await framework.get_system_status()
        logger.info("\n📊 System Status:")
        logger.info(f"   • AtomSpace atoms: {status['component_stats']['atomspace_atoms']}")
        logger.info(f"   • Financial atoms: {status['component_stats']['financial_atoms']}")
        logger.info(f"   • Active plugins: {status['component_stats']['active_plugins']}")
        logger.info(f"   • Cognitive agents: {status['component_stats']['cognitive_agents']}")
        logger.info(f"   • GnuCash operational: {status['component_stats']['gnucash_operational']}")
        
        # Demonstrate cognitive financial queries
        logger.info("\n🧠 Cognitive Financial Query Processing Demo:")
        logger.info("-" * 50)
        
        # Test queries demonstrating different capabilities
        test_queries = [
            {
                'query': 'What is my current account balance?',
                'context': {'query_type': 'balance_inquiry'}
            },
            {
                'query': 'How much did I spend on groceries last month?',
                'context': {'query_type': 'spending_analysis', 'category': 'groceries', 'timeframe': 'monthly'}
            },
            {
                'query': 'Are there any unusual transactions in my account?',
                'context': {'query_type': 'anomaly_detection'}
            },
            {
                'query': 'Can you help me plan my budget for next month?',
                'context': {'query_type': 'budget_planning', 'timeframe': 'monthly'}
            },
            {
                'query': 'What patterns do you see in my spending habits?',
                'context': {'query_type': 'pattern_analysis'}
            }
        ]
        
        processed_queries = 0
        for i, test_case in enumerate(test_queries, 1):
            logger.info(f"\n🔍 Query {i}: {test_case['query']}")
            
            try:
                response = await framework.process_financial_query(
                    test_case['query'], 
                    test_case['context']
                )
                
                # Display results
                plugin_count = len(response['plugin_results'])
                agent_count = len(response['cognitive_analysis']['active_agents'])
                insight_count = len(response['cognitive_analysis']['insights'])
                
                logger.info(f"   ✓ Processed by {plugin_count} plugins")
                logger.info(f"   ✓ Analyzed by {agent_count} cognitive agents")
                logger.info(f"   ✓ Generated {insight_count} insights")
                
                # Show cognitive insights
                for insight in response['cognitive_analysis']['insights']:
                    logger.info(f"   💡 {insight['agent']}: {insight['insight']} (confidence: {insight['confidence']:.1%})")
                
                processed_queries += 1
                
            except Exception as e:
                logger.error(f"   ❌ Query processing failed: {e}")
        
        # Demonstrate cognitive agents capabilities
        logger.info(f"\n🤖 Cognitive Agents Status:")
        logger.info("-" * 30)
        
        for agent_name, agent_info in status['cognitive_agents'].items():
            status_icon = "✅" if agent_info['active'] else "⏸️"
            capabilities = ", ".join(agent_info['capabilities'])
            logger.info(f"   {status_icon} {agent_name}")
            logger.info(f"      Capabilities: {capabilities}")
        
        # Integration summary
        logger.info(f"\n📈 Integration Summary:")
        logger.info("-" * 25)
        logger.info(f"   • Successfully processed {processed_queries}/{len(test_queries)} queries")
        logger.info(f"   • {status['component_stats']['cognitive_agents']} cognitive agents active")
        logger.info(f"   • {status['component_stats']['active_plugins']} plugins operational")
        logger.info(f"   • {status['component_stats']['atomspace_atoms'] + status['component_stats']['financial_atoms']} total atoms in knowledge base")
        
        # Phase roadmap status
        logger.info(f"\n🎯 Phase Implementation Status:")
        logger.info("-" * 35)
        for phase, status_val in status['integration_status'].items():
            status_icon = "✅" if status_val == 'active' else "⏸️"
            phase_name = phase.replace('_', ' ').title()
            logger.info(f"   {status_icon} {phase_name}: {status_val}")
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 COMPREHENSIVE INTEGRATION DEMO COMPLETED SUCCESSFULLY!")
        logger.info("🚀 ElizaOS-OpenCog-GnuCash Integration Framework is OPERATIONAL")
        logger.info("🧠💰 World's first cognitive-financial intelligence system is LIVE!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main demo runner"""
    success = await comprehensive_integration_demo()
    
    if success:
        logger.info("\n🌟 Demo completed successfully - integration framework operational!")
        return 0
    else:
        logger.error("\n💥 Demo encountered issues - check logs for details")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)