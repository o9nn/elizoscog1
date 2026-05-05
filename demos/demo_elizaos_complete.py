#!/usr/bin/env python3
"""
Complete ElizaOS Integration Demo

Demonstrates all the implemented ElizaOS features integrated with the 
OpenCog-GnuCash cognitive-financial intelligence framework.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the complete framework
from src.integration.master_integration import HybridCognitiveFinancialFramework
from src.elizaos.integration import create_default_elizaos_config

async def demo_elizaos_features():
    """Comprehensive demo of all ElizaOS features"""
    
    print("🌟" * 50)
    print("     COMPLETE ELIZAOS INTEGRATION DEMO")
    print("   OpenCog-GnuCash-ElizaOS Framework")
    print("🌟" * 50)
    
    # Create configuration with ElizaOS features enabled
    config = {
        'elizaos': {
            'models': {
                'openai': {
                    'enabled': True,
                    'model_name': 'gpt-3.5-turbo',
                    'api_key': None  # Would be read from environment in real use
                }
            },
            'connectors': {
                'discord': {
                    'enabled': True,
                    'bot_token': 'mock_discord_token',
                    'application_id': 'mock_app_id'
                },
                'telegram': {
                    'enabled': True,
                    'bot_token': 'mock_telegram_token'
                }
            },
            'memory': {
                'db_path': 'data/demo_memory.db',
                'max_memory_items': 1000,
                'retention_days': 30
            },
            'dashboard': {
                'host': '0.0.0.0',
                'port': 3000,
                'debug': True
            }
        }
    }
    
    # Initialize the complete framework
    print("\n🚀 Initializing Complete ElizaOS Framework...")
    framework = HybridCognitiveFinancialFramework(config)
    
    # Initialize framework
    success = await framework.initialize()
    
    if not success:
        print("❌ Framework initialization failed!")
        return
    
    print("✅ Framework initialized successfully!")
    
    # Display system status
    print("\n📊 System Status:")
    status = await framework.get_system_status()
    print(f"  Integration Status: {json.dumps(status['integration_status'], indent=2)}")
    
    # Display ElizaOS specific status
    print("\n🤖 ElizaOS Status:")
    elizaos_status = await framework.get_elizaos_status()
    if 'error' not in elizaos_status:
        print(f"  Components: {elizaos_status['elizaos']['components']}")
        if 'connectors' in elizaos_status:
            print(f"  Connectors: {elizaos_status['connectors']}")
        if 'models' in elizaos_status:
            print(f"  Models: {elizaos_status['models']}")
        if 'dashboard' in elizaos_status:
            dashboard = elizaos_status['dashboard']
            print(f"  Dashboard: http://{dashboard['host']}:{dashboard['port']}")
    
    # Demo 1: Natural Language Processing
    print("\n" + "="*60)
    print("DEMO 1: Natural Language Message Processing")
    print("="*60)
    
    test_messages = [
        "What's my account balance?",
        "How much did I spend on groceries last month?",
        "Can you help me create a budget for next year?",
        "I think there might be an unusual transaction - can you analyze my recent spending?",
        "Send a message to Discord channel #general saying 'Hello from ElizaOS!'"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n💬 Test Message {i}: '{message}'")
        
        context = {
            'source': 'demo',
            'user_id': 'demo_user',
            'platform': 'api'
        }
        
        try:
            result = await framework.process_natural_language_message(message, context)
            print(f"🤖 Response: {result.get('response', 'No response')}")
            print(f"📋 Source: {result.get('source', 'Unknown')}")
            
            if 'model' in result:
                print(f"🧠 Model: {result['model']} ({result.get('provider', 'Unknown')})")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Demo 2: Action Execution
    print("\n" + "="*60)
    print("DEMO 2: ElizaOS Action System")
    print("="*60)
    
    # List available actions
    if framework.elizaos and framework.elizaos.action_executor:
        actions = framework.elizaos.action_registry.list_actions()
        print(f"\n📋 Available Actions ({len(actions)}):")
        for action in actions:
            print(f"  • {action['name']}: {action['description']}")
            for param in action['parameters']:
                req_str = "required" if param['required'] else "optional"
                print(f"    - {param['name']} ({param['type']}, {req_str}): {param['description']}")
    
    # Execute sample actions
    sample_actions = [
        {
            'name': 'financial_query',
            'parameters': {
                'query': 'Show me my spending summary for this month',
                'account_filter': 'checking'
            }
        },
        {
            'name': 'budget_analysis',
            'parameters': {
                'analysis_type': 'summary',
                'time_period': 'monthly'
            }
        },
        {
            'name': 'send_message',
            'parameters': {
                'platform': 'discord',
                'channel_id': 'general',
                'message': 'Hello from the ElizaOS action system!'
            }
        }
    ]
    
    for i, action_config in enumerate(sample_actions, 1):
        print(f"\n⚡ Executing Action {i}: {action_config['name']}")
        print(f"📝 Parameters: {json.dumps(action_config['parameters'], indent=2)}")
        
        try:
            result = await framework.execute_elizaos_action(
                action_config['name'], 
                action_config['parameters']
            )
            
            if result['success']:
                print(f"✅ Action completed successfully")
                print(f"📊 Result: {json.dumps(result.get('data', {}), indent=2)}")
                print(f"⏱️ Execution time: {result.get('execution_time', 0):.3f}s")
            else:
                print(f"❌ Action failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Action execution error: {e}")
    
    # Demo 3: Memory System
    print("\n" + "="*60)
    print("DEMO 3: Enhanced Memory Management")
    print("="*60)
    
    if framework.elizaos and framework.elizaos.memory_manager:
        memory_manager = framework.elizaos.memory_manager
        
        # Store some sample memories
        sample_memories = [
            {
                'content': 'User asked about account balance',
                'content_type': 'conversation',
                'source': 'demo',
                'metadata': {'user_id': 'demo_user', 'intent': 'balance_inquiry'}
            },
            {
                'content': 'Monthly budget analysis completed',
                'content_type': 'financial',
                'source': 'budget_agent',
                'metadata': {'analysis_type': 'monthly', 'success': True}
            },
            {
                'content': 'User spent $45.67 at grocery store',
                'content_type': 'financial',
                'source': 'transaction_analysis',
                'metadata': {'amount': 45.67, 'category': 'groceries'}
            }
        ]
        
        print("\n💾 Storing sample memories...")
        stored_ids = []
        for memory in sample_memories:
            memory_id = await memory_manager.store_memory(**memory)
            stored_ids.append(memory_id)
            print(f"  📝 Stored: {memory['content'][:50]}... (ID: {memory_id[:8]})")
        
        # Retrieve memories
        print(f"\n🔍 Retrieving memories...")
        memories = await memory_manager.retrieve_memories(limit=5)
        print(f"  Found {len(memories)} memories:")
        for memory in memories:
            print(f"  • {memory.content[:60]}... ({memory.content_type}, importance: {memory.importance_score:.2f})")
        
        # Search memories
        print(f"\n🔍 Searching for 'budget' memories...")
        budget_memories = await memory_manager.retrieve_memories(query='budget', limit=3)
        print(f"  Found {len(budget_memories)} budget-related memories:")
        for memory in budget_memories:
            print(f"  • {memory.content[:60]}... (source: {memory.source})")
        
        # Memory statistics
        print(f"\n📊 Memory Statistics:")
        stats = await memory_manager.get_memory_statistics()
        print(f"  Total memories: {stats.get('total_memories', 0)}")
        print(f"  By type: {stats.get('by_type', {})}")
        print(f"  Average importance: {stats.get('average_importance', 0)}")
        print(f"  Recent (24h): {stats.get('recent_memories_24h', 0)}")
    
    # Final status summary
    print("\n" + "🎉" * 50)
    print("        ELIZAOS INTEGRATION DEMO COMPLETE!")
    print("🎉" * 50)
    
    print(f"\n✅ Successfully demonstrated:")
    print(f"  🤖 Natural language processing with cognitive agents")
    print(f"  ⚡ Extensible action system with built-in actions")
    print(f"  💾 Enhanced memory management with semantic search")
    print(f"  🌐 Web dashboard with management interface")
    print(f"  📱 Social platform connectors (Discord, Telegram)")
    print(f"  🧠 Multi-model AI provider support")
    print(f"  📊 Performance monitoring and metrics")
    print(f"  🔗 Full integration with OpenCog-GnuCash framework")
    
    # Keep dashboard running for a moment
    print(f"\n⏳ Dashboard will remain accessible for 10 seconds...")
    if framework.elizaos and framework.elizaos.dashboard:
        print(f"🌐 Visit: http://localhost:{framework.elizaos.dashboard.port}")
    await asyncio.sleep(10)
    
    # Shutdown
    print(f"\n🛑 Shutting down framework...")
    await framework.shutdown()
    print(f"✅ Demo complete!")

if __name__ == "__main__":
    try:
        asyncio.run(demo_elizaos_features())
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.exception("Demo error")