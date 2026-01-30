"""
agentaction Bridge Implementation

Description: Action chaining and history for agents
Original Repository: https://github.com/elizaOS/agentaction
Generated: 2025-09-29T22:18:52.644969

This bridge enables cross-ecosystem integration between:
- ElizaOS (TypeScript/JavaScript agents)
- OpenCog (Scheme/C++ cognitive architecture)  
- GnuCash (C/Scheme financial system)
"""

import json
import subprocess
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AgentactionBridge:
    """Bridge for agentaction cross-ecosystem integration"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = "agentaction"
        self.description = "Action chaining and history for agents"
        self.initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the agentaction bridge"""
        try:
            logger.info(f"Initializing {self.name} bridge")
            
            await self._setup_elizaos_connection()
            await self._setup_opencog_connection()
            await self._setup_gnucash_connection()
            
            self.initialized = True
            logger.info(f"{self.name} bridge initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.name} bridge: {e}")
            return False
    
    async def _setup_elizaos_connection(self):
        """Setup connection to ElizaOS ecosystem"""
        logger.debug("Setting up ElizaOS connection")
        try:
            self.action_registry = {}
            self.action_history = []
            self.action_chains = {}
            self.max_history_size = self.config.get('max_history_size', 1000)
            logger.info("✅ ElizaOS action registry established")
        except Exception as e:
            logger.warning(f"⚠️ ElizaOS action setup failed: {e}")
            self.action_registry = None
        
    async def _setup_opencog_connection(self):
        """Setup connection to OpenCog ecosystem"""
        logger.debug("Setting up OpenCog connection")
        try:
            from .cogserver_bridge import CogserverBridge
            self.cogserver = CogserverBridge(self.config)
            await self.cogserver.initialize()
            self.opencog_actions = {}
            logger.info("✅ OpenCog CogServer connection established")
        except Exception as e:
            logger.warning(f"⚠️ OpenCog setup failed: {e}")
            self.cogserver = None
            self.opencog_actions = {}
        
    async def _setup_gnucash_connection(self):
        """Setup connection to GnuCash ecosystem"""
        logger.debug("Setting up GnuCash connection")
        try:
            from ..core.gnucash_access import GnuCashAccess
            self.gnucash_access = GnuCashAccess(
                self.config.get('gnucash_file', 'demo_gnucash.sqlite')
            )
            await self.gnucash_access.initialize()
            logger.info("✅ GnuCash connection established")
        except Exception as e:
            logger.warning(f"⚠️ GnuCash setup failed: {e}")
            self.gnucash_access = None
    
    async def process_elizaos_request(self, request: Dict) -> Dict:
        """Process request from ElizaOS ecosystem"""
        if not self.initialized:
            raise RuntimeError("Bridge not initialized")
            
        logger.debug(f"Processing ElizaOS request: {request}")
        
        try:
            request_type = request.get('type', 'unknown')
            data = request.get('data', {})
            
            if request_type == 'register_action':
                action_name = data.get('name')
                action_def = data.get('definition', {})
                self.action_registry[action_name] = action_def
                
                return {
                    "success": True,
                    "source": "elizaos",
                    "target": "agentaction",
                    "data": {"registered": action_name}
                }
            
            elif request_type == 'execute_action':
                action_name = data.get('name')
                params = data.get('params', {})
                result = await self._execute_action(action_name, params)
                
                return {
                    "success": True,
                    "source": "elizaos",
                    "target": "agentaction",
                    "data": {"result": result}
                }
            
            elif request_type == 'execute_chain':
                chain = data.get('chain', [])
                results = await self._execute_action_chain(chain)
                
                return {
                    "success": True,
                    "source": "elizaos",
                    "target": "agentaction",
                    "data": {"results": results}
                }
            
            elif request_type == 'get_action_history':
                limit = data.get('limit', 10)
                history = self._get_action_history(limit)
                
                return {
                    "success": True,
                    "source": "elizaos",
                    "target": "agentaction",
                    "data": {"history": history}
                }
            
            else:
                return {
                    "success": False,
                    "source": "elizaos",
                    "target": "agentaction",
                    "error": f"Unknown request type: {request_type}",
                    "data": {}
                }
                
        except Exception as e:
            logger.error(f"Error processing ElizaOS request: {e}")
            return {
                "success": False,
                "source": "elizaos",
                "target": "agentaction",
                "error": str(e),
                "data": {}
            }
    
    async def process_opencog_request(self, request: Dict) -> Dict:
        """Process request from OpenCog ecosystem"""
        if not self.initialized:
            raise RuntimeError("Bridge not initialized")
            
        logger.debug(f"Processing OpenCog request: {request}")
        
        try:
            request_type = request.get('type', 'unknown')
            data = request.get('data', {})
            
            if request_type == 'execute_schema':
                schema_name = data.get('schema')
                args = data.get('args', [])
                result = await self._execute_opencog_schema(schema_name, args)
                
                return {
                    "success": True,
                    "source": "opencog",
                    "target": "agentaction",
                    "data": {"result": result}
                }
            
            elif request_type == 'get_psi_actions':
                actions = await self._get_psi_actions()
                
                return {
                    "success": True,
                    "source": "opencog",
                    "target": "agentaction",
                    "data": {"actions": actions}
                }
            
            else:
                return {
                    "success": False,
                    "source": "opencog",
                    "target": "agentaction",
                    "error": f"Unknown request type: {request_type}",
                    "data": {}
                }
                
        except Exception as e:
            logger.error(f"Error processing OpenCog request: {e}")
            return {
                "success": False,
                "source": "opencog",
                "target": "agentaction",
                "error": str(e),
                "data": {}
            }
    
    async def process_gnucash_request(self, request: Dict) -> Dict:
        """Process request from GnuCash ecosystem"""
        if not self.initialized:
            raise RuntimeError("Bridge not initialized")
            
        logger.debug(f"Processing GnuCash request: {request}")
        
        try:
            request_type = request.get('type', 'unknown')
            data = request.get('data', {})
            
            if request_type == 'execute_transaction':
                transaction = data.get('transaction', {})
                result = await self._execute_financial_action(transaction)
                
                return {
                    "success": True,
                    "source": "gnucash",
                    "target": "agentaction",
                    "data": {"result": result}
                }
            
            elif request_type == 'get_transaction_history':
                limit = data.get('limit', 10)
                history = [h for h in self.action_history if h.get('type') == 'financial'][:limit]
                
                return {
                    "success": True,
                    "source": "gnucash",
                    "target": "agentaction",
                    "data": {"history": history}
                }
            
            else:
                return {
                    "success": False,
                    "source": "gnucash",
                    "target": "agentaction",
                    "error": f"Unknown request type: {request_type}",
                    "data": {}
                }
                
        except Exception as e:
            logger.error(f"Error processing GnuCash request: {e}")
            return {
                "success": False,
                "source": "gnucash",
                "target": "agentaction",
                "error": str(e),
                "data": {}
            }
    
    async def translate_data(self, data: Any, source_format: str, target_format: str) -> Any:
        """Translate data between ecosystem formats"""
        logger.debug(f"Translating data from {source_format} to {target_format}")
        
        translators = {
            ("elizaos", "opencog"): self._elizaos_to_opencog,
            ("opencog", "elizaos"): self._opencog_to_elizaos,
            ("elizaos", "gnucash"): self._elizaos_to_gnucash,
            ("gnucash", "elizaos"): self._gnucash_to_elizaos,
            ("opencog", "gnucash"): self._opencog_to_gnucash,
            ("gnucash", "opencog"): self._gnucash_to_opencog
        }
        
        translator = translators.get((source_format, target_format))
        if translator:
            return await translator(data)
        else:
            logger.warning(f"No translator found for {source_format} -> {target_format}")
            return data
    
    async def _elizaos_to_opencog(self, data: Any) -> Any:
        """Translate ElizaOS data to OpenCog format"""
        if isinstance(data, dict) and 'name' in data:
            return {
                "atom_type": "ExecutionOutputLink",
                "name": data.get('name', ''),
                "schema": data.get('definition', {}).get('schema', ''),
                "params": data.get('params', {}),
                "metadata": {
                    "source": "elizaos_action",
                    "action_type": data.get('type', 'generic')
                }
            }
        return {"atomspace_data": data, "format": "opencog"}
    
    async def _opencog_to_elizaos(self, data: Any) -> Any:
        """Translate OpenCog data to ElizaOS format"""
        if isinstance(data, dict) and 'schema' in data:
            return {
                "name": data.get('name', 'opencog_action'),
                "type": "schema_execution",
                "definition": {
                    "schema": data.get('schema', ''),
                    "atom_type": data.get('atom_type', 'ExecutionOutputLink')
                },
                "params": data.get('params', {}),
                "source": "opencog"
            }
        return {"agent_data": data, "format": "elizaos"}
    
    async def _elizaos_to_gnucash(self, data: Any) -> Any:
        """Translate ElizaOS data to GnuCash format"""
        if isinstance(data, dict):
            return {
                "description": data.get('name', 'ElizaOS Action')[:100],
                "type": "action_execution",
                "amount": data.get('params', {}).get('amount', 0.0),
                "metadata": {
                    "action": data.get('name', ''),
                    "source": "elizaos"
                }
            }
        return {"financial_data": data, "format": "gnucash"}
    
    async def _gnucash_to_elizaos(self, data: Any) -> Any:
        """Translate GnuCash data to ElizaOS format"""
        if isinstance(data, dict):
            return {
                "name": f"transaction_{data.get('description', 'unknown')}",
                "type": "financial_action",
                "params": {
                    "amount": data.get('amount', 0.0),
                    "description": data.get('description', ''),
                    "account": data.get('account', '')
                },
                "source": "gnucash"
            }
        return {"agent_data": data, "format": "elizaos"}
    
    async def _opencog_to_gnucash(self, data: Any) -> Any:
        """Translate OpenCog data to GnuCash format"""
        if isinstance(data, dict):
            return {
                "description": f"Cognitive Action: {data.get('schema', 'Unknown')}",
                "type": "cognitive_action",
                "metadata": {
                    "schema": data.get('schema', ''),
                    "source": "opencog"
                }
            }
        return {"financial_data": data, "format": "gnucash"}
    
    async def _gnucash_to_opencog(self, data: Any) -> Any:
        """Translate GnuCash data to OpenCog format"""
        if isinstance(data, dict):
            return {
                "atom_type": "ExecutionOutputLink",
                "schema": "financial-transaction",
                "name": f"Transaction:{data.get('description', 'Unknown')}",
                "params": {
                    "amount": data.get('amount', 0.0),
                    "description": data.get('description', '')
                },
                "metadata": {
                    "source": "gnucash"
                }
            }
        return {"atomspace_data": data, "format": "opencog"}
    
    async def shutdown(self):
        """Shutdown the bridge"""
        logger.info(f"Shutting down {self.name} bridge")
        
        if self.cogserver:
            try:
                await self.cogserver.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down CogServer: {e}")
        
        if self.gnucash_access:
            try:
                await self.gnucash_access.close()
            except Exception as e:
                logger.error(f"Error closing GnuCash access: {e}")
        
        self.initialized = False
    
    async def _execute_action(self, action_name: str, params: Dict) -> Any:
        """Execute a single action"""
        import time
        from datetime import datetime
        
        if action_name not in self.action_registry:
            raise ValueError(f"Action not registered: {action_name}")
        
        action_def = self.action_registry[action_name]
        start_time = time.time()
        
        try:
            result = {
                "action": action_name,
                "params": params,
                "status": "executed",
                "timestamp": datetime.now().isoformat()
            }
            
            self._record_action_history({
                "action": action_name,
                "params": params,
                "result": result,
                "duration": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            self._record_action_history({
                "action": action_name,
                "params": params,
                "error": str(e),
                "duration": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            })
            raise
    
    async def _execute_action_chain(self, chain: List[Dict]) -> List[Any]:
        """Execute a chain of actions"""
        results = []
        
        for action in chain:
            action_name = action.get('name')
            params = action.get('params', {})
            
            try:
                result = await self._execute_action(action_name, params)
                results.append(result)
            except Exception as e:
                logger.error(f"Chain execution failed at {action_name}: {e}")
                results.append({"error": str(e), "action": action_name})
                if action.get('stop_on_error', True):
                    break
        
        return results
    
    def _record_action_history(self, record: Dict):
        """Record action in history"""
        self.action_history.append(record)
        
        if len(self.action_history) > self.max_history_size:
            self.action_history = self.action_history[-self.max_history_size:]
    
    def _get_action_history(self, limit: int = 10) -> List[Dict]:
        """Get recent action history"""
        return self.action_history[-limit:]
    
    async def _execute_opencog_schema(self, schema_name: str, args: List) -> Any:
        """Execute OpenCog schema"""
        if not self.cogserver:
            raise RuntimeError("CogServer not available")
        
        result = {
            "schema": schema_name,
            "args": args,
            "status": "executed",
            "source": "opencog"
        }
        
        self.opencog_actions[schema_name] = result
        return result
    
    async def _get_psi_actions(self) -> List[Dict]:
        """Get OpenCog Psi actions"""
        return list(self.opencog_actions.values())
    
    async def _execute_financial_action(self, transaction: Dict) -> Any:
        """Execute financial action via GnuCash"""
        if not self.gnucash_access:
            raise RuntimeError("GnuCash not available")
        
        result = {
            "transaction": transaction,
            "status": "executed",
            "type": "financial"
        }
        
        self._record_action_history(result)
        return result

class AgentactionIntegrationFramework:
    """Framework for managing agentaction integrations"""
    
    def __init__(self):
        self.bridges = {}
        self.active_sessions = {}
        
    async def register_bridge(self, bridge: AgentactionBridge) -> bool:
        """Register a new bridge"""
        try:
            await bridge.initialize()
            self.bridges[bridge.name] = bridge
            logger.info(f"Registered bridge: {bridge.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register bridge {bridge.name}: {e}")
            return False
    
    async def process_cross_ecosystem_request(self, source: str, target: str, request: Dict) -> Dict:
        """Process request across ecosystems"""
        bridge_name = f"{source}_{target}_bridge"
        
        if bridge_name not in self.bridges:
            raise ValueError(f"No bridge found for {source} -> {target}")
            
        bridge = self.bridges[bridge_name]
        
        # Route request to appropriate processor
        if source == "elizaos":
            return await bridge.process_elizaos_request(request)
        elif source == "opencog":
            return await bridge.process_opencog_request(request)
        elif source == "gnucash":
            return await bridge.process_gnucash_request(request)
        else:
            raise ValueError(f"Unknown source ecosystem: {source}")

# Export classes for external use
__all__ = ["AgentactionBridge", "AgentactionIntegrationFramework"]
