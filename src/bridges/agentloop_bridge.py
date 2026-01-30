"""
agentloop Bridge Implementation

Description: A simple, lightweight loop for your agent
Original Repository: https://github.com/elizaOS/agentloop
Generated: 2025-09-29T22:18:52.644447

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

class AgentloopBridge:
    """Bridge for agentloop cross-ecosystem integration"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = "agentloop"
        self.description = "A simple, lightweight loop for your agent"
        self.initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the agentloop bridge"""
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
        self.active_loops = {}
        self.loop_schedules = {}
        self.loop_history = []
        self.max_history_size = self.config.get('max_history_size', 1000)
        try:
            logger.info("✅ ElizaOS loop manager established")
        except Exception as e:
            logger.warning(f"⚠️ ElizaOS loop setup warning: {e}")
        
    async def _setup_opencog_connection(self):
        """Setup connection to OpenCog ecosystem"""
        logger.debug("Setting up OpenCog connection")
        try:
            from .cogserver_bridge import CogserverBridge
            self.cogserver = CogserverBridge(self.config)
            await self.cogserver.initialize()
            self.cognitive_loops = {}
            logger.info("✅ OpenCog CogServer connection established")
        except Exception as e:
            logger.warning(f"⚠️ OpenCog setup failed: {e}")
            self.cogserver = None
            self.cognitive_loops = {}
        
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
            
            if request_type == 'start_loop':
                loop_id = data.get('loop_id', f"loop_{len(self.active_loops)}")
                loop_config = data.get('config', {})
                result = await self._start_agent_loop(loop_id, loop_config)
                
                return {
                    "success": True,
                    "source": "elizaos",
                    "target": "agentloop",
                    "data": {"loop_id": loop_id, "started": result}
                }
            
            elif request_type == 'stop_loop':
                loop_id = data.get('loop_id')
                result = await self._stop_agent_loop(loop_id)
                
                return {
                    "success": True,
                    "source": "elizaos",
                    "target": "agentloop",
                    "data": {"loop_id": loop_id, "stopped": result}
                }
            
            elif request_type == 'get_loop_status':
                loop_id = data.get('loop_id')
                status = self._get_loop_status(loop_id)
                
                return {
                    "success": True,
                    "source": "elizaos",
                    "target": "agentloop",
                    "data": {"status": status}
                }
            
            elif request_type == 'list_loops':
                loops = list(self.active_loops.keys())
                
                return {
                    "success": True,
                    "source": "elizaos",
                    "target": "agentloop",
                    "data": {"loops": loops}
                }
            
            else:
                return {
                    "success": False,
                    "source": "elizaos",
                    "target": "agentloop",
                    "error": f"Unknown request type: {request_type}",
                    "data": {}
                }
                
        except Exception as e:
            logger.error(f"Error processing ElizaOS request: {e}")
            return {
                "success": False,
                "source": "elizaos",
                "target": "agentloop",
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
            
            if request_type == 'start_cognitive_loop':
                loop_id = data.get('loop_id', f"cog_loop_{len(self.cognitive_loops)}")
                loop_type = data.get('loop_type', 'psi')
                result = await self._start_cognitive_loop(loop_id, loop_type)
                
                return {
                    "success": True,
                    "source": "opencog",
                    "target": "agentloop",
                    "data": {"loop_id": loop_id, "started": result}
                }
            
            elif request_type == 'get_cognitive_state':
                state = await self._get_cognitive_state()
                
                return {
                    "success": True,
                    "source": "opencog",
                    "target": "agentloop",
                    "data": {"state": state}
                }
            
            else:
                return {
                    "success": False,
                    "source": "opencog",
                    "target": "agentloop",
                    "error": f"Unknown request type: {request_type}",
                    "data": {}
                }
                
        except Exception as e:
            logger.error(f"Error processing OpenCog request: {e}")
            return {
                "success": False,
                "source": "opencog",
                "target": "agentloop",
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
            
            if request_type == 'start_financial_loop':
                loop_id = data.get('loop_id', 'financial_loop')
                interval = data.get('interval', 3600)
                result = await self._start_financial_loop(loop_id, interval)
                
                return {
                    "success": True,
                    "source": "gnucash",
                    "target": "agentloop",
                    "data": {"loop_id": loop_id, "started": result}
                }
            
            else:
                return {
                    "success": False,
                    "source": "gnucash",
                    "target": "agentloop",
                    "error": f"Unknown request type: {request_type}",
                    "data": {}
                }
                
        except Exception as e:
            logger.error(f"Error processing GnuCash request: {e}")
            return {
                "success": False,
                "source": "gnucash",
                "target": "agentloop",
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
        if isinstance(data, dict) and 'loop_id' in data:
            return {
                "atom_type": "ConceptNode",
                "name": f"AgentLoop:{data.get('loop_id', 'unknown')}",
                "metadata": {
                    "source": "elizaos_loop",
                    "status": data.get('status', 'unknown'),
                    "interval": data.get('interval', 0)
                }
            }
        return {"atomspace_data": data, "format": "opencog"}
    
    async def _opencog_to_elizaos(self, data: Any) -> Any:
        """Translate OpenCog data to ElizaOS format"""
        if isinstance(data, dict) and 'loop_type' in data:
            return {
                "loop_id": data.get('name', 'opencog_loop'),
                "type": "cognitive_loop",
                "loop_type": data.get('loop_type', 'psi'),
                "source": "opencog",
                "metadata": data.get('metadata', {})
            }
        return {"agent_data": data, "format": "elizaos"}
    
    async def _elizaos_to_gnucash(self, data: Any) -> Any:
        """Translate ElizaOS data to GnuCash format"""
        if isinstance(data, dict):
            return {
                "description": f"Agent Loop: {data.get('loop_id', 'unknown')}",
                "type": "loop_execution",
                "metadata": {
                    "loop_id": data.get('loop_id', ''),
                    "source": "elizaos"
                }
            }
        return {"financial_data": data, "format": "gnucash"}
    
    async def _gnucash_to_elizaos(self, data: Any) -> Any:
        """Translate GnuCash data to ElizaOS format"""
        if isinstance(data, dict):
            return {
                "loop_id": f"financial_loop_{data.get('id', 'unknown')}",
                "type": "financial_loop",
                "interval": data.get('interval', 3600),
                "source": "gnucash"
            }
        return {"agent_data": data, "format": "elizaos"}
    
    async def _opencog_to_gnucash(self, data: Any) -> Any:
        """Translate OpenCog data to GnuCash format"""
        if isinstance(data, dict):
            return {
                "description": f"Cognitive Loop: {data.get('loop_type', 'Unknown')}",
                "type": "cognitive_loop",
                "metadata": {
                    "loop_type": data.get('loop_type', ''),
                    "source": "opencog"
                }
            }
        return {"financial_data": data, "format": "gnucash"}
    
    async def _gnucash_to_opencog(self, data: Any) -> Any:
        """Translate GnuCash data to OpenCog format"""
        if isinstance(data, dict):
            return {
                "atom_type": "ConceptNode",
                "name": f"FinancialLoop:{data.get('id', 'unknown')}",
                "metadata": {
                    "source": "gnucash",
                    "interval": data.get('interval', 0)
                }
            }
        return {"atomspace_data": data, "format": "opencog"}
    
    async def shutdown(self):
        """Shutdown the bridge"""
        logger.info(f"Shutting down {self.name} bridge")
        
        for loop_id in list(self.active_loops.keys()):
            await self._stop_agent_loop(loop_id)
        
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
    
    async def _start_agent_loop(self, loop_id: str, config: Dict) -> bool:
        """Start an agent loop"""
        from datetime import datetime
        
        if loop_id in self.active_loops:
            logger.warning(f"Loop {loop_id} already running")
            return False
        
        loop_state = {
            "id": loop_id,
            "config": config,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "iterations": 0
        }
        
        self.active_loops[loop_id] = loop_state
        self.loop_schedules[loop_id] = config.get('interval', 1.0)
        
        logger.info(f"Started loop {loop_id}")
        return True
    
    async def _stop_agent_loop(self, loop_id: str) -> bool:
        """Stop an agent loop"""
        if loop_id not in self.active_loops:
            logger.warning(f"Loop {loop_id} not found")
            return False
        
        loop_state = self.active_loops[loop_id]
        loop_state['status'] = 'stopped'
        
        self.loop_history.append(loop_state)
        
        del self.active_loops[loop_id]
        if loop_id in self.loop_schedules:
            del self.loop_schedules[loop_id]
        
        if len(self.loop_history) > self.max_history_size:
            self.loop_history = self.loop_history[-self.max_history_size:]
        
        logger.info(f"Stopped loop {loop_id}")
        return True
    
    def _get_loop_status(self, loop_id: str) -> Dict:
        """Get status of a loop"""
        if loop_id in self.active_loops:
            return self.active_loops[loop_id]
        
        for historical in reversed(self.loop_history):
            if historical['id'] == loop_id:
                return historical
        
        return {"status": "not_found"}
    
    async def _start_cognitive_loop(self, loop_id: str, loop_type: str) -> bool:
        """Start OpenCog cognitive loop"""
        if not self.cogserver:
            raise RuntimeError("CogServer not available")
        
        from datetime import datetime
        
        cognitive_state = {
            "id": loop_id,
            "type": loop_type,
            "status": "running",
            "started_at": datetime.now().isoformat()
        }
        
        self.cognitive_loops[loop_id] = cognitive_state
        logger.info(f"Started cognitive loop {loop_id} of type {loop_type}")
        return True
    
    async def _get_cognitive_state(self) -> Dict:
        """Get current cognitive state"""
        return {
            "active_loops": len(self.cognitive_loops),
            "loops": list(self.cognitive_loops.keys())
        }
    
    async def _start_financial_loop(self, loop_id: str, interval: int) -> bool:
        """Start financial monitoring loop"""
        if not self.gnucash_access:
            raise RuntimeError("GnuCash not available")
        
        config = {
            "type": "financial",
            "interval": interval
        }
        
        return await self._start_agent_loop(loop_id, config)

class AgentloopIntegrationFramework:
    """Framework for managing agentloop integrations"""
    
    def __init__(self):
        self.bridges = {}
        self.active_sessions = {}
        
    async def register_bridge(self, bridge: AgentloopBridge) -> bool:
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
__all__ = ["AgentloopBridge", "AgentloopIntegrationFramework"]
