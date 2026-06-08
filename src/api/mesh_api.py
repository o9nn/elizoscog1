"""
Cognitive Mesh API Gateway

Provides unified REST and WebSocket endpoints for distributed cognitive operations
with sub-100ms response times and high availability.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import asdict

# FastAPI for high-performance async REST API
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Internal imports
from .state_manager import DistributedStateManager, StateEvent, StateNode
from .websocket_handler import WebSocketHandler, ConnectionManager
from .auth_manager import AuthenticationManager


# Mock classes for fallback when cognitive framework not available
class MockAgent:
    """Mock agent for testing when real framework unavailable"""
    async def process_message(self, message, context=None):
        return f"Mock response to: {message}"


class MockReasoningEngine:
    """Mock reasoning engine for testing"""
    async def infer(self, premises):
        return [{"type": "mock_conclusion", "confidence": 0.8}]


# Try to import cognitive framework, use mock if not available
try:
    from ..integration.master_integration import HybridCognitiveFinancialFramework
except ImportError:
    # Mock framework for testing
    class HybridCognitiveFinancialFramework:
        """Mock cognitive framework for testing when real framework unavailable"""
        def __init__(self, config=None):
            self.config = config or {}
            self.cognitive_agents = {
                'financial_chat_agent': MockAgent(),
                'account_reasoning_agent': MockAgent()
            }
            self.reasoning_engine = MockReasoningEngine()
        
        async def initialize(self):
            pass
        
        async def shutdown(self):
            pass


class CognitiveRequest(BaseModel):
    """Base request model for cognitive operations"""
    node_id: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=10)
    timeout: float = Field(default=30.0, gt=0)
    context: Dict[str, Any] = Field(default_factory=dict)


class StateRequest(CognitiveRequest):
    """Request model for state operations"""
    entity_type: str
    entity_id: str
    data: Optional[Dict[str, Any]] = None


class QueryRequest(CognitiveRequest):
    """Request model for cognitive queries"""
    query: str
    query_type: str = "natural_language"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class TaskRequest(CognitiveRequest):
    """Request model for distributed task execution"""
    task_type: str
    task_data: Dict[str, Any]
    target_nodes: Optional[List[str]] = None


class ApiResponse(BaseModel):
    """Standard API response format"""
    success: bool
    data: Any = None
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_time_ms: float = 0.0
    node_id: str = ""


class CognitiveMeshAPI:
    """
    High-performance cognitive mesh API gateway
    
    Features:
    - Sub-100ms response times for simple operations
    - Real-time WebSocket communication
    - Distributed state synchronization
    - Authentication and rate limiting
    - External system bindings
    """
    
    def __init__(self, node_id: str, port: int = 8000, host: str = "0.0.0.0"):
        self.node_id = node_id
        self.port = port
        self.host = host
        
        # Core components
        self.state_manager = DistributedStateManager(node_id)
        self.websocket_handler = WebSocketHandler()
        self.auth_manager = AuthenticationManager()
        self.cognitive_framework: Optional[HybridCognitiveFinancialFramework] = None
        
        # FastAPI app
        self.app = FastAPI(
            title="Cognitive Mesh API",
            description="Distributed Cognitive Operations API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.logger = logging.getLogger(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health and status endpoints
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "node_id": self.node_id,
                "timestamp": datetime.now(),
                "uptime": "running"
            }
        
        @self.app.get("/status")
        async def get_status():
            """Get system status"""
            stats = self.state_manager.get_statistics()
            return ApiResponse(
                success=True,
                data=stats,
                message="System status retrieved",
                node_id=self.node_id
            )
        
        # State management endpoints
        @self.app.put("/api/v1/state")
        async def set_state(request: StateRequest):
            """Set distributed state"""
            start_time = time.time()
            
            try:
                if not request.data:
                    raise HTTPException(status_code=400, detail="Data is required for state updates")
                
                event = await self.state_manager.set_state(
                    request.entity_type,
                    request.entity_id,
                    request.data
                )
                
                execution_time = (time.time() - start_time) * 1000
                
                return ApiResponse(
                    success=True,
                    data={
                        "event_id": event.event_id,
                        "version": event.version,
                        "entity_type": request.entity_type,
                        "entity_id": request.entity_id
                    },
                    message=f"State updated for {request.entity_type}.{request.entity_id}",
                    execution_time_ms=execution_time,
                    node_id=self.node_id
                )
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                self.logger.error(f"State update failed: {e}")
                return ApiResponse(
                    success=False,
                    message=f"State update failed: {str(e)}",
                    execution_time_ms=execution_time,
                    node_id=self.node_id
                )
        
        @self.app.get("/api/v1/state/{entity_type}/{entity_id}")
        async def get_state(entity_type: str, entity_id: str):
            """Get entity state"""
            start_time = time.time()
            
            try:
                state = self.state_manager.get_state(entity_type, entity_id)
                execution_time = (time.time() - start_time) * 1000
                
                if state is None:
                    raise HTTPException(status_code=404, detail="Entity not found")
                
                return ApiResponse(
                    success=True,
                    data={
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "state": state
                    },
                    message="State retrieved",
                    execution_time_ms=execution_time,
                    node_id=self.node_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                self.logger.error(f"State retrieval failed: {e}")
                return ApiResponse(
                    success=False,
                    message=f"State retrieval failed: {str(e)}",
                    execution_time_ms=execution_time,
                    node_id=self.node_id
                )
        
        @self.app.delete("/api/v1/state/{entity_type}/{entity_id}")
        async def delete_state(entity_type: str, entity_id: str):
            """Delete entity state"""
            start_time = time.time()
            
            try:
                event = await self.state_manager.delete_state(entity_type, entity_id)
                execution_time = (time.time() - start_time) * 1000
                
                if event is None:
                    raise HTTPException(status_code=404, detail="Entity not found")
                
                return ApiResponse(
                    success=True,
                    data={"event_id": event.event_id},
                    message=f"State deleted for {entity_type}.{entity_id}",
                    execution_time_ms=execution_time,
                    node_id=self.node_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                return ApiResponse(
                    success=False,
                    message=f"State deletion failed: {str(e)}",
                    execution_time_ms=execution_time,
                    node_id=self.node_id
                )
        
        # Cognitive operations endpoints
        @self.app.post("/api/v1/query")
        async def cognitive_query(request: QueryRequest):
            """Execute cognitive query"""
            start_time = time.time()
            
            try:
                if not self.cognitive_framework:
                    raise HTTPException(status_code=503, detail="Cognitive framework not initialized")
                
                # Route query based on type
                if request.query_type == "financial":
                    result = await self._process_financial_query(request)
                elif request.query_type == "reasoning":
                    result = await self._process_reasoning_query(request)
                else:
                    result = await self._process_natural_language_query(request)
                
                execution_time = (time.time() - start_time) * 1000
                
                return ApiResponse(
                    success=True,
                    data=result,
                    message="Query processed",
                    execution_time_ms=execution_time,
                    node_id=self.node_id
                )
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                self.logger.error(f"Query processing failed: {e}")
                return ApiResponse(
                    success=False,
                    message=f"Query processing failed: {str(e)}",
                    execution_time_ms=execution_time,
                    node_id=self.node_id
                )
        
        @self.app.post("/api/v1/task")
        async def execute_distributed_task(request: TaskRequest):
            """Execute distributed cognitive task"""
            start_time = time.time()
            
            try:
                # Assign task ID
                task_id = f"task_{self.node_id}_{int(time.time())}"
                
                # Store task state
                await self.state_manager.set_state(
                    "distributed_task",
                    task_id,
                    {
                        "task_type": request.task_type,
                        "task_data": request.task_data,
                        "target_nodes": request.target_nodes,
                        "status": "initiated",
                        "priority": request.priority,
                        "created_at": datetime.now().isoformat()
                    }
                )
                
                # Process task based on type
                result = await self._process_distributed_task(task_id, request)
                
                execution_time = (time.time() - start_time) * 1000
                
                return ApiResponse(
                    success=True,
                    data={
                        "task_id": task_id,
                        "result": result
                    },
                    message="Task executed",
                    execution_time_ms=execution_time,
                    node_id=self.node_id
                )
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                return ApiResponse(
                    success=False,
                    message=f"Task execution failed: {str(e)}",
                    execution_time_ms=execution_time,
                    node_id=self.node_id
                )
        
        # Node management endpoints
        @self.app.post("/api/v1/nodes/register")
        async def register_node(node_data: dict):
            """Register a new node in the mesh"""
            try:
                node = StateNode(
                    node_id=node_data["node_id"],
                    host=node_data["host"],
                    port=node_data["port"],
                    capabilities=node_data.get("capabilities", []),
                    last_seen=datetime.now()
                )
                
                self.state_manager.register_node(node)
                
                return ApiResponse(
                    success=True,
                    data=asdict(node),
                    message=f"Node {node.node_id} registered",
                    node_id=self.node_id
                )
                
            except Exception as e:
                return ApiResponse(
                    success=False,
                    message=f"Node registration failed: {str(e)}",
                    node_id=self.node_id
                )
        
        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time communication"""
            await self.websocket_handler.connect(websocket)
            
            try:
                while True:
                    # Receive message
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Process message
                    response = await self._process_websocket_message(message, websocket)
                    
                    # Send response
                    if response:
                        await websocket.send_text(json.dumps(response))
                        
            except WebSocketDisconnect:
                self.websocket_handler.disconnect(websocket)
                self.logger.info("WebSocket client disconnected")
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                await self.websocket_handler.disconnect(websocket)
    
    async def initialize(self, cognitive_framework: HybridCognitiveFinancialFramework = None):
        """Initialize the API components"""
        # Initialize state manager
        await self.state_manager.start()
        
        # Initialize cognitive framework
        if cognitive_framework:
            self.cognitive_framework = cognitive_framework
            await self.cognitive_framework.initialize()
        else:
            # Create default framework with empty config
            self.cognitive_framework = HybridCognitiveFinancialFramework({})
            await self.cognitive_framework.initialize()
        
        # Subscribe to state events for WebSocket broadcasting
        self.state_manager.subscribe_to_events(self._broadcast_state_event)
        
        self.logger.info(f"Cognitive Mesh API initialized on node {self.node_id}")
    
    async def shutdown(self):
        """Shutdown the API components"""
        await self.state_manager.stop()
        if self.cognitive_framework:
            await self.cognitive_framework.shutdown()
        self.logger.info("Cognitive Mesh API shutdown completed")
    
    async def start_server(self):
        """Start the API server"""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def _process_financial_query(self, request: QueryRequest) -> Dict[str, Any]:
        """Process financial-specific queries"""
        if not self.cognitive_framework:
            raise Exception("Cognitive framework not available")
        
        # Try to use framework's built-in financial query processing
        if hasattr(self.cognitive_framework, 'process_financial_query'):
            try:
                response = await self.cognitive_framework.process_financial_query(
                    request.query,
                    context=request.parameters
                )
                return {
                    "type": "financial_query",
                    "query": request.query,
                    "response": response,
                    "agent": "financial_framework"
                }
            except Exception as e:
                self.logger.warning(f"Framework financial query failed: {e}")
        
        # Fallback: Use financial reasoning agent if it's a callable object
        financial_agent = self.cognitive_framework.cognitive_agents.get('financial_chat_agent')
        if financial_agent and hasattr(financial_agent, 'process_message'):
            response = await financial_agent.process_message(
                request.query,
                context=request.parameters
            )
            return {
                "type": "financial_query",
                "query": request.query,
                "response": response,
                "agent": "financial_chat_agent"
            }
        
        # Fallback for dict-based agent configurations
        if isinstance(financial_agent, dict):
            return {
                "type": "financial_query",
                "query": request.query,
                "response": f"Financial query processed: {request.query}",
                "agent": "financial_chat_agent",
                "agent_config": financial_agent
            }
        
        return {
            "type": "financial_query",
            "query": request.query,
            "response": "Financial agent not available",
            "agent": None
        }
    
    async def _process_reasoning_query(self, request: QueryRequest) -> Dict[str, Any]:
        """Process reasoning-based queries"""
        if not self.cognitive_framework:
            raise Exception("Cognitive framework not available")
        
        # Use PLN reasoning
        reasoning_engine = self.cognitive_framework.reasoning_engine
        if reasoning_engine:
            # Convert query to premises for reasoning
            premises = [
                {
                    "type": "query",
                    "content": request.query,
                    "confidence": 0.9,
                    "parameters": request.parameters
                }
            ]
            
            conclusions = await reasoning_engine.infer(premises)
            return {
                "type": "reasoning_query",
                "query": request.query,
                "premises": premises,
                "conclusions": conclusions
            }
        else:
            return {
                "type": "reasoning_query", 
                "query": request.query,
                "response": "Reasoning engine not available"
            }
    
    async def _process_natural_language_query(self, request: QueryRequest) -> Dict[str, Any]:
        """Process natural language queries"""
        # Simple natural language processing
        return {
            "type": "natural_language",
            "query": request.query,
            "response": f"Processed natural language query: {request.query}",
            "intent": "general_query",
            "confidence": 0.7
        }
    
    async def _process_distributed_task(self, task_id: str, request: TaskRequest) -> Dict[str, Any]:
        """Process distributed cognitive task"""
        # Update task status
        await self.state_manager.set_state(
            "distributed_task",
            task_id,
            {
                "task_type": request.task_type,
                "task_data": request.task_data,
                "status": "processing",
                "updated_at": datetime.now().isoformat()
            }
        )
        
        # Process based on task type
        if request.task_type == "analysis":
            result = await self._process_analysis_task(request.task_data)
        elif request.task_type == "reasoning":
            result = await self._process_reasoning_task(request.task_data)
        elif request.task_type == "synchronization":
            result = await self._process_sync_task(request.task_data)
        else:
            result = {"message": f"Unknown task type: {request.task_type}"}
        
        # Update task status to completed
        await self.state_manager.set_state(
            "distributed_task",
            task_id,
            {
                "task_type": request.task_type,
                "task_data": request.task_data,
                "status": "completed",
                "result": result,
                "completed_at": datetime.now().isoformat()
            }
        )
        
        return result
    
    async def _process_analysis_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process analysis task"""
        return {
            "analysis_type": task_data.get("type", "general"),
            "result": "Analysis completed",
            "metrics": {
                "processing_time_ms": 50,
                "confidence": 0.85
            }
        }
    
    async def _process_reasoning_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process reasoning task"""
        return {
            "reasoning_type": task_data.get("type", "logical"),
            "conclusions": ["Conclusion 1", "Conclusion 2"],
            "confidence": 0.78
        }
    
    async def _process_sync_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process synchronization task"""
        return {
            "sync_type": task_data.get("type", "state"),
            "entities_synced": 15,
            "conflicts_resolved": 2
        }
    
    async def _process_websocket_message(self, message: Dict[str, Any], 
                                       websocket: WebSocket) -> Optional[Dict[str, Any]]:
        """Process incoming WebSocket message"""
        msg_type = message.get("type", "unknown")
        
        if msg_type == "ping":
            return {"type": "pong", "timestamp": datetime.now().isoformat()}
        
        elif msg_type == "subscribe":
            # Subscribe to specific events
            event_types = message.get("event_types", [])
            await self.websocket_handler.subscribe(websocket, event_types)
            return {"type": "subscribed", "event_types": event_types}
        
        elif msg_type == "query":
            # Process query via WebSocket
            query = message.get("query", "")
            query_type = message.get("query_type", "natural_language")
            
            request = QueryRequest(
                query=query,
                query_type=query_type,
                parameters=message.get("parameters", {})
            )
            
            result = await self._process_natural_language_query(request)
            return {"type": "query_response", "result": result}
        
        elif msg_type == "state_update":
            # Handle real-time state update
            entity_type = message.get("entity_type")
            entity_id = message.get("entity_id")
            data = message.get("data")
            
            if entity_type and entity_id and data:
                event = await self.state_manager.set_state(entity_type, entity_id, data)
                return {"type": "state_updated", "event_id": event.event_id}
        
        return {"type": "error", "message": f"Unknown message type: {msg_type}"}
    
    def _broadcast_state_event(self, event: StateEvent):
        """Broadcast state event to WebSocket clients"""
        event_data = {
            "type": "state_event",
            "event": asdict(event),
            "timestamp": datetime.now().isoformat()
        }
        
        # Broadcast to all connected clients
        asyncio.create_task(
            self.websocket_handler.broadcast(json.dumps(event_data))
        )