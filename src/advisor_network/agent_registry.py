"""
Distributed Agent Registry Service - Phase 12 Implementation

Implements a distributed registry service for AI advisor agents with:
- Agent registration and discovery
- Federation protocols for cross-cluster communication
- Health monitoring and heartbeat management
- Sub-100ms discovery performance
"""

import asyncio
import uuid
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of an advisor agent"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class AgentCapability(Enum):
    """Capabilities that an advisor agent can provide"""
    FINANCIAL_ANALYSIS = "financial_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    MARKET_PREDICTION = "market_prediction"
    COGNITIVE_REASONING = "cognitive_reasoning"
    PATTERN_RECOGNITION = "pattern_recognition"
    NATURAL_LANGUAGE = "natural_language"
    HYPERGRAPH_ANALYSIS = "hypergraph_analysis"
    GGML_INFERENCE = "ggml_inference"
    COLLECTIVE_DECISION = "collective_decision"


class FederationProtocol(Enum):
    """Protocols for cross-cluster agent federation"""
    DIRECT = "direct"  # Direct peer-to-peer communication
    GOSSIP = "gossip"  # Gossip-based propagation
    CONSENSUS = "consensus"  # Consensus-based updates
    HIERARCHICAL = "hierarchical"  # Hierarchical routing


@dataclass
class AdvisorAgent:
    """
    Represents a distributed AI advisor agent
    """
    agent_id: str
    name: str
    host: str
    port: int
    cluster_id: str
    cloud_provider: str
    region: str
    
    # Capabilities
    capabilities: List[AgentCapability] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    
    # Status and health
    status: AgentStatus = AgentStatus.INITIALIZING
    health_score: float = 1.0
    last_heartbeat: float = field(default_factory=time.time)
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    requests_processed: int = 0
    success_rate: float = 1.0
    
    # Configuration
    max_concurrent_requests: int = 100
    current_load: float = 0.0
    
    # Metadata
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    @property
    def endpoint(self) -> str:
        """Return the agent's API endpoint"""
        return f"https://{self.host}:{self.port}"
    
    @property
    def is_healthy(self) -> bool:
        """Check if agent is considered healthy"""
        return (
            self.status in [AgentStatus.ACTIVE, AgentStatus.IDLE, AgentStatus.BUSY] and
            self.health_score >= 0.5 and
            time.time() - self.last_heartbeat < 30
        )
    
    @property
    def is_available(self) -> bool:
        """Check if agent is available to accept requests"""
        return (
            self.is_healthy and
            self.status != AgentStatus.BUSY and
            self.current_load < 0.9
        )
    
    def update_heartbeat(self):
        """Update the last heartbeat timestamp"""
        self.last_heartbeat = time.time()
    
    def update_metrics(self, response_time_ms: float, success: bool):
        """Update performance metrics after request completion"""
        self.requests_processed += 1
        
        # Exponential moving average for response time
        alpha = 0.1
        self.avg_response_time_ms = (
            alpha * response_time_ms + 
            (1 - alpha) * self.avg_response_time_ms
        )
        
        # Update success rate
        self.success_rate = (
            (self.success_rate * (self.requests_processed - 1) + (1 if success else 0)) 
            / self.requests_processed
        )


@dataclass
class AgentRegistryConfig:
    """Configuration for the distributed agent registry"""
    cluster_id: str = "default"
    federation_protocol: FederationProtocol = FederationProtocol.GOSSIP
    
    # Timing configuration
    heartbeat_interval_seconds: int = 5
    agent_timeout_seconds: int = 30
    cleanup_interval_seconds: int = 60
    
    # Federation configuration
    federation_sync_interval_seconds: int = 30
    max_federation_peers: int = 10
    
    # Performance configuration
    max_agents_per_cluster: int = 10000
    discovery_cache_ttl_seconds: int = 10
    
    # Security
    require_tls: bool = True
    verify_agent_signatures: bool = True


class DistributedAgentRegistry:
    """
    Distributed registry for AI advisor agents with federation support
    
    Features:
    - Sub-100ms agent discovery
    - Federation across multiple clusters
    - Automatic health monitoring
    - Load-aware routing
    - Capability-based discovery
    """
    
    def __init__(self, config: AgentRegistryConfig = None):
        self.config = config or AgentRegistryConfig()
        self.cluster_id = self.config.cluster_id
        
        # Local agent storage
        self._agents: Dict[str, AdvisorAgent] = {}
        self._agents_by_capability: Dict[AgentCapability, Set[str]] = {
            cap: set() for cap in AgentCapability
        }
        self._agents_by_cluster: Dict[str, Set[str]] = {}
        self._agents_by_region: Dict[str, Set[str]] = {}
        
        # Federation peers
        self._federation_peers: Dict[str, Dict[str, Any]] = {}
        self._federated_agents: Dict[str, AdvisorAgent] = {}
        
        # Discovery cache
        self._discovery_cache: Dict[str, tuple] = {}
        
        # Event subscribers
        self._subscribers: List[Callable[[str, AdvisorAgent], None]] = []
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._federation_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Thread pool for CPU-bound operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized DistributedAgentRegistry for cluster {self.cluster_id}")
    
    async def start(self):
        """Start the registry and background tasks"""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._federation_task = asyncio.create_task(self._federation_sync_loop())
        
        logger.info(f"Agent registry started for cluster {self.cluster_id}")
    
    async def stop(self):
        """Stop the registry and cleanup"""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._federation_task:
            self._federation_task.cancel()
            try:
                await self._federation_task
            except asyncio.CancelledError:
                pass
        
        self._executor.shutdown(wait=False)
        logger.info(f"Agent registry stopped for cluster {self.cluster_id}")
    
    async def register_agent(self, agent: AdvisorAgent) -> bool:
        """
        Register an advisor agent in the registry
        
        Returns True if registration successful
        """
        start_time = time.time()
        
        try:
            # Validate agent
            if not self._validate_agent(agent):
                logger.warning(f"Agent validation failed for {agent.agent_id}")
                return False
            
            # Check capacity
            if len(self._agents) >= self.config.max_agents_per_cluster:
                logger.warning(f"Cluster {self.cluster_id} at max capacity")
                return False
            
            # Update heartbeat and status
            agent.update_heartbeat()
            agent.status = AgentStatus.ACTIVE
            
            # Store agent
            self._agents[agent.agent_id] = agent
            
            # Index by capability
            for capability in agent.capabilities:
                self._agents_by_capability[capability].add(agent.agent_id)
            
            # Index by cluster
            if agent.cluster_id not in self._agents_by_cluster:
                self._agents_by_cluster[agent.cluster_id] = set()
            self._agents_by_cluster[agent.cluster_id].add(agent.agent_id)
            
            # Index by region
            if agent.region not in self._agents_by_region:
                self._agents_by_region[agent.region] = set()
            self._agents_by_region[agent.region].add(agent.agent_id)
            
            # Clear relevant cache entries
            self._invalidate_cache()
            
            # Notify subscribers
            await self._notify_subscribers("registered", agent)
            
            registration_time = (time.time() - start_time) * 1000
            logger.info(
                f"Registered agent {agent.name} ({agent.agent_id}) in {registration_time:.2f}ms"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """
        Deregister an agent from the registry
        
        Returns True if deregistration successful
        """
        try:
            agent = self._agents.get(agent_id)
            if not agent:
                return False
            
            # Remove from main storage
            del self._agents[agent_id]
            
            # Remove from capability index
            for capability in agent.capabilities:
                self._agents_by_capability[capability].discard(agent_id)
            
            # Remove from cluster index
            if agent.cluster_id in self._agents_by_cluster:
                self._agents_by_cluster[agent.cluster_id].discard(agent_id)
            
            # Remove from region index
            if agent.region in self._agents_by_region:
                self._agents_by_region[agent.region].discard(agent_id)
            
            # Clear cache
            self._invalidate_cache()
            
            # Notify subscribers
            await self._notify_subscribers("deregistered", agent)
            
            logger.info(f"Deregistered agent {agent.name} ({agent_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister agent {agent_id}: {e}")
            return False
    
    async def discover_agents(
        self,
        capabilities: List[AgentCapability] = None,
        region: str = None,
        cluster_id: str = None,
        healthy_only: bool = True,
        available_only: bool = False,
        include_federated: bool = True,
        limit: int = 100
    ) -> List[AdvisorAgent]:
        """
        Discover agents matching the specified criteria
        
        Designed for sub-100ms discovery performance
        """
        start_time = time.time()
        
        # Build cache key
        cache_key = self._build_cache_key(
            capabilities, region, cluster_id, healthy_only, available_only
        )
        
        # Check cache
        if cache_key in self._discovery_cache:
            cache_time, cached_agents = self._discovery_cache[cache_key]
            if time.time() - cache_time < self.config.discovery_cache_ttl_seconds:
                discovery_time = (time.time() - start_time) * 1000
                logger.debug(f"Cache hit for discovery in {discovery_time:.2f}ms")
                return cached_agents[:limit]
        
        # Build candidate set
        candidates = set(self._agents.keys())
        
        # Filter by capability
        if capabilities:
            capability_matches = set()
            for cap in capabilities:
                capability_matches.update(self._agents_by_capability.get(cap, set()))
            candidates &= capability_matches
        
        # Filter by region
        if region:
            region_agents = self._agents_by_region.get(region, set())
            candidates &= region_agents
        
        # Filter by cluster
        if cluster_id:
            cluster_agents = self._agents_by_cluster.get(cluster_id, set())
            candidates &= cluster_agents
        
        # Get agent objects and apply additional filters
        results = []
        for agent_id in candidates:
            agent = self._agents.get(agent_id)
            if not agent:
                continue
            
            if healthy_only and not agent.is_healthy:
                continue
            
            if available_only and not agent.is_available:
                continue
            
            results.append(agent)
        
        # Include federated agents if requested
        if include_federated:
            for agent in self._federated_agents.values():
                if capabilities:
                    if not any(cap in agent.capabilities for cap in capabilities):
                        continue
                
                if region and agent.region != region:
                    continue
                
                if healthy_only and not agent.is_healthy:
                    continue
                
                if available_only and not agent.is_available:
                    continue
                
                results.append(agent)
        
        # Sort by health score and availability
        results.sort(key=lambda a: (-a.health_score, a.current_load))
        
        # Cache results
        self._discovery_cache[cache_key] = (time.time(), results)
        
        discovery_time = (time.time() - start_time) * 1000
        logger.debug(
            f"Discovered {len(results)} agents in {discovery_time:.2f}ms"
        )
        
        return results[:limit]
    
    async def get_agent(self, agent_id: str) -> Optional[AdvisorAgent]:
        """Get a specific agent by ID"""
        return self._agents.get(agent_id) or self._federated_agents.get(agent_id)
    
    async def update_agent_status(
        self,
        agent_id: str,
        status: AgentStatus,
        health_score: float = None,
        current_load: float = None
    ) -> bool:
        """Update an agent's status and metrics"""
        agent = self._agents.get(agent_id)
        if not agent:
            return False
        
        agent.status = status
        agent.update_heartbeat()
        
        if health_score is not None:
            agent.health_score = max(0.0, min(1.0, health_score))
        
        if current_load is not None:
            agent.current_load = max(0.0, min(1.0, current_load))
        
        return True
    
    async def heartbeat(self, agent_id: str) -> bool:
        """Process heartbeat from an agent"""
        agent = self._agents.get(agent_id)
        if agent:
            agent.update_heartbeat()
            return True
        return False
    
    # Federation methods
    
    async def add_federation_peer(
        self,
        peer_id: str,
        host: str,
        port: int,
        protocol: FederationProtocol = None
    ) -> bool:
        """Add a federation peer for cross-cluster discovery"""
        if len(self._federation_peers) >= self.config.max_federation_peers:
            logger.warning("Maximum federation peers reached")
            return False
        
        self._federation_peers[peer_id] = {
            "host": host,
            "port": port,
            "protocol": protocol or self.config.federation_protocol,
            "last_sync": 0,
            "healthy": True
        }
        
        logger.info(f"Added federation peer {peer_id} at {host}:{port}")
        return True
    
    async def remove_federation_peer(self, peer_id: str) -> bool:
        """Remove a federation peer"""
        if peer_id in self._federation_peers:
            del self._federation_peers[peer_id]
            
            # Remove federated agents from this peer
            to_remove = [
                agent_id for agent_id, agent in self._federated_agents.items()
                if agent.cluster_id == peer_id
            ]
            for agent_id in to_remove:
                del self._federated_agents[agent_id]
            
            logger.info(f"Removed federation peer {peer_id}")
            return True
        
        return False
    
    async def sync_with_peer(self, peer_id: str) -> bool:
        """Synchronize agent registry with a federation peer"""
        peer = self._federation_peers.get(peer_id)
        if not peer:
            return False
        
        try:
            # In a real implementation, this would make an HTTP/gRPC call
            # to the peer registry to get their agent list
            # For now, we simulate a successful sync
            
            peer["last_sync"] = time.time()
            peer["healthy"] = True
            
            logger.debug(f"Synced with federation peer {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync with peer {peer_id}: {e}")
            peer["healthy"] = False
            return False
    
    # Statistics methods
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_agents = len(self._agents)
        healthy_agents = sum(1 for a in self._agents.values() if a.is_healthy)
        available_agents = sum(1 for a in self._agents.values() if a.is_available)
        
        capability_counts = {
            cap.value: len(agents) 
            for cap, agents in self._agents_by_capability.items()
        }
        
        region_counts = {
            region: len(agents) 
            for region, agents in self._agents_by_region.items()
        }
        
        return {
            "cluster_id": self.cluster_id,
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "available_agents": available_agents,
            "federated_agents": len(self._federated_agents),
            "federation_peers": len(self._federation_peers),
            "capabilities": capability_counts,
            "regions": region_counts,
            "cache_entries": len(self._discovery_cache),
        }
    
    # Subscription methods
    
    def subscribe(self, callback: Callable[[str, AdvisorAgent], None]):
        """Subscribe to agent registration/deregistration events"""
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[str, AdvisorAgent], None]):
        """Unsubscribe from agent events"""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    # Private methods
    
    def _validate_agent(self, agent: AdvisorAgent) -> bool:
        """Validate agent configuration"""
        if not agent.agent_id or not agent.name:
            return False
        
        if not agent.host or not agent.port:
            return False
        
        if agent.port < 1 or agent.port > 65535:
            return False
        
        return True
    
    def _build_cache_key(
        self,
        capabilities: List[AgentCapability] = None,
        region: str = None,
        cluster_id: str = None,
        healthy_only: bool = True,
        available_only: bool = False
    ) -> str:
        """Build a cache key for discovery results"""
        parts = [
            ",".join(sorted([c.value for c in (capabilities or [])])),
            region or "",
            cluster_id or "",
            str(healthy_only),
            str(available_only)
        ]
        return hashlib.md5("|".join(parts).encode()).hexdigest()
    
    def _invalidate_cache(self):
        """Invalidate all discovery cache entries"""
        self._discovery_cache.clear()
    
    async def _notify_subscribers(self, event_type: str, agent: AdvisorAgent):
        """Notify all subscribers of an event"""
        for callback in self._subscribers:
            try:
                callback(event_type, agent)
            except Exception as e:
                logger.error(f"Error in subscriber callback: {e}")
    
    async def _cleanup_loop(self):
        """Background task to cleanup unhealthy agents"""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                await self._cleanup_unhealthy_agents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_unhealthy_agents(self):
        """Remove agents that have timed out"""
        current_time = time.time()
        to_remove = []
        
        for agent_id, agent in self._agents.items():
            if current_time - agent.last_heartbeat > self.config.agent_timeout_seconds:
                to_remove.append(agent_id)
        
        for agent_id in to_remove:
            await self.deregister_agent(agent_id)
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} unhealthy agents")
    
    async def _federation_sync_loop(self):
        """Background task to sync with federation peers"""
        while self._running:
            try:
                await asyncio.sleep(self.config.federation_sync_interval_seconds)
                
                for peer_id in list(self._federation_peers.keys()):
                    await self.sync_with_peer(peer_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in federation sync loop: {e}")
