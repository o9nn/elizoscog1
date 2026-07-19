"""
Cognitive Load Balancer - Phase 12 Implementation

Implements intelligent load balancing for advisor agent requests:
- Cognitive-aware routing based on agent capabilities
- Workload distribution with minimal latency
- Sub-100ms routing decisions
- Adaptive load balancing strategies
"""

import asyncio
import uuid
import time
import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import heapq


logger = logging.getLogger(__name__)


class LoadBalanceStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"
    COGNITIVE_OPTIMAL = "cognitive_optimal"
    GEOGRAPHIC = "geographic"
    CAPABILITY_MATCH = "capability_match"


@dataclass
class AgentWorkload:
    """Represents the current workload state of an agent"""
    agent_id: str
    
    # Current load metrics
    active_requests: int = 0
    queued_requests: int = 0
    max_concurrent: int = 100
    
    # Performance metrics
    avg_response_time_ms: float = 50.0
    p99_response_time_ms: float = 100.0
    success_rate: float = 1.0
    
    # Resource utilization
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    
    # Cognitive metrics
    cognitive_load: float = 0.0
    reasoning_depth: int = 0
    
    # Timing
    last_updated: float = field(default_factory=time.time)
    
    @property
    def current_load_factor(self) -> float:
        """Calculate current load factor (0.0 - 1.0)"""
        request_load = self.active_requests / self.max_concurrent if self.max_concurrent > 0 else 1.0
        resource_load = max(self.cpu_utilization, self.memory_utilization)
        cognitive_factor = self.cognitive_load
        
        return min(1.0, (request_load * 0.4 + resource_load * 0.3 + cognitive_factor * 0.3))
    
    @property
    def is_available(self) -> bool:
        """Check if agent can accept more requests"""
        return (
            self.active_requests < self.max_concurrent and
            self.current_load_factor < 0.95
        )
    
    @property
    def effective_capacity(self) -> int:
        """Calculate remaining capacity"""
        return max(0, self.max_concurrent - self.active_requests - self.queued_requests)


@dataclass
class RoutingRequest:
    """Represents a request to be routed to an agent"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Request properties
    required_capabilities: List[str] = field(default_factory=list)
    preferred_region: str = ""
    
    # Priority and constraints
    priority: int = 5  # 1-10, higher is more important
    max_latency_ms: float = 100.0
    timeout_seconds: float = 30.0
    
    # Cognitive requirements
    required_reasoning_depth: int = 0
    complexity_score: float = 0.5
    
    # Metadata
    source_agent_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Result of a routing decision"""
    request_id: str
    selected_agent_id: str
    strategy_used: LoadBalanceStrategy
    
    # Decision metrics
    decision_time_ms: float
    expected_latency_ms: float
    confidence_score: float
    
    # Alternatives considered
    candidates_evaluated: int
    alternatives: List[str] = field(default_factory=list)
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)


class CognitiveLoadBalancer:
    """
    Cognitive-aware load balancer for advisor agent network
    
    Features:
    - Sub-100ms routing decisions
    - Multiple load balancing strategies
    - Cognitive workload awareness
    - Adaptive strategy selection
    - Geographic affinity
    - Capability-based routing
    """
    
    def __init__(
        self,
        default_strategy: LoadBalanceStrategy = LoadBalanceStrategy.COGNITIVE_OPTIMAL,
        enable_adaptive_strategy: bool = True
    ):
        self.default_strategy = default_strategy
        self.enable_adaptive_strategy = enable_adaptive_strategy
        
        # Agent workload tracking
        self._agent_workloads: Dict[str, AgentWorkload] = {}
        
        # Agent metadata (capabilities, region, etc.)
        self._agent_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Round-robin counters
        self._rr_counters: Dict[str, int] = {}
        
        # Routing statistics
        self._routing_decisions: int = 0
        self._routing_time_total_ms: float = 0.0
        self._successful_routes: int = 0
        
        # Strategy performance tracking
        self._strategy_performance: Dict[LoadBalanceStrategy, Dict[str, float]] = {
            strategy: {"success_rate": 1.0, "avg_latency": 50.0, "decisions": 0}
            for strategy in LoadBalanceStrategy
        }
        
        logger.info(f"Initialized CognitiveLoadBalancer with strategy: {default_strategy.value}")
    
    def register_agent(
        self,
        agent_id: str,
        capabilities: List[str] = None,
        region: str = "",
        max_concurrent: int = 100,
        metadata: Dict[str, Any] = None
    ):
        """Register an agent with the load balancer"""
        self._agent_workloads[agent_id] = AgentWorkload(
            agent_id=agent_id,
            max_concurrent=max_concurrent
        )
        
        self._agent_metadata[agent_id] = {
            "capabilities": capabilities or [],
            "region": region,
            "metadata": metadata or {}
        }
        
        logger.debug(f"Registered agent {agent_id} with load balancer")
    
    def deregister_agent(self, agent_id: str):
        """Remove an agent from the load balancer"""
        self._agent_workloads.pop(agent_id, None)
        self._agent_metadata.pop(agent_id, None)
        logger.debug(f"Deregistered agent {agent_id} from load balancer")
    
    def update_workload(
        self,
        agent_id: str,
        active_requests: int = None,
        avg_response_time_ms: float = None,
        cpu_utilization: float = None,
        memory_utilization: float = None,
        cognitive_load: float = None
    ):
        """Update workload metrics for an agent"""
        workload = self._agent_workloads.get(agent_id)
        if not workload:
            return
        
        if active_requests is not None:
            workload.active_requests = active_requests
        if avg_response_time_ms is not None:
            workload.avg_response_time_ms = avg_response_time_ms
        if cpu_utilization is not None:
            workload.cpu_utilization = cpu_utilization
        if memory_utilization is not None:
            workload.memory_utilization = memory_utilization
        if cognitive_load is not None:
            workload.cognitive_load = cognitive_load
        
        workload.last_updated = time.time()
    
    async def route_request(
        self,
        request: RoutingRequest,
        strategy: LoadBalanceStrategy = None
    ) -> Optional[RoutingDecision]:
        """
        Route a request to the best available agent
        
        Returns RoutingDecision or None if no agent available
        """
        start_time = time.time()
        
        # Select strategy
        active_strategy = strategy or self.default_strategy
        
        if self.enable_adaptive_strategy and strategy is None:
            active_strategy = self._select_adaptive_strategy(request)
        
        # Get candidate agents
        candidates = self._get_candidate_agents(request)
        
        if not candidates:
            logger.warning(f"No candidate agents for request {request.request_id[:8]}")
            return None
        
        # Apply routing strategy
        selected_agent = None
        
        if active_strategy == LoadBalanceStrategy.ROUND_ROBIN:
            selected_agent = self._route_round_robin(candidates)
        elif active_strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            selected_agent = self._route_weighted_round_robin(candidates)
        elif active_strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            selected_agent = self._route_least_connections(candidates)
        elif active_strategy == LoadBalanceStrategy.LEAST_RESPONSE_TIME:
            selected_agent = self._route_least_response_time(candidates)
        elif active_strategy == LoadBalanceStrategy.RANDOM:
            selected_agent = self._route_random(candidates)
        elif active_strategy == LoadBalanceStrategy.COGNITIVE_OPTIMAL:
            selected_agent = self._route_cognitive_optimal(candidates, request)
        elif active_strategy == LoadBalanceStrategy.GEOGRAPHIC:
            selected_agent = self._route_geographic(candidates, request)
        elif active_strategy == LoadBalanceStrategy.CAPABILITY_MATCH:
            selected_agent = self._route_capability_match(candidates, request)
        else:
            selected_agent = self._route_round_robin(candidates)
        
        if not selected_agent:
            return None
        
        # Calculate decision time
        decision_time_ms = (time.time() - start_time) * 1000
        
        # Update statistics
        self._routing_decisions += 1
        self._routing_time_total_ms += decision_time_ms
        
        # Create routing decision
        workload = self._agent_workloads.get(selected_agent)
        expected_latency = workload.avg_response_time_ms if workload else 50.0
        
        decision = RoutingDecision(
            request_id=request.request_id,
            selected_agent_id=selected_agent,
            strategy_used=active_strategy,
            decision_time_ms=decision_time_ms,
            expected_latency_ms=expected_latency,
            confidence_score=self._calculate_confidence(selected_agent, request),
            candidates_evaluated=len(candidates),
            alternatives=[c for c in candidates if c != selected_agent][:3]
        )
        
        # Update agent workload
        if workload:
            workload.active_requests += 1
        
        logger.debug(
            f"Routed request {request.request_id[:8]} to {selected_agent} "
            f"using {active_strategy.value} in {decision_time_ms:.2f}ms"
        )
        
        return decision
    
    def complete_request(
        self,
        agent_id: str,
        success: bool,
        response_time_ms: float
    ):
        """Record completion of a request"""
        workload = self._agent_workloads.get(agent_id)
        if workload:
            workload.active_requests = max(0, workload.active_requests - 1)
            
            # Update average response time (exponential moving average)
            alpha = 0.1
            workload.avg_response_time_ms = (
                alpha * response_time_ms +
                (1 - alpha) * workload.avg_response_time_ms
            )
            
            # Update success rate
            if success:
                self._successful_routes += 1
            
            workload.last_updated = time.time()
    
    def get_agent_workloads(self) -> Dict[str, Dict[str, Any]]:
        """Get current workload status for all agents"""
        return {
            agent_id: {
                "active_requests": w.active_requests,
                "load_factor": w.current_load_factor,
                "avg_response_time_ms": w.avg_response_time_ms,
                "is_available": w.is_available,
                "effective_capacity": w.effective_capacity
            }
            for agent_id, w in self._agent_workloads.items()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        avg_routing_time = (
            self._routing_time_total_ms / self._routing_decisions
            if self._routing_decisions > 0 else 0.0
        )
        
        total_active = sum(w.active_requests for w in self._agent_workloads.values())
        total_capacity = sum(w.max_concurrent for w in self._agent_workloads.values())
        
        return {
            "registered_agents": len(self._agent_workloads),
            "total_active_requests": total_active,
            "total_capacity": total_capacity,
            "utilization": total_active / total_capacity if total_capacity > 0 else 0.0,
            "routing_decisions": self._routing_decisions,
            "avg_routing_time_ms": avg_routing_time,
            "successful_routes": self._successful_routes,
            "default_strategy": self.default_strategy.value
        }
    
    # Private routing methods
    
    def _get_candidate_agents(self, request: RoutingRequest) -> List[str]:
        """Get list of candidate agents for a request"""
        candidates = []
        
        for agent_id, workload in self._agent_workloads.items():
            if not workload.is_available:
                continue
            
            # Check capability requirements
            if request.required_capabilities:
                agent_caps = self._agent_metadata.get(agent_id, {}).get("capabilities", [])
                if not all(cap in agent_caps for cap in request.required_capabilities):
                    continue
            
            # Check cognitive requirements
            if request.required_reasoning_depth > 0:
                if workload.reasoning_depth < request.required_reasoning_depth:
                    continue
            
            candidates.append(agent_id)
        
        return candidates
    
    def _route_round_robin(self, candidates: List[str]) -> Optional[str]:
        """Round-robin routing"""
        if not candidates:
            return None
        
        key = "global"
        counter = self._rr_counters.get(key, 0)
        selected = candidates[counter % len(candidates)]
        self._rr_counters[key] = counter + 1
        
        return selected
    
    def _route_weighted_round_robin(self, candidates: List[str]) -> Optional[str]:
        """Weighted round-robin based on capacity"""
        if not candidates:
            return None
        
        # Calculate weights based on available capacity
        weights = []
        for agent_id in candidates:
            workload = self._agent_workloads.get(agent_id)
            if workload:
                weight = workload.effective_capacity
            else:
                weight = 1
            weights.append(weight)
        
        # Weighted selection
        total_weight = sum(weights)
        if total_weight == 0:
            return candidates[0]
        
        r = random.uniform(0, total_weight)
        cumsum = 0
        for i, weight in enumerate(weights):
            cumsum += weight
            if r <= cumsum:
                return candidates[i]
        
        return candidates[-1]
    
    def _route_least_connections(self, candidates: List[str]) -> Optional[str]:
        """Route to agent with fewest active connections"""
        if not candidates:
            return None
        
        min_connections = float('inf')
        selected = None
        
        for agent_id in candidates:
            workload = self._agent_workloads.get(agent_id)
            if workload and workload.active_requests < min_connections:
                min_connections = workload.active_requests
                selected = agent_id
        
        return selected or candidates[0]
    
    def _route_least_response_time(self, candidates: List[str]) -> Optional[str]:
        """Route to agent with lowest average response time"""
        if not candidates:
            return None
        
        min_response_time = float('inf')
        selected = None
        
        for agent_id in candidates:
            workload = self._agent_workloads.get(agent_id)
            if workload and workload.avg_response_time_ms < min_response_time:
                min_response_time = workload.avg_response_time_ms
                selected = agent_id
        
        return selected or candidates[0]
    
    def _route_random(self, candidates: List[str]) -> Optional[str]:
        """Random routing"""
        if not candidates:
            return None
        return random.choice(candidates)
    
    def _route_cognitive_optimal(
        self,
        candidates: List[str],
        request: RoutingRequest
    ) -> Optional[str]:
        """
        Cognitive-optimal routing considering:
        - Current cognitive load
        - Request complexity
        - Response time
        - Success rate
        """
        if not candidates:
            return None
        
        scores = []
        
        for agent_id in candidates:
            workload = self._agent_workloads.get(agent_id)
            if not workload:
                scores.append((agent_id, 0.5))
                continue
            
            # Calculate cognitive fitness score
            load_score = 1.0 - workload.current_load_factor
            response_score = 1.0 - min(1.0, workload.avg_response_time_ms / 200.0)
            success_score = workload.success_rate
            cognitive_fit = 1.0 - abs(workload.cognitive_load - request.complexity_score)
            
            # Weighted combination
            total_score = (
                load_score * 0.3 +
                response_score * 0.2 +
                success_score * 0.3 +
                cognitive_fit * 0.2
            )
            
            scores.append((agent_id, total_score))
        
        # Select best scoring agent
        scores.sort(key=lambda x: -x[1])
        return scores[0][0]
    
    def _route_geographic(
        self,
        candidates: List[str],
        request: RoutingRequest
    ) -> Optional[str]:
        """Route based on geographic proximity"""
        if not candidates:
            return None
        
        if not request.preferred_region:
            return self._route_round_robin(candidates)
        
        # Prefer agents in the same region
        same_region = []
        different_region = []
        
        for agent_id in candidates:
            metadata = self._agent_metadata.get(agent_id, {})
            if metadata.get("region") == request.preferred_region:
                same_region.append(agent_id)
            else:
                different_region.append(agent_id)
        
        if same_region:
            return self._route_least_connections(same_region)
        else:
            return self._route_least_connections(different_region)
    
    def _route_capability_match(
        self,
        candidates: List[str],
        request: RoutingRequest
    ) -> Optional[str]:
        """Route based on capability match score"""
        if not candidates:
            return None
        
        scores = []
        
        for agent_id in candidates:
            metadata = self._agent_metadata.get(agent_id, {})
            agent_caps = set(metadata.get("capabilities", []))
            required_caps = set(request.required_capabilities)
            
            if not required_caps:
                score = 0.5
            else:
                match_count = len(agent_caps & required_caps)
                score = match_count / len(required_caps)
            
            # Also consider workload
            workload = self._agent_workloads.get(agent_id)
            if workload:
                score *= (1.0 - workload.current_load_factor * 0.5)
            
            scores.append((agent_id, score))
        
        scores.sort(key=lambda x: -x[1])
        return scores[0][0]
    
    def _select_adaptive_strategy(self, request: RoutingRequest) -> LoadBalanceStrategy:
        """Select the best strategy based on request characteristics and history"""
        # High priority requests use cognitive optimal
        if request.priority >= 8:
            return LoadBalanceStrategy.COGNITIVE_OPTIMAL
        
        # Complex requests need capability matching
        if request.required_capabilities and len(request.required_capabilities) > 2:
            return LoadBalanceStrategy.CAPABILITY_MATCH
        
        # Geographic affinity for latency-sensitive requests
        if request.max_latency_ms < 50 and request.preferred_region:
            return LoadBalanceStrategy.GEOGRAPHIC
        
        # Default to least connections for balanced load
        return LoadBalanceStrategy.LEAST_CONNECTIONS
    
    def _calculate_confidence(self, agent_id: str, request: RoutingRequest) -> float:
        """Calculate confidence score for a routing decision"""
        workload = self._agent_workloads.get(agent_id)
        if not workload:
            return 0.5
        
        confidence = 0.7
        
        # Higher confidence for lower load
        confidence += (1.0 - workload.current_load_factor) * 0.15
        
        # Higher confidence for good success rate
        confidence += (workload.success_rate - 0.5) * 0.1
        
        # Lower confidence if response time might exceed max latency
        if workload.avg_response_time_ms > request.max_latency_ms * 0.8:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
