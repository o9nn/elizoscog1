"""
Hypergraph Agent Relationships - Phase 12 Implementation

Implements hypergraph-based modeling for agent relationships:
- Agent relationship graph with hyperedge connections
- Cognitive synergy modeling between agents
- Network topology optimization
- Pattern-based agent clustering
"""

import asyncio
import uuid
import time
import math
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np


logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Types of relationships between agents"""
    COLLABORATION = "collaboration"
    DELEGATION = "delegation"
    SPECIALIZATION = "specialization"
    SUPERVISION = "supervision"
    PEER = "peer"
    DEPENDENCY = "dependency"
    SIMILARITY = "similarity"
    COMPLEMENTARY = "complementary"


class NodeType(Enum):
    """Types of nodes in the hypergraph"""
    AGENT = "agent"
    CAPABILITY = "capability"
    TASK = "task"
    CLUSTER = "cluster"
    CONCEPT = "concept"


@dataclass
class AgentNode:
    """
    Node representing an agent or concept in the hypergraph
    """
    node_id: str
    node_type: NodeType = NodeType.AGENT
    
    # Agent properties (if agent type)
    agent_id: str = ""
    capabilities: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    
    # Graph metrics
    centrality_score: float = 0.0
    clustering_coefficient: float = 0.0
    
    # Embedding (for similarity computations)
    embedding: Optional[np.ndarray] = None
    
    # Connection tracking
    connected_edges: Set[str] = field(default_factory=set)
    
    # Metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class RelationshipEdge:
    """
    Hyperedge connecting multiple agents
    """
    edge_id: str
    edge_type: RelationshipType
    
    # Connected nodes (hyperedge can connect multiple nodes)
    connected_nodes: List[str] = field(default_factory=list)
    
    # Edge properties
    weight: float = 1.0
    strength: float = 1.0
    
    # Directional (if applicable)
    source_node: str = ""
    target_nodes: List[str] = field(default_factory=list)
    
    # Metrics
    interaction_count: int = 0
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0
    
    # Metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_interaction: float = field(default_factory=time.time)


class AgentRelationshipGraph:
    """
    Hypergraph-based agent relationship modeling
    
    Features:
    - Multi-way relationships between agents
    - Cognitive synergy analysis
    - Network topology optimization
    - Intelligent agent clustering
    - Relationship strength evolution
    """
    
    def __init__(self):
        # Graph storage
        self._nodes: Dict[str, AgentNode] = {}
        self._edges: Dict[str, RelationshipEdge] = {}
        
        # Index structures
        self._nodes_by_type: Dict[NodeType, Set[str]] = {
            nt: set() for nt in NodeType
        }
        self._edges_by_type: Dict[RelationshipType, Set[str]] = {
            rt: set() for rt in RelationshipType
        }
        
        # Clusters
        self._clusters: Dict[str, Set[str]] = {}
        
        # Graph metrics cache
        self._metrics_cache: Dict[str, Any] = {}
        self._metrics_cache_valid: bool = False
        
        logger.info("Initialized AgentRelationshipGraph")
    
    def add_agent_node(
        self,
        agent_id: str,
        capabilities: List[str] = None,
        specializations: List[str] = None,
        embedding: np.ndarray = None,
        attributes: Dict[str, Any] = None
    ) -> AgentNode:
        """Add an agent node to the hypergraph"""
        node_id = f"agent_{agent_id}"
        
        node = AgentNode(
            node_id=node_id,
            node_type=NodeType.AGENT,
            agent_id=agent_id,
            capabilities=capabilities or [],
            specializations=specializations or [],
            embedding=embedding,
            attributes=attributes or {}
        )
        
        self._nodes[node_id] = node
        self._nodes_by_type[NodeType.AGENT].add(node_id)
        
        # Invalidate metrics cache
        self._metrics_cache_valid = False
        
        logger.debug(f"Added agent node {node_id}")
        return node
    
    def add_capability_node(
        self,
        capability_name: str,
        attributes: Dict[str, Any] = None
    ) -> AgentNode:
        """Add a capability concept node"""
        node_id = f"capability_{capability_name}"
        
        node = AgentNode(
            node_id=node_id,
            node_type=NodeType.CAPABILITY,
            attributes=attributes or {"name": capability_name}
        )
        
        self._nodes[node_id] = node
        self._nodes_by_type[NodeType.CAPABILITY].add(node_id)
        
        self._metrics_cache_valid = False
        
        logger.debug(f"Added capability node {node_id}")
        return node
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its connected edges"""
        node = self._nodes.get(node_id)
        if not node:
            return False
        
        # Remove connected edges
        for edge_id in list(node.connected_edges):
            self.remove_edge(edge_id)
        
        # Remove from type index
        self._nodes_by_type[node.node_type].discard(node_id)
        
        # Remove node
        del self._nodes[node_id]
        
        self._metrics_cache_valid = False
        logger.debug(f"Removed node {node_id}")
        return True
    
    def add_relationship(
        self,
        source_agent_id: str,
        target_agent_ids: List[str],
        relationship_type: RelationshipType,
        weight: float = 1.0,
        attributes: Dict[str, Any] = None
    ) -> RelationshipEdge:
        """Add a relationship (hyperedge) between agents"""
        # Generate edge ID
        edge_id = f"rel_{relationship_type.value}_{uuid.uuid4().hex[:8]}"
        
        # Convert agent IDs to node IDs
        source_node = f"agent_{source_agent_id}"
        target_nodes = [f"agent_{aid}" for aid in target_agent_ids]
        all_nodes = [source_node] + target_nodes
        
        # Verify all nodes exist
        for node_id in all_nodes:
            if node_id not in self._nodes:
                logger.warning(f"Node {node_id} does not exist")
                return None
        
        # Create edge
        edge = RelationshipEdge(
            edge_id=edge_id,
            edge_type=relationship_type,
            connected_nodes=all_nodes,
            source_node=source_node,
            target_nodes=target_nodes,
            weight=weight,
            attributes=attributes or {}
        )
        
        # Store edge
        self._edges[edge_id] = edge
        self._edges_by_type[relationship_type].add(edge_id)
        
        # Update node connections
        for node_id in all_nodes:
            self._nodes[node_id].connected_edges.add(edge_id)
        
        self._metrics_cache_valid = False
        
        logger.debug(f"Added relationship {edge_id} ({relationship_type.value})")
        return edge
    
    def remove_edge(self, edge_id: str) -> bool:
        """Remove a relationship edge"""
        edge = self._edges.get(edge_id)
        if not edge:
            return False
        
        # Remove from connected nodes
        for node_id in edge.connected_nodes:
            if node_id in self._nodes:
                self._nodes[node_id].connected_edges.discard(edge_id)
        
        # Remove from type index
        self._edges_by_type[edge.edge_type].discard(edge_id)
        
        # Remove edge
        del self._edges[edge_id]
        
        self._metrics_cache_valid = False
        return True
    
    def update_relationship_metrics(
        self,
        edge_id: str,
        interaction_success: bool,
        latency_ms: float
    ):
        """Update relationship metrics after an interaction"""
        edge = self._edges.get(edge_id)
        if not edge:
            return
        
        edge.interaction_count += 1
        edge.last_interaction = time.time()
        
        # Update success rate (exponential moving average)
        alpha = 0.1
        edge.success_rate = alpha * (1.0 if interaction_success else 0.0) + (1 - alpha) * edge.success_rate
        
        # Update latency
        edge.avg_latency_ms = alpha * latency_ms + (1 - alpha) * edge.avg_latency_ms
        
        # Adjust strength based on success
        if interaction_success:
            edge.strength = min(2.0, edge.strength * 1.01)
        else:
            edge.strength = max(0.1, edge.strength * 0.99)
    
    def find_related_agents(
        self,
        agent_id: str,
        relationship_types: List[RelationshipType] = None,
        max_hops: int = 2,
        min_strength: float = 0.0
    ) -> List[Tuple[str, float, int]]:
        """
        Find agents related to the given agent
        
        Returns list of (agent_id, relationship_strength, hops)
        """
        source_node = f"agent_{agent_id}"
        if source_node not in self._nodes:
            return []
        
        visited = {source_node}
        results = []
        queue = [(source_node, 1.0, 0)]  # (node, strength, hops)
        
        while queue:
            current_node, current_strength, hops = queue.pop(0)
            
            if hops >= max_hops:
                continue
            
            node = self._nodes.get(current_node)
            if not node:
                continue
            
            for edge_id in node.connected_edges:
                edge = self._edges.get(edge_id)
                if not edge:
                    continue
                
                # Filter by relationship type
                if relationship_types and edge.edge_type not in relationship_types:
                    continue
                
                # Filter by strength
                combined_strength = current_strength * edge.strength
                if combined_strength < min_strength:
                    continue
                
                # Find connected agents
                for connected_node in edge.connected_nodes:
                    if connected_node in visited:
                        continue
                    
                    visited.add(connected_node)
                    
                    if connected_node.startswith("agent_"):
                        related_agent_id = connected_node[6:]  # Remove "agent_" prefix
                        results.append((related_agent_id, combined_strength, hops + 1))
                    
                    queue.append((connected_node, combined_strength, hops + 1))
        
        # Sort by strength
        results.sort(key=lambda x: -x[1])
        return results
    
    def find_optimal_collaborators(
        self,
        agent_id: str,
        required_capabilities: List[str],
        max_collaborators: int = 5
    ) -> List[str]:
        """
        Find optimal collaborators for a task based on relationship graph
        """
        source_node = f"agent_{agent_id}"
        if source_node not in self._nodes:
            return []
        
        # Find all related agents
        related = self.find_related_agents(
            agent_id,
            relationship_types=[
                RelationshipType.COLLABORATION,
                RelationshipType.COMPLEMENTARY,
                RelationshipType.PEER
            ],
            max_hops=3
        )
        
        # Score collaborators based on capabilities and relationship
        scored_agents = []
        
        for related_agent_id, strength, hops in related:
            node = self._nodes.get(f"agent_{related_agent_id}")
            if not node:
                continue
            
            # Calculate capability match
            agent_caps = set(node.capabilities)
            required = set(required_capabilities)
            
            if not required:
                cap_score = 0.5
            else:
                cap_score = len(agent_caps & required) / len(required)
            
            # Combined score
            total_score = (strength * 0.4 + cap_score * 0.4 + (1.0 / hops) * 0.2)
            scored_agents.append((related_agent_id, total_score))
        
        # Sort and return top collaborators
        scored_agents.sort(key=lambda x: -x[1])
        return [agent_id for agent_id, score in scored_agents[:max_collaborators]]
    
    def calculate_cognitive_synergy(
        self,
        agent_ids: List[str]
    ) -> float:
        """
        Calculate cognitive synergy score for a group of agents
        
        Higher scores indicate better collaboration potential
        """
        if len(agent_ids) < 2:
            return 0.0
        
        # Count pairwise relationships
        relationship_count = 0
        total_strength = 0.0
        capability_coverage = set()
        
        for i, agent1 in enumerate(agent_ids):
            node1 = self._nodes.get(f"agent_{agent1}")
            if node1:
                capability_coverage.update(node1.capabilities)
            
            for agent2 in agent_ids[i+1:]:
                # Find relationship between agents
                node1 = self._nodes.get(f"agent_{agent1}")
                node2 = self._nodes.get(f"agent_{agent2}")
                
                if not node1 or not node2:
                    continue
                
                # Check for shared edges
                shared_edges = node1.connected_edges & node2.connected_edges
                
                for edge_id in shared_edges:
                    edge = self._edges.get(edge_id)
                    if edge:
                        relationship_count += 1
                        total_strength += edge.strength
        
        # Calculate synergy components
        possible_pairs = len(agent_ids) * (len(agent_ids) - 1) / 2
        connectivity_score = relationship_count / possible_pairs if possible_pairs > 0 else 0
        
        avg_strength = total_strength / relationship_count if relationship_count > 0 else 0
        
        # More unique capabilities = higher potential synergy
        diversity_score = min(1.0, len(capability_coverage) / (len(agent_ids) * 3))
        
        # Combined synergy score
        synergy = (connectivity_score * 0.4 + avg_strength * 0.3 + diversity_score * 0.3)
        
        return synergy
    
    def cluster_agents(
        self,
        min_cluster_size: int = 2,
        max_clusters: int = 10
    ) -> Dict[str, List[str]]:
        """
        Cluster agents based on relationship patterns
        
        Returns dict of cluster_id -> list of agent_ids
        """
        agent_nodes = list(self._nodes_by_type[NodeType.AGENT])
        
        if len(agent_nodes) < min_cluster_size:
            return {}
        
        # Simple clustering based on connectivity
        clusters = {}
        assigned = set()
        
        # Start with most connected agents
        sorted_nodes = sorted(
            agent_nodes,
            key=lambda n: len(self._nodes[n].connected_edges),
            reverse=True
        )
        
        cluster_id = 0
        
        for seed_node in sorted_nodes:
            if seed_node in assigned or cluster_id >= max_clusters:
                break
            
            # Find agents connected to this seed
            cluster_members = [seed_node]
            assigned.add(seed_node)
            
            node = self._nodes.get(seed_node)
            if not node:
                continue
            
            for edge_id in node.connected_edges:
                edge = self._edges.get(edge_id)
                if not edge:
                    continue
                
                for connected_node in edge.connected_nodes:
                    if (connected_node not in assigned and 
                        connected_node in agent_nodes and
                        edge.strength >= 0.5):
                        cluster_members.append(connected_node)
                        assigned.add(connected_node)
            
            if len(cluster_members) >= min_cluster_size:
                cluster_name = f"cluster_{cluster_id}"
                clusters[cluster_name] = [
                    n[6:] for n in cluster_members  # Remove "agent_" prefix
                ]
                cluster_id += 1
        
        self._clusters = {k: set(f"agent_{a}" for a in v) for k, v in clusters.items()}
        
        logger.info(f"Created {len(clusters)} agent clusters")
        return clusters
    
    def calculate_node_centrality(self) -> Dict[str, float]:
        """Calculate centrality scores for all nodes"""
        if self._metrics_cache_valid and "centrality" in self._metrics_cache:
            return self._metrics_cache["centrality"]
        
        centrality = {}
        
        for node_id, node in self._nodes.items():
            # Degree centrality
            degree = len(node.connected_edges)
            
            # Weighted degree (sum of edge weights)
            weighted_degree = 0.0
            for edge_id in node.connected_edges:
                edge = self._edges.get(edge_id)
                if edge:
                    weighted_degree += edge.weight * edge.strength
            
            # Normalize
            max_possible = len(self._edges) if self._edges else 1
            centrality[node_id] = (degree + weighted_degree) / (2 * max_possible)
            
            # Update node
            node.centrality_score = centrality[node_id]
        
        self._metrics_cache["centrality"] = centrality
        return centrality
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the relationship graph"""
        # Calculate metrics
        self.calculate_node_centrality()
        
        agent_count = len(self._nodes_by_type[NodeType.AGENT])
        edge_count = len(self._edges)
        
        avg_centrality = 0.0
        if self._nodes:
            avg_centrality = sum(n.centrality_score for n in self._nodes.values()) / len(self._nodes)
        
        edges_by_type = {
            rt.value: len(edges)
            for rt, edges in self._edges_by_type.items()
        }
        
        return {
            "total_nodes": len(self._nodes),
            "agent_nodes": agent_count,
            "capability_nodes": len(self._nodes_by_type[NodeType.CAPABILITY]),
            "total_edges": edge_count,
            "edges_by_type": edges_by_type,
            "avg_centrality": avg_centrality,
            "cluster_count": len(self._clusters),
            "density": edge_count / (agent_count * (agent_count - 1) / 2) if agent_count > 1 else 0
        }
