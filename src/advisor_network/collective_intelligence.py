"""
Collective Intelligence Synthesizer - Phase 12 Implementation

Implements collective intelligence synthesis for advisor agent network:
- Consensus decision making across agents
- Knowledge aggregation and synthesis
- Collective reasoning protocols
- Emergent intelligence coordination
"""

import asyncio
import uuid
import time
import logging
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np


logger = logging.getLogger(__name__)


class ConsensusProtocol(Enum):
    """Consensus protocols for collective decisions"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    UNANIMOUS = "unanimous"
    SUPERMAJORITY = "supermajority"  # 2/3 majority
    BYZANTINE_FAULT_TOLERANT = "bft"
    RAFT = "raft"
    PAXOS = "paxos"


class SynthesisStrategy(Enum):
    """Strategies for synthesizing collective intelligence"""
    AVERAGING = "averaging"
    WEIGHTED_AVERAGING = "weighted_averaging"
    BAYESIAN_AGGREGATION = "bayesian_aggregation"
    ENSEMBLE = "ensemble"
    HIERARCHICAL = "hierarchical"


class DecisionStatus(Enum):
    """Status of a collective decision"""
    PENDING = "pending"
    VOTING = "voting"
    CONSENSUS_REACHED = "consensus_reached"
    NO_CONSENSUS = "no_consensus"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class AgentVote:
    """A vote from an agent in a collective decision"""
    agent_id: str
    vote_value: Any
    confidence: float = 1.0
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class CollectiveDecision:
    """Represents a collective decision being made"""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Decision metadata
    topic: str = ""
    description: str = ""
    
    # Options and votes
    options: List[Any] = field(default_factory=list)
    votes: Dict[str, AgentVote] = field(default_factory=dict)
    
    # Participants
    required_participants: Set[str] = field(default_factory=set)
    actual_participants: Set[str] = field(default_factory=set)
    
    # Protocol and status
    protocol: ConsensusProtocol = ConsensusProtocol.MAJORITY_VOTE
    status: DecisionStatus = DecisionStatus.PENDING
    
    # Timing
    created_at: float = field(default_factory=time.time)
    deadline: float = 0.0
    completed_at: float = 0.0
    
    # Result
    final_decision: Any = None
    consensus_strength: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisResult:
    """Result of knowledge synthesis"""
    synthesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Input summary
    num_contributions: int = 0
    contributing_agents: List[str] = field(default_factory=list)
    
    # Synthesized output
    synthesized_knowledge: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    # Quality metrics
    coherence_score: float = 0.0
    agreement_score: float = 0.0
    novelty_score: float = 0.0
    
    # Processing info
    synthesis_time_ms: float = 0.0
    strategy_used: SynthesisStrategy = SynthesisStrategy.WEIGHTED_AVERAGING
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)


class CollectiveIntelligenceSynthesizer:
    """
    Synthesizes collective intelligence from distributed advisor agents
    
    Features:
    - Multiple consensus protocols
    - Knowledge aggregation and synthesis
    - Weighted confidence integration
    - Emergent pattern detection
    - Collective decision orchestration
    """
    
    def __init__(
        self,
        default_protocol: ConsensusProtocol = ConsensusProtocol.WEIGHTED_VOTE,
        default_strategy: SynthesisStrategy = SynthesisStrategy.WEIGHTED_AVERAGING
    ):
        self.default_protocol = default_protocol
        self.default_strategy = default_strategy
        
        # Active decisions
        self._decisions: Dict[str, CollectiveDecision] = {}
        
        # Agent weights for voting
        self._agent_weights: Dict[str, float] = {}
        
        # Knowledge base from synthesis
        self._collective_knowledge: Dict[str, Any] = {}
        
        # Statistics
        self._decisions_made: int = 0
        self._consensus_reached: int = 0
        self._total_synthesis_time_ms: float = 0.0
        
        # Event callbacks
        self._decision_callbacks: List[Callable[[CollectiveDecision], None]] = []
        
        logger.info(
            f"Initialized CollectiveIntelligenceSynthesizer "
            f"(protocol: {default_protocol.value})"
        )
    
    def set_agent_weight(self, agent_id: str, weight: float):
        """Set the voting weight for an agent"""
        self._agent_weights[agent_id] = max(0.0, min(2.0, weight))
    
    def get_agent_weight(self, agent_id: str) -> float:
        """Get the voting weight for an agent"""
        return self._agent_weights.get(agent_id, 1.0)
    
    async def create_collective_decision(
        self,
        topic: str,
        options: List[Any],
        required_participants: List[str],
        protocol: ConsensusProtocol = None,
        timeout_seconds: float = 60.0,
        description: str = "",
        metadata: Dict[str, Any] = None
    ) -> CollectiveDecision:
        """
        Create a new collective decision process
        
        Returns the CollectiveDecision object
        """
        decision = CollectiveDecision(
            topic=topic,
            description=description,
            options=options,
            required_participants=set(required_participants),
            protocol=protocol or self.default_protocol,
            status=DecisionStatus.VOTING,
            deadline=time.time() + timeout_seconds,
            metadata=metadata or {}
        )
        
        self._decisions[decision.decision_id] = decision
        
        logger.info(
            f"Created collective decision {decision.decision_id[:8]} "
            f"on '{topic}' with {len(required_participants)} participants"
        )
        
        return decision
    
    async def submit_vote(
        self,
        decision_id: str,
        agent_id: str,
        vote_value: Any,
        confidence: float = 1.0,
        reasoning: str = ""
    ) -> bool:
        """
        Submit a vote from an agent
        
        Returns True if vote was accepted
        """
        decision = self._decisions.get(decision_id)
        if not decision:
            logger.warning(f"Decision {decision_id} not found")
            return False
        
        if decision.status != DecisionStatus.VOTING:
            logger.warning(f"Decision {decision_id} is not accepting votes")
            return False
        
        if time.time() > decision.deadline:
            decision.status = DecisionStatus.TIMEOUT
            return False
        
        # Validate vote value
        if vote_value not in decision.options:
            logger.warning(f"Invalid vote value from {agent_id}")
            return False
        
        # Record vote
        vote = AgentVote(
            agent_id=agent_id,
            vote_value=vote_value,
            confidence=confidence,
            reasoning=reasoning
        )
        
        decision.votes[agent_id] = vote
        decision.actual_participants.add(agent_id)
        
        logger.debug(f"Agent {agent_id} voted on decision {decision_id[:8]}")
        
        # Check if all participants have voted
        if decision.actual_participants >= decision.required_participants:
            await self._finalize_decision(decision)
        
        return True
    
    async def check_consensus(
        self,
        decision_id: str
    ) -> Tuple[bool, Optional[Any], float]:
        """
        Check if consensus has been reached
        
        Returns (has_consensus, winning_option, consensus_strength)
        """
        decision = self._decisions.get(decision_id)
        if not decision:
            return False, None, 0.0
        
        if decision.status == DecisionStatus.CONSENSUS_REACHED:
            return True, decision.final_decision, decision.consensus_strength
        
        # Calculate current consensus
        return self._calculate_consensus(decision)
    
    async def finalize_decision(self, decision_id: str) -> Optional[CollectiveDecision]:
        """Force finalization of a decision"""
        decision = self._decisions.get(decision_id)
        if not decision:
            return None
        
        if decision.status not in [DecisionStatus.PENDING, DecisionStatus.VOTING]:
            return decision
        
        await self._finalize_decision(decision)
        return decision
    
    async def synthesize_knowledge(
        self,
        contributions: List[Dict[str, Any]],
        contributing_agents: List[str],
        strategy: SynthesisStrategy = None
    ) -> SynthesisResult:
        """
        Synthesize knowledge from multiple agent contributions
        
        Returns SynthesisResult with aggregated knowledge
        """
        start_time = time.time()
        
        active_strategy = strategy or self.default_strategy
        
        if not contributions:
            return SynthesisResult(
                num_contributions=0,
                confidence=0.0,
                synthesis_time_ms=0.0
            )
        
        # Apply synthesis strategy
        if active_strategy == SynthesisStrategy.AVERAGING:
            synthesized = self._average_synthesis(contributions)
        elif active_strategy == SynthesisStrategy.WEIGHTED_AVERAGING:
            synthesized = self._weighted_synthesis(contributions, contributing_agents)
        elif active_strategy == SynthesisStrategy.BAYESIAN_AGGREGATION:
            synthesized = self._bayesian_synthesis(contributions)
        elif active_strategy == SynthesisStrategy.ENSEMBLE:
            synthesized = self._ensemble_synthesis(contributions)
        else:
            synthesized = self._average_synthesis(contributions)
        
        # Calculate quality metrics
        coherence = self._calculate_coherence(contributions)
        agreement = self._calculate_agreement(contributions)
        novelty = self._calculate_novelty(synthesized)
        
        synthesis_time = (time.time() - start_time) * 1000
        
        result = SynthesisResult(
            num_contributions=len(contributions),
            contributing_agents=contributing_agents,
            synthesized_knowledge=synthesized,
            confidence=min(1.0, (coherence + agreement) / 2),
            coherence_score=coherence,
            agreement_score=agreement,
            novelty_score=novelty,
            synthesis_time_ms=synthesis_time,
            strategy_used=active_strategy
        )
        
        # Update collective knowledge base
        self._update_collective_knowledge(synthesized)
        
        self._total_synthesis_time_ms += synthesis_time
        
        logger.debug(
            f"Synthesized knowledge from {len(contributions)} contributions "
            f"in {synthesis_time:.2f}ms"
        )
        
        return result
    
    async def aggregate_predictions(
        self,
        predictions: List[Tuple[str, float, float]],  # (agent_id, prediction, confidence)
    ) -> Tuple[float, float]:
        """
        Aggregate predictions from multiple agents
        
        Returns (aggregated_prediction, combined_confidence)
        """
        if not predictions:
            return 0.0, 0.0
        
        # Weighted average based on confidence and agent weight
        total_weight = 0.0
        weighted_sum = 0.0
        
        for agent_id, prediction, confidence in predictions:
            agent_weight = self.get_agent_weight(agent_id)
            combined_weight = agent_weight * confidence
            
            weighted_sum += prediction * combined_weight
            total_weight += combined_weight
        
        if total_weight == 0:
            return 0.0, 0.0
        
        aggregated = weighted_sum / total_weight
        
        # Calculate combined confidence
        confidences = [c for _, _, c in predictions]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Agreement bonus: higher confidence if predictions are similar
        prediction_values = [p for _, p, _ in predictions]
        if len(prediction_values) > 1:
            variance = np.var(prediction_values)
            agreement_factor = max(0.5, 1.0 - variance / (np.mean(prediction_values) + 1e-6))
        else:
            agreement_factor = 1.0
        
        combined_confidence = avg_confidence * agreement_factor
        
        return aggregated, combined_confidence
    
    def get_decision_status(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a collective decision"""
        decision = self._decisions.get(decision_id)
        if not decision:
            return None
        
        return {
            "decision_id": decision.decision_id,
            "topic": decision.topic,
            "status": decision.status.value,
            "votes_received": len(decision.votes),
            "required_participants": len(decision.required_participants),
            "final_decision": decision.final_decision,
            "consensus_strength": decision.consensus_strength,
            "time_remaining": max(0, decision.deadline - time.time())
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get synthesizer statistics"""
        avg_synthesis_time = (
            self._total_synthesis_time_ms / max(1, self._decisions_made)
        )
        
        consensus_rate = (
            self._consensus_reached / max(1, self._decisions_made)
        )
        
        return {
            "active_decisions": len([d for d in self._decisions.values() 
                                     if d.status == DecisionStatus.VOTING]),
            "total_decisions": self._decisions_made,
            "consensus_reached": self._consensus_reached,
            "consensus_rate": consensus_rate,
            "avg_synthesis_time_ms": avg_synthesis_time,
            "collective_knowledge_items": len(self._collective_knowledge),
            "registered_agents": len(self._agent_weights)
        }
    
    def register_decision_callback(
        self,
        callback: Callable[[CollectiveDecision], None]
    ):
        """Register callback for decision completion"""
        self._decision_callbacks.append(callback)
    
    # Private methods
    
    async def _finalize_decision(self, decision: CollectiveDecision):
        """Finalize a decision and determine outcome"""
        has_consensus, winner, strength = self._calculate_consensus(decision)
        
        decision.completed_at = time.time()
        decision.final_decision = winner
        decision.consensus_strength = strength
        
        if has_consensus:
            decision.status = DecisionStatus.CONSENSUS_REACHED
            self._consensus_reached += 1
        else:
            decision.status = DecisionStatus.NO_CONSENSUS
        
        self._decisions_made += 1
        
        # Notify callbacks
        for callback in self._decision_callbacks:
            try:
                callback(decision)
            except Exception as e:
                logger.error(f"Error in decision callback: {e}")
        
        logger.info(
            f"Decision {decision.decision_id[:8]} finalized: "
            f"{'consensus' if has_consensus else 'no consensus'} "
            f"(strength: {strength:.2f})"
        )
    
    def _calculate_consensus(
        self,
        decision: CollectiveDecision
    ) -> Tuple[bool, Optional[Any], float]:
        """Calculate consensus based on protocol"""
        votes = decision.votes
        
        if not votes:
            return False, None, 0.0
        
        # Count votes per option
        vote_counts: Dict[Any, float] = {}
        total_weight = 0.0
        
        for agent_id, vote in votes.items():
            weight = self.get_agent_weight(agent_id) * vote.confidence
            
            if vote.vote_value not in vote_counts:
                vote_counts[vote.vote_value] = 0.0
            
            vote_counts[vote.vote_value] += weight
            total_weight += weight
        
        if total_weight == 0:
            return False, None, 0.0
        
        # Find winner
        winner = max(vote_counts, key=vote_counts.get)
        winner_weight = vote_counts[winner]
        strength = winner_weight / total_weight
        
        # Check against protocol requirements
        protocol = decision.protocol
        
        if protocol == ConsensusProtocol.MAJORITY_VOTE:
            has_consensus = strength > 0.5
            
        elif protocol == ConsensusProtocol.WEIGHTED_VOTE:
            has_consensus = strength > 0.5
            
        elif protocol == ConsensusProtocol.SUPERMAJORITY:
            has_consensus = strength >= (2.0 / 3.0)
            
        elif protocol == ConsensusProtocol.UNANIMOUS:
            has_consensus = strength >= 0.99
            
        else:
            has_consensus = strength > 0.5
        
        return has_consensus, winner, strength
    
    def _average_synthesis(
        self,
        contributions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Simple averaging synthesis"""
        if not contributions:
            return {}
        
        synthesized = {}
        
        # Aggregate numeric values
        numeric_keys = set()
        for contrib in contributions:
            for key, value in contrib.items():
                if isinstance(value, (int, float)):
                    numeric_keys.add(key)
        
        for key in numeric_keys:
            values = [c.get(key, 0) for c in contributions if key in c]
            if values:
                synthesized[key] = sum(values) / len(values)
        
        # Merge non-numeric values (take most common)
        for contrib in contributions:
            for key, value in contrib.items():
                if key not in synthesized:
                    synthesized[key] = value
        
        return synthesized
    
    def _weighted_synthesis(
        self,
        contributions: List[Dict[str, Any]],
        contributing_agents: List[str]
    ) -> Dict[str, Any]:
        """Weighted averaging synthesis based on agent weights"""
        if not contributions or len(contributions) != len(contributing_agents):
            return self._average_synthesis(contributions)
        
        synthesized = {}
        weights = [self.get_agent_weight(aid) for aid in contributing_agents]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return self._average_synthesis(contributions)
        
        # Weighted average for numeric values
        numeric_keys = set()
        for contrib in contributions:
            for key, value in contrib.items():
                if isinstance(value, (int, float)):
                    numeric_keys.add(key)
        
        for key in numeric_keys:
            weighted_sum = 0.0
            key_weight = 0.0
            
            for i, contrib in enumerate(contributions):
                if key in contrib:
                    weighted_sum += contrib[key] * weights[i]
                    key_weight += weights[i]
            
            if key_weight > 0:
                synthesized[key] = weighted_sum / key_weight
        
        # Merge non-numeric values
        for contrib in contributions:
            for key, value in contrib.items():
                if key not in synthesized:
                    synthesized[key] = value
        
        return synthesized
    
    def _bayesian_synthesis(
        self,
        contributions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Bayesian aggregation synthesis"""
        # Simplified Bayesian approach
        # In production, would use proper Bayesian inference
        return self._weighted_synthesis(
            contributions,
            [str(i) for i in range(len(contributions))]
        )
    
    def _ensemble_synthesis(
        self,
        contributions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Ensemble synthesis preserving all perspectives"""
        synthesized = {
            "ensemble_members": len(contributions),
            "contributions": contributions,
            "consensus_values": self._average_synthesis(contributions)
        }
        return synthesized
    
    def _calculate_coherence(
        self,
        contributions: List[Dict[str, Any]]
    ) -> float:
        """Calculate coherence score for contributions"""
        if len(contributions) < 2:
            return 1.0
        
        # Check key overlap
        all_keys = set()
        key_counts = {}
        
        for contrib in contributions:
            for key in contrib.keys():
                all_keys.add(key)
                key_counts[key] = key_counts.get(key, 0) + 1
        
        if not all_keys:
            return 0.0
        
        # Higher coherence if same keys appear across contributions
        avg_overlap = sum(key_counts.values()) / len(key_counts) / len(contributions)
        
        return min(1.0, avg_overlap)
    
    def _calculate_agreement(
        self,
        contributions: List[Dict[str, Any]]
    ) -> float:
        """Calculate agreement score between contributions"""
        if len(contributions) < 2:
            return 1.0
        
        # Find common numeric keys
        numeric_keys = set()
        for contrib in contributions:
            for key, value in contrib.items():
                if isinstance(value, (int, float)):
                    numeric_keys.add(key)
        
        if not numeric_keys:
            return 0.5
        
        # Calculate variance for each key
        agreement_scores = []
        
        for key in numeric_keys:
            values = [c.get(key) for c in contributions if key in c and isinstance(c.get(key), (int, float))]
            
            if len(values) >= 2:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                
                # Normalize variance to agreement score
                if mean != 0:
                    coefficient_of_variation = (variance ** 0.5) / abs(mean)
                    agreement = max(0, 1.0 - coefficient_of_variation)
                else:
                    agreement = 1.0 if variance == 0 else 0.0
                
                agreement_scores.append(agreement)
        
        if not agreement_scores:
            return 0.5
        
        return sum(agreement_scores) / len(agreement_scores)
    
    def _calculate_novelty(
        self,
        synthesized: Dict[str, Any]
    ) -> float:
        """Calculate novelty of synthesized knowledge"""
        if not self._collective_knowledge:
            return 1.0
        
        # Check overlap with existing knowledge
        new_keys = set(synthesized.keys()) - set(self._collective_knowledge.keys())
        
        novelty = len(new_keys) / max(1, len(synthesized))
        
        return novelty
    
    def _update_collective_knowledge(self, synthesized: Dict[str, Any]):
        """Update collective knowledge base"""
        for key, value in synthesized.items():
            if key not in self._collective_knowledge:
                self._collective_knowledge[key] = value
            elif isinstance(value, (int, float)) and isinstance(self._collective_knowledge[key], (int, float)):
                # Exponential moving average for numeric values
                alpha = 0.3
                self._collective_knowledge[key] = (
                    alpha * value + (1 - alpha) * self._collective_knowledge[key]
                )


# Helper type hint
Tuple = tuple
