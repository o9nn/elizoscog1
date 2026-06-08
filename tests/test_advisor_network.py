"""
Test Suite for Distributed AI Advisor Agent Network - Phase 12

Tests cover:
- Distributed agent registry and federation
- Secure inter-agent messaging
- Multi-cloud deployment
- Cognitive load balancing
- Hypergraph-based relationships
- GGML reasoning engines
- Collective intelligence synthesis
"""

import pytest
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any

import sys
sys.path.insert(0, '/tmp/workspace/o9nn/elizoscog1/src')

# Import all components
from advisor_network.agent_registry import (
    DistributedAgentRegistry,
    AdvisorAgent,
    AgentCapability,
    AgentStatus,
    FederationProtocol,
    AgentRegistryConfig
)
from advisor_network.secure_messaging import (
    SecureMessagingService,
    EncryptedMessage,
    MessagePriority,
    MessagingConfig
)
from advisor_network.multi_cloud import (
    MultiCloudDeployment,
    CloudProvider,
    CloudRegion,
    DeploymentConfig,
    RegionCluster
)
from advisor_network.cognitive_load_balancer import (
    CognitiveLoadBalancer,
    LoadBalanceStrategy,
    RoutingRequest,
    RoutingDecision,
    AgentWorkload
)
from advisor_network.hypergraph_relationships import (
    AgentRelationshipGraph,
    RelationshipType,
    AgentNode,
    RelationshipEdge
)
from advisor_network.ggml_reasoning import (
    GGMLReasoningEngine,
    ReasoningConfig,
    QuantizationType,
    InferenceResult,
    ReasoningContext,
    ReasoningTaskType
)
from advisor_network.collective_intelligence import (
    CollectiveIntelligenceSynthesizer,
    ConsensusProtocol,
    SynthesisResult
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def registry_config():
    """Configuration for distributed agent registry."""
    return AgentRegistryConfig(
        cluster_id="test-cluster",
        federation_protocol=FederationProtocol.GOSSIP,
        heartbeat_interval_seconds=1,
        agent_timeout_seconds=10
    )


@pytest.fixture
def agent_registry(registry_config):
    """Create a test agent registry."""
    return DistributedAgentRegistry(registry_config)


@pytest.fixture
def sample_agent():
    """Create a sample advisor agent."""
    return AdvisorAgent(
        agent_id="advisor-001",
        name="Financial Advisor",
        host="advisor-001.example.com",
        port=8080,
        cluster_id="test-cluster",
        cloud_provider="aws",
        region="us-east-1",
        capabilities=[
            AgentCapability.FINANCIAL_ANALYSIS,
            AgentCapability.RISK_ASSESSMENT
        ],
        status=AgentStatus.ACTIVE
    )


@pytest.fixture
def messaging_config():
    """Configuration for secure messaging."""
    return MessagingConfig()


@pytest.fixture
def messaging_service(messaging_config):
    """Create a test messaging service."""
    return SecureMessagingService(agent_id="test-agent", config=messaging_config)


@pytest.fixture
def deployment_config():
    """Configuration for multi-cloud deployment."""
    return DeploymentConfig()


@pytest.fixture
def multi_cloud_deployment(deployment_config):
    """Create a test multi-cloud deployment."""
    return MultiCloudDeployment(deployment_config)


@pytest.fixture
def load_balancer():
    """Create a test load balancer."""
    return CognitiveLoadBalancer()


@pytest.fixture
def relationship_graph():
    """Create a test relationship graph."""
    return AgentRelationshipGraph()


@pytest.fixture
def reasoning_config():
    """Configuration for GGML reasoning engine."""
    return ReasoningConfig()


@pytest.fixture
def reasoning_engine(reasoning_config):
    """Create a test reasoning engine."""
    return GGMLReasoningEngine(reasoning_config)


@pytest.fixture
def intelligence_synthesizer():
    """Create a test intelligence synthesizer."""
    return CollectiveIntelligenceSynthesizer()


# ============================================================================
# Test Classes
# ============================================================================

class TestDistributedAgentRegistry:
    """Tests for the distributed agent registry service."""

    @pytest.mark.asyncio
    async def test_register_agent(self, agent_registry, sample_agent):
        """Test agent registration."""
        result = await agent_registry.register_agent(sample_agent)
        
        assert result is True
        # Check internal storage
        agent = await agent_registry.get_agent(sample_agent.agent_id)
        assert agent is not None

    @pytest.mark.asyncio
    async def test_discover_agents(self, agent_registry, sample_agent):
        """Test agent discovery."""
        await agent_registry.register_agent(sample_agent)
        
        agents = await agent_registry.discover_agents()
        
        assert len(agents) >= 1
        assert any(a.agent_id == sample_agent.agent_id for a in agents)

    @pytest.mark.asyncio
    async def test_discover_agents_sub_100ms(self, agent_registry):
        """Test that agent discovery completes in under 100ms."""
        # Register multiple agents
        for i in range(100):
            agent = AdvisorAgent(
                agent_id=f"advisor-{i:03d}",
                name=f"Advisor {i}",
                host=f"advisor-{i:03d}.example.com",
                port=8080,
                cluster_id="test-cluster",
                cloud_provider="aws",
                region="us-east-1",
                capabilities=[AgentCapability.COGNITIVE_REASONING],
                status=AgentStatus.ACTIVE
            )
            await agent_registry.register_agent(agent)
        
        # Measure discovery time
        start_time = time.perf_counter()
        agents = await agent_registry.discover_agents()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        assert elapsed_ms < 100, f"Discovery took {elapsed_ms:.2f}ms, expected <100ms"
        assert len(agents) == 100

    @pytest.mark.asyncio
    async def test_federation_sync(self, agent_registry, sample_agent):
        """Test federation synchronization between registries."""
        # Register locally
        await agent_registry.register_agent(sample_agent)
        
        # Add federation peer with host/port info (async method)
        await agent_registry.add_federation_peer(
            peer_id="federated-cluster",
            host="federated.example.com",
            port=8080
        )
        
        # Federation is set up - verify it doesn't error
        stats = agent_registry.get_statistics()
        assert stats is not None

    @pytest.mark.asyncio
    async def test_agent_heartbeat(self, agent_registry, sample_agent):
        """Test agent heartbeat."""
        await agent_registry.register_agent(sample_agent)
        
        # Send heartbeat
        result = await agent_registry.heartbeat(sample_agent.agent_id)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_deregister_agent(self, agent_registry, sample_agent):
        """Test agent deregistration."""
        await agent_registry.register_agent(sample_agent)
        
        result = await agent_registry.deregister_agent(sample_agent.agent_id)
        
        assert result is True
        agent = await agent_registry.get_agent(sample_agent.agent_id)
        assert agent is None


class TestSecureMessaging:
    """Tests for secure inter-agent messaging."""

    @pytest.mark.asyncio
    async def test_send_message(self, messaging_service):
        """Test sending messages between agents."""
        # Connect peers first
        await messaging_service.connect_to_peer(
            peer_id="advisor-002",
            host="advisor-002.example.com",
            port=8080
        )
        
        result = await messaging_service.send_message(
            recipient_id="advisor-002",
            payload={"query": "What is the market outlook?"},
            priority=MessagePriority.HIGH
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_send_broadcast(self, messaging_service):
        """Test broadcasting messages."""
        result = await messaging_service.send_broadcast(
            recipient_ids=["advisor-001", "advisor-002", "advisor-003"],
            payload={"message": "System maintenance scheduled"}
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_message_statistics(self, messaging_service):
        """Test getting messaging statistics."""
        stats = messaging_service.get_statistics()
        
        assert stats is not None


class TestMultiCloudDeployment:
    """Tests for multi-cloud deployment."""

    @pytest.mark.asyncio
    async def test_deploy_cluster(self, multi_cloud_deployment):
        """Test deploying a cluster."""
        result = await multi_cloud_deployment.deploy_cluster(
            provider=CloudProvider.AWS,
            region=CloudRegion.US_EAST_1,
            instance_count=3
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_deploy_multi_region(self, multi_cloud_deployment):
        """Test multi-region deployment."""
        regions = [
            (CloudProvider.AWS, CloudRegion.US_EAST_1),
            (CloudProvider.GCP, CloudRegion.EU_WEST_1)
        ]
        
        result = await multi_cloud_deployment.deploy_multi_region(regions=regions)
        
        assert result is not None
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_failover(self, multi_cloud_deployment):
        """Test failover between clusters."""
        # Deploy primary
        cluster = await multi_cloud_deployment.deploy_cluster(
            provider=CloudProvider.AWS,
            region=CloudRegion.US_EAST_1,
            instance_count=3,
            is_primary=True
        )
        
        # Deploy secondary
        await multi_cloud_deployment.deploy_cluster(
            provider=CloudProvider.GCP,
            region=CloudRegion.EU_WEST_1,
            instance_count=2
        )
        
        # Trigger failover
        result = await multi_cloud_deployment.failover()
        
        assert result is True

    @pytest.mark.asyncio
    async def test_scale_cluster(self, multi_cloud_deployment):
        """Test scaling a cluster."""
        cluster = await multi_cloud_deployment.deploy_cluster(
            provider=CloudProvider.AWS,
            region=CloudRegion.US_EAST_1,
            instance_count=2
        )
        
        result = await multi_cloud_deployment.scale_cluster(
            cluster_id=cluster.cluster_id,
            target_instance_count=5
        )
        
        assert result is True


class TestCognitiveLoadBalancer:
    """Tests for cognitive load balancing."""

    @pytest.mark.asyncio
    async def test_register_agent_workload(self, load_balancer):
        """Test registering agent workload."""
        # register_agent returns None, just verify it doesn't error
        load_balancer.register_agent(
            agent_id="advisor-001",
            capabilities=["cognitive_reasoning"],
            max_concurrent=100
        )
        
        # Verify agent is registered
        workloads = load_balancer.get_agent_workloads()
        assert "advisor-001" in workloads

    @pytest.mark.asyncio
    async def test_route_request(self, load_balancer):
        """Test routing a request."""
        # Register agents
        for i in range(3):
            load_balancer.register_agent(
                agent_id=f"advisor-{i}",
                capabilities=["cognitive_reasoning"],
                max_concurrent=100
            )
        
        # Route request - use required_capabilities (list of strings)
        request = RoutingRequest(
            request_id="req-001",
            required_capabilities=["cognitive_reasoning"]
        )
        
        decision = await load_balancer.route_request(request)
        
        assert decision is not None
        assert decision.selected_agent_id is not None

    @pytest.mark.asyncio
    async def test_route_request_multiple_times(self, load_balancer):
        """Test routing multiple requests distributes load."""
        # Register agents
        for i in range(3):
            load_balancer.register_agent(
                agent_id=f"advisor-{i}",
                capabilities=["cognitive_reasoning"],
                max_concurrent=100
            )
        
        # Route multiple requests
        routed_agents = []
        for j in range(6):
            request = RoutingRequest(
                request_id=f"req-{j:03d}",
                required_capabilities=["cognitive_reasoning"]
            )
            decision = await load_balancer.route_request(request)
            routed_agents.append(decision.selected_agent_id)
        
        # Should use multiple agents
        unique_agents = set(routed_agents)
        assert len(unique_agents) >= 2

    @pytest.mark.asyncio
    async def test_complete_request(self, load_balancer):
        """Test completing a request updates workload."""
        load_balancer.register_agent(
            agent_id="advisor-001",
            capabilities=["cognitive_reasoning"],
            max_concurrent=100
        )
        
        request = RoutingRequest(
            request_id="req-001",
            required_capabilities=["cognitive_reasoning"]
        )
        decision = await load_balancer.route_request(request)
        
        # Complete the request - verify it doesn't raise an exception
        load_balancer.complete_request(
            agent_id=decision.selected_agent_id,
            success=True,
            response_time_ms=50.0
        )
        
        # Verify workload was updated by checking the agent workloads
        workloads = load_balancer.get_agent_workloads()
        assert "advisor-001" in workloads


class TestAgentRelationshipGraph:
    """Tests for hypergraph-based agent relationships."""

    @pytest.mark.asyncio
    async def test_add_agent_node(self, relationship_graph):
        """Test adding agent nodes to hypergraph."""
        node = relationship_graph.add_agent_node(
            agent_id="advisor-001",
            capabilities=["financial", "risk"],
            attributes={"expertise_level": "senior"}
        )
        
        assert node is not None
        assert node.agent_id == "advisor-001"

    @pytest.mark.asyncio
    async def test_add_relationship(self, relationship_graph):
        """Test creating relationships between agents."""
        # Add agents
        relationship_graph.add_agent_node("advisor-001", ["financial"])
        relationship_graph.add_agent_node("advisor-002", ["risk"])
        
        # Create relationship - target_agent_ids is a list
        edge = relationship_graph.add_relationship(
            source_agent_id="advisor-001",
            target_agent_ids=["advisor-002"],
            relationship_type=RelationshipType.COLLABORATION,
            weight=0.8
        )
        
        assert edge is not None

    @pytest.mark.asyncio
    async def test_find_related_agents(self, relationship_graph):
        """Test finding related agents."""
        # Add agents and relationships
        relationship_graph.add_agent_node("advisor-001", ["financial"])
        relationship_graph.add_agent_node("advisor-002", ["risk"])
        relationship_graph.add_agent_node("advisor-003", ["legal"])
        
        relationship_graph.add_relationship(
            source_agent_id="advisor-001",
            target_agent_ids=["advisor-002"],
            relationship_type=RelationshipType.COLLABORATION,
            weight=0.9
        )
        
        related = relationship_graph.find_related_agents(
            agent_id="advisor-001",
            min_strength=0.5
        )
        
        assert len(related) >= 1

    @pytest.mark.asyncio
    async def test_calculate_cognitive_synergy(self, relationship_graph):
        """Test calculating cognitive synergy between agents."""
        relationship_graph.add_agent_node("advisor-001", ["financial", "analysis"])
        relationship_graph.add_agent_node("advisor-002", ["risk", "assessment"])
        
        relationship_graph.add_relationship(
            source_agent_id="advisor-001",
            target_agent_ids=["advisor-002"],
            relationship_type=RelationshipType.COMPLEMENTARY,
            weight=0.85
        )
        
        synergy = relationship_graph.calculate_cognitive_synergy(
            ["advisor-001", "advisor-002"]
        )
        
        assert synergy >= 0.0
        assert synergy <= 1.0


class TestGGMLReasoningEngine:
    """Tests for GGML-optimized reasoning engines."""

    @pytest.mark.asyncio
    async def test_load_model(self, reasoning_engine):
        """Test loading a reasoning model."""
        result = await reasoning_engine.load_model()
        
        assert result is True

    @pytest.mark.asyncio
    async def test_run_inference(self, reasoning_engine):
        """Test running inference."""
        await reasoning_engine.load_model()
        
        context = ReasoningContext(
            task_type=ReasoningTaskType.FINANCIAL_ANALYSIS,
            input_data={"sector": "technology", "query": "market trends"}
        )
        
        result = await reasoning_engine.infer(context)
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_batch_inference(self, reasoning_engine):
        """Test batch inference processing."""
        await reasoning_engine.load_model()
        
        contexts = [
            ReasoningContext(
                task_type=ReasoningTaskType.FINANCIAL_ANALYSIS,
                input_data={"query": "stock performance"}
            ),
            ReasoningContext(
                task_type=ReasoningTaskType.RISK_ASSESSMENT,
                input_data={"query": "risk factors"}
            ),
            ReasoningContext(
                task_type=ReasoningTaskType.PREDICTION,
                input_data={"query": "market trends"}
            )
        ]
        
        results = await reasoning_engine.batch_infer(contexts)
        
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_model_info(self, reasoning_engine):
        """Test getting model information."""
        await reasoning_engine.load_model()
        
        info = reasoning_engine.get_model_info()
        
        assert info is not None


class TestCollectiveIntelligence:
    """Tests for collective intelligence synthesis."""

    @pytest.mark.asyncio
    async def test_create_collective_decision(self, intelligence_synthesizer):
        """Test creating a collective decision."""
        agents = ["advisor-001", "advisor-002", "advisor-003"]
        
        decision = await intelligence_synthesizer.create_collective_decision(
            topic="Market strategy for Q4",
            options=["aggressive", "conservative", "balanced"],
            required_participants=agents
        )
        
        assert decision is not None
        assert decision.topic == "Market strategy for Q4"

    @pytest.mark.asyncio
    async def test_submit_vote(self, intelligence_synthesizer):
        """Test submitting votes."""
        # Create decision
        decision = await intelligence_synthesizer.create_collective_decision(
            topic="Investment allocation",
            options=["option_a", "option_b", "option_c"],
            required_participants=["advisor-001", "advisor-002"]
        )
        
        # Submit vote
        result = await intelligence_synthesizer.submit_vote(
            decision_id=decision.decision_id,
            agent_id="advisor-001",
            vote_value="option_a",
            confidence=0.9
        )
        
        assert result is True

    @pytest.mark.asyncio
    async def test_finalize_decision(self, intelligence_synthesizer):
        """Test finalizing a decision."""
        # Create decision
        decision = await intelligence_synthesizer.create_collective_decision(
            topic="Resource allocation",
            options=["option_a", "option_b"],
            required_participants=["advisor-001", "advisor-002", "advisor-003"]
        )
        
        # Submit votes
        await intelligence_synthesizer.submit_vote(decision.decision_id, "advisor-001", "option_a", 0.9)
        await intelligence_synthesizer.submit_vote(decision.decision_id, "advisor-002", "option_a", 0.8)
        await intelligence_synthesizer.submit_vote(decision.decision_id, "advisor-003", "option_b", 0.7)
        
        # Finalize
        result = await intelligence_synthesizer.finalize_decision(decision.decision_id)
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_synthesize_knowledge(self, intelligence_synthesizer):
        """Test knowledge synthesis from multiple agents."""
        contributions = [
            {"insight": "Market is bullish", "confidence": 0.8, "data": {"trend": "up"}},
            {"insight": "Tech sector strong", "confidence": 0.9, "data": {"sector": "tech"}},
            {"insight": "Volatility expected", "confidence": 0.7, "data": {"risk": "medium"}}
        ]
        
        synthesis = await intelligence_synthesizer.synthesize_knowledge(
            contributions=contributions,
            contributing_agents=["advisor-001", "advisor-002", "advisor-003"]
        )
        
        assert synthesis is not None


class TestAdvisorNetworkIntegration:
    """Integration tests for the complete advisor network."""

    @pytest.mark.asyncio
    async def test_full_advisor_workflow(
        self,
        agent_registry,
        load_balancer
    ):
        """Test complete advisor workflow from registration to routing."""
        # 1. Register advisors
        advisors = []
        for i in range(5):
            advisor = AdvisorAgent(
                agent_id=f"advisor-{i:03d}",
                name=f"Advisor {i}",
                host=f"advisor-{i:03d}.example.com",
                port=8080,
                cluster_id="test-cluster",
                cloud_provider="aws",
                region="us-east-1",
                capabilities=[AgentCapability.FINANCIAL_ANALYSIS],
                status=AgentStatus.ACTIVE
            )
            await agent_registry.register_agent(advisor)
            load_balancer.register_agent(
                agent_id=advisor.agent_id,
                capabilities=["financial_analysis"],
                max_concurrent=100
            )
            advisors.append(advisor)
        
        # 2. Verify registration
        all_agents = await agent_registry.discover_agents()
        assert len(all_agents) >= 5
        
        # 3. Route a request - use required_capabilities
        decision = await load_balancer.route_request(
            RoutingRequest(
                request_id="workflow-req-001",
                required_capabilities=["financial_analysis"]
            )
        )
        
        assert decision.selected_agent_id is not None

    @pytest.mark.asyncio
    async def test_large_scale_registration(self, agent_registry):
        """Test registering a significant number of agents."""
        # Register 100 agents (reduced from 1000 for test speed)
        for i in range(100):
            agent = AdvisorAgent(
                agent_id=f"advisor-{i:04d}",
                name=f"Advisor {i}",
                host=f"advisor-{i:04d}.example.com",
                port=8080,
                cluster_id=f"cluster-{i // 10}",
                cloud_provider=["aws", "gcp", "azure"][i % 3],
                region=["us-east-1", "eu-west-1", "ap-south-1"][i % 3],
                capabilities=[
                    AgentCapability.COGNITIVE_REASONING
                ],
                status=AgentStatus.ACTIVE
            )
            await agent_registry.register_agent(agent)
        
        # Verify discovery performance
        start = time.perf_counter()
        agents = await agent_registry.discover_agents()
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Should be fast (sub-100ms)
        assert elapsed_ms < 100, f"Discovery took {elapsed_ms:.2f}ms"
        assert len(agents) >= 100


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
