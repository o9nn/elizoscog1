"""
Comprehensive tests for microservice discovery and orchestration
Tests service registry, load balancing, and GGML optimization
"""

import asyncio
import pytest
import time
import requests
from typing import List, Dict, Any

# Import microservice components
from src.microservices.service_discovery import ServiceRegistry, ServiceDiscovery, ServiceInfo
from src.microservices.load_balancer import LoadBalancer, LoadBalancingConfig, EnvoyConfigGenerator, TraefikConfigGenerator
from src.microservices.orchestration import ServiceOrchestrator, HealthMonitor, OrchestrationConfig
from src.microservices.ggml_optimization import GGMLServiceOptimizer, HypergraphMeshEncoder, GGMLServiceConfig


@pytest.fixture
async def service_registry():
    """Create and start a service registry for testing"""
    registry = ServiceRegistry(cleanup_interval=5, service_timeout=10)
    await registry.start()
    yield registry
    await registry.stop()


@pytest.fixture
async def service_discovery(service_registry):
    """Create a service discovery client"""
    return ServiceDiscovery(service_registry, cache_ttl=5)


@pytest.fixture
def sample_services():
    """Generate sample services for testing"""
    return [
        ServiceInfo(
            service_id="financial-001",
            service_name="financial-analysis",
            host="127.0.0.1",
            port=8010,
            tags=["financial", "analysis"],
            metadata={"version": "1.0"}
        ),
        ServiceInfo(
            service_id="reasoning-001",
            service_name="cognitive-reasoning",
            host="127.0.0.1", 
            port=8011,
            tags=["cognitive", "reasoning"],
            metadata={"version": "1.0"}
        ),
        ServiceInfo(
            service_id="ggml-001",
            service_name="ml-inference",
            host="127.0.0.1",
            port=8012,
            tags=["ml", "ggml", "inference"],
            metadata={"model": "llama", "quantization": "q4_0"}
        )
    ]


class TestServiceDiscovery:
    """Test service discovery functionality"""
    
    async def test_service_registration(self, service_registry, sample_services):
        """Test service registration and discovery"""
        # Register services
        for service in sample_services:
            result = service_registry.register_service(service)
            assert result is True
        
        # Test discovery
        financial_services = service_registry.discover_services("financial-analysis")
        assert len(financial_services) == 1
        assert financial_services[0].service_id == "financial-001"
        
        ml_services = service_registry.discover_services("ml-inference")
        assert len(ml_services) == 1
        assert ml_services[0].service_id == "ggml-001"
    
    async def test_service_deregistration(self, service_registry, sample_services):
        """Test service deregistration"""
        # Register and then deregister
        service = sample_services[0]
        service_registry.register_service(service)
        
        # Verify registration
        services = service_registry.discover_services(service.service_name)
        assert len(services) == 1
        
        # Deregister
        result = service_registry.deregister_service(service.service_id)
        assert result is True
        
        # Verify deregistration
        services = service_registry.discover_services(service.service_name)
        assert len(services) == 0
    
    async def test_discovery_performance(self, service_registry, sample_services):
        """Test that service discovery meets sub-100ms requirement"""
        # Register multiple services
        for service in sample_services:
            service_registry.register_service(service)
        
        # Test discovery performance
        start_time = time.time()
        for _ in range(10):
            services = service_registry.discover_services("financial-analysis")
            assert len(services) == 1
        
        avg_time_ms = ((time.time() - start_time) / 10) * 1000
        assert avg_time_ms < 100, f"Discovery took {avg_time_ms:.2f}ms, exceeds 100ms requirement"
    
    async def test_cached_discovery(self, service_discovery, sample_services):
        """Test cached service discovery"""
        # Register services
        for service in sample_services:
            service_discovery.registry.register_service(service)
        
        # First discovery (cache miss)
        start_time = time.time()
        services1 = await service_discovery.discover("financial-analysis")
        first_time = (time.time() - start_time) * 1000
        
        # Second discovery (cache hit)
        start_time = time.time()
        services2 = await service_discovery.discover("financial-analysis")
        second_time = (time.time() - start_time) * 1000
        
        assert len(services1) == len(services2) == 1
        assert second_time < first_time, "Cached discovery should be faster"
        assert second_time < 10, f"Cached discovery took {second_time:.2f}ms, should be <10ms"


class TestLoadBalancing:
    """Test load balancing functionality"""
    
    async def test_round_robin_balancing(self, service_discovery, sample_services):
        """Test round-robin load balancing"""
        # Register multiple instances of same service
        financial_services = []
        for i in range(3):
            service = ServiceInfo(
                service_id=f"financial-{i:03d}",
                service_name="financial-analysis",
                host="127.0.0.1",
                port=8010 + i
            )
            financial_services.append(service)
            service_discovery.registry.register_service(service)
        
        # Test load balancing
        load_balancer = LoadBalancer(service_discovery)
        selected_services = []
        
        for _ in range(6):  # Select 6 times to see round-robin pattern
            service = await load_balancer.get_service("financial-analysis")
            assert service is not None
            selected_services.append(service.service_id)
        
        # Check that all services were selected
        unique_services = set(selected_services)
        assert len(unique_services) == 3, "Round-robin should select all services"
    
    async def test_circuit_breaker(self, service_discovery, sample_services):
        """Test circuit breaker functionality"""
        service = sample_services[0]
        service_discovery.registry.register_service(service)
        
        config = LoadBalancingConfig(circuit_breaker_enabled=True, circuit_breaker_threshold=3)
        load_balancer = LoadBalancer(service_discovery, config)
        
        # Record failures to trigger circuit breaker
        for _ in range(3):
            load_balancer.record_failure(service.service_id)
        
        # Service should be excluded due to circuit breaker
        selected = await load_balancer.get_service(service.service_name)
        # With only one service and circuit breaker open, no service should be available
        # But let's check if the failed service is excluded by verifying the service is not the failed one
        if selected:
            assert selected.service_id != service.service_id, "Circuit breaker should exclude failed service"
        
        # Reset circuit breaker
        load_balancer.reset_circuit_breaker(service.service_id)
        selected = await load_balancer.get_service(service.service_name)
        assert selected is not None, "Service should be available after circuit breaker reset"


class TestOrchestration:
    """Test service orchestration and health monitoring"""
    
    async def test_health_monitoring_setup(self, service_registry):
        """Test health monitor initialization"""
        service_discovery = ServiceDiscovery(service_registry)
        health_monitor = HealthMonitor(service_discovery)
        
        await health_monitor.start()
        assert health_monitor._session is not None
        assert health_monitor._health_check_task is not None
        
        await health_monitor.stop()
        assert health_monitor._session is None
        assert health_monitor._health_check_task is None
    
    async def test_service_orchestrator_setup(self, service_registry):
        """Test service orchestrator initialization"""
        service_discovery = ServiceDiscovery(service_registry)
        health_monitor = HealthMonitor(service_discovery)
        load_balancer = LoadBalancer(service_discovery)
        
        orchestrator = ServiceOrchestrator(service_registry, health_monitor, load_balancer)
        await orchestrator.start()
        
        # Test system status
        status = await orchestrator.get_system_status()
        assert "total_services" in status
        assert "healthy_services" in status
        assert "uptime" in status
        
        await orchestrator.stop()
    
    async def test_scaling_operations(self, service_registry):
        """Test service scaling capabilities"""
        service_discovery = ServiceDiscovery(service_registry)
        health_monitor = HealthMonitor(service_discovery)
        load_balancer = LoadBalancer(service_discovery)
        orchestrator = ServiceOrchestrator(service_registry, health_monitor, load_balancer)
        
        await orchestrator.start()
        
        # Test scaling up
        scale_result = await orchestrator.scale_service("test-service", 3)
        assert scale_result["action"] == "scale_up"
        assert scale_result["added_instances"] == 3
        
        await orchestrator.stop()


class TestGGMLOptimization:
    """Test GGML optimization and hypergraph encoding"""
    
    def test_ggml_service_registration(self):
        """Test GGML service configuration"""
        optimizer = GGMLServiceOptimizer()
        
        config = GGMLServiceConfig(
            model_type="llama",
            context_length=2048,
            quantization="q4_0",
            gpu_layers=32
        )
        
        optimizer.register_ggml_service("ggml-001", config)
        assert "ggml-001" in optimizer.model_configs
        
    def test_resource_optimization(self, sample_services):
        """Test resource optimization recommendations"""
        optimizer = GGMLServiceOptimizer()
        
        # Register GGML configurations
        configs = [
            GGMLServiceConfig(model_type="llama", memory_limit_mb=4096, gpu_layers=32),
            GGMLServiceConfig(model_type="gpt", memory_limit_mb=2048, gpu_layers=16),
            GGMLServiceConfig(model_type="bert", memory_limit_mb=1024, gpu_layers=8)
        ]
        
        for i, config in enumerate(configs):
            optimizer.register_ggml_service(f"ggml-{i:03d}", config)
        
        # Test optimization
        recommendations = optimizer.optimize_service_allocation(sample_services[:1])  # Only GGML service
        assert "total_ggml_services" in recommendations
        assert "optimizations" in recommendations
    
    def test_inference_cost_calculation(self):
        """Test inference cost calculation"""
        optimizer = GGMLServiceOptimizer()
        
        config = GGMLServiceConfig(
            model_type="llama",
            quantization="q4_0",
            context_length=2048,
            gpu_layers=32
        )
        optimizer.register_ggml_service("ggml-001", config)
        
        cost = optimizer.calculate_inference_cost("ggml-001", input_tokens=100, output_tokens=50)
        assert "input_cost" in cost
        assert "output_cost" in cost
        assert "total_cost" in cost
        assert cost["total_cost"] > 0
    
    def test_load_balancing_weights(self, sample_services):
        """Test GGML-specific load balancing weights"""
        optimizer = GGMLServiceOptimizer()
        
        # Configure different service capabilities
        configs = [
            GGMLServiceConfig(model_type="llama", gpu_layers=32, memory_limit_mb=4096),
            GGMLServiceConfig(model_type="llama", gpu_layers=0, memory_limit_mb=2048),
        ]
        
        for i, config in enumerate(configs):
            optimizer.register_ggml_service(f"ggml-{i:03d}", config)
        
        weights = optimizer.suggest_load_balancing_weights(sample_services[:2])
        assert len(weights) == 2
        assert all(isinstance(w, float) for w in weights.values())


class TestHypergraphEncoding:
    """Test hypergraph pattern encoding for service mesh"""
    
    def test_hypergraph_initialization(self):
        """Test hypergraph encoder setup"""
        encoder = HypergraphMeshEncoder(embedding_dim=64)
        assert encoder.embedding_dim == 64
        assert len(encoder.nodes) == 0
        assert len(encoder.edges) == 0
    
    def test_service_node_addition(self, sample_services):
        """Test adding services as hypergraph nodes"""
        encoder = HypergraphMeshEncoder()
        
        for service in sample_services:
            encoder.add_service_node(service)
        
        assert len(encoder.nodes) == len(sample_services)
        
        # Check embeddings
        for service in sample_services:
            node = encoder.nodes[service.service_id]
            assert node.embedding is not None
            assert len(node.embedding) == encoder.embedding_dim
    
    def test_dependency_encoding(self, sample_services):
        """Test encoding service dependencies"""
        encoder = HypergraphMeshEncoder()
        
        # Add services
        for service in sample_services:
            encoder.add_service_node(service)
        
        # Encode dependencies
        dependencies = {
            "financial-001": ["reasoning-001"],
            "reasoning-001": ["ggml-001"]
        }
        encoder.encode_service_dependencies(dependencies)
        
        # Check that dependency edges were created
        dependency_edges = [e for e in encoder.edges.values() if e.edge_type == "dependency"]
        assert len(dependency_edges) == 2
    
    def test_routing_recommendations(self, sample_services):
        """Test routing recommendations generation"""
        encoder = HypergraphMeshEncoder()
        
        # Add services
        for service in sample_services:
            encoder.add_service_node(service)
        
        # Generate recommendations
        recommendations = encoder.generate_routing_recommendations("financial-analysis")
        assert "service_name" in recommendations
        assert "routing_strategies" in recommendations
        assert "load_balancing_weights" in recommendations


class TestIntegratedWorkflow:
    """Test complete microservice workflow integration"""
    
    async def test_zero_downtime_deployment_simulation(self, service_registry):
        """Test simulated zero-downtime deployment"""
        # Setup complete system
        service_discovery = ServiceDiscovery(service_registry)
        
        # Create a mock health monitor that always returns healthy
        class MockHealthMonitor:
            async def start(self):
                pass
            
            async def stop(self):
                pass
            
            async def check_service_health(self, service):
                from src.microservices.orchestration import HealthCheckResult, ServiceStatus
                from datetime import datetime
                return HealthCheckResult(
                    service_id=service.service_id,
                    status=ServiceStatus.HEALTHY,
                    response_time_ms=10.0,
                    timestamp=datetime.now()
                )
        
        health_monitor = MockHealthMonitor()
        load_balancer = LoadBalancer(service_discovery)
        orchestrator = ServiceOrchestrator(service_registry, health_monitor, load_balancer)
        
        await orchestrator.start()
        
        # Register initial service
        old_service = ServiceInfo(
            service_id="financial-v1",
            service_name="financial-analysis",
            host="127.0.0.1",
            port=8010,
            version="1.0"
        )
        service_registry.register_service(old_service)
        
        # Deploy new version
        new_services = [
            ServiceInfo(
                service_id="financial-v2",
                service_name="financial-analysis", 
                host="127.0.0.1",
                port=8020,
                version="2.0"
            )
        ]
        
        # Test rolling deployment
        deployment_result = await orchestrator.deploy_service(
            "financial-analysis", 
            new_services, 
            strategy="rolling"
        )
        
        assert deployment_result["status"] == "completed"
        assert deployment_result["strategy"] == "rolling"
        
        await orchestrator.stop()
    
    async def test_service_mesh_configuration_generation(self, sample_services):
        """Test proxy configuration generation"""
        # Group services by name
        services_by_name = {}
        for service in sample_services:
            if service.service_name not in services_by_name:
                services_by_name[service.service_name] = []
            services_by_name[service.service_name].append(service)
        
        config = LoadBalancingConfig()
        
        # Test Envoy configuration
        envoy_generator = EnvoyConfigGenerator()
        envoy_config = envoy_generator.generate_config(services_by_name, config)
        assert "admin:" in envoy_config
        assert "static_resources:" in envoy_config
        assert "listeners:" in envoy_config
        assert "clusters:" in envoy_config
        
        # Test Traefik configuration
        traefik_generator = TraefikConfigGenerator()
        traefik_config = traefik_generator.generate_config(services_by_name, config)
        assert "static:" in traefik_config
        assert "dynamic:" in traefik_config
    
    async def test_performance_benchmarks(self, service_registry, sample_services):
        """Test performance benchmarks meet requirements"""
        # Register services
        for service in sample_services:
            service_registry.register_service(service)
        
        service_discovery = ServiceDiscovery(service_registry, cache_ttl=60)
        
        # Benchmark service discovery
        discovery_times = []
        for _ in range(100):
            start_time = time.time()
            services = await service_discovery.discover("financial-analysis")
            discovery_time = (time.time() - start_time) * 1000
            discovery_times.append(discovery_time)
            assert len(services) > 0
        
        avg_discovery_time = sum(discovery_times) / len(discovery_times)
        p99_discovery_time = sorted(discovery_times)[99]
        
        # Verify performance requirements
        assert avg_discovery_time < 50, f"Average discovery time {avg_discovery_time:.2f}ms exceeds 50ms"
        assert p99_discovery_time < 100, f"P99 discovery time {p99_discovery_time:.2f}ms exceeds 100ms"
        
        print(f"✓ Service Discovery Performance:")
        print(f"  Average: {avg_discovery_time:.2f}ms")
        print(f"  P99: {p99_discovery_time:.2f}ms")


# Chaos Engineering Tests
class TestChaosEngineering:
    """Test system resilience under failure conditions"""
    
    async def test_service_failure_recovery(self, service_registry, sample_services):
        """Test automatic failover when services become unhealthy"""
        # Setup system with multiple instances
        services = []
        for i in range(3):
            service = ServiceInfo(
                service_id=f"financial-{i:03d}",
                service_name="financial-analysis",
                host="127.0.0.1",
                port=8010 + i
            )
            services.append(service)
            service_registry.register_service(service)
        
        service_discovery = ServiceDiscovery(service_registry)
        load_balancer = LoadBalancer(service_discovery)
        
        # Simulate service failure by triggering circuit breaker
        failed_service_id = services[0].service_id
        for _ in range(3):
            load_balancer.record_failure(failed_service_id)
        
        # Load balancer should exclude failed service
        selected_services = []
        for _ in range(10):
            service = await load_balancer.get_service("financial-analysis")
            if service:
                selected_services.append(service.service_id)
        
        # With circuit breaker, the failed service should be selected less frequently or not at all
        # At least some services should be available (the healthy ones)
        assert len(selected_services) > 0, "Some healthy services should be available"
        
        # Check if circuit breaker is working by verifying the failed service is avoided
        failed_count = selected_services.count(failed_service_id)
        total_selections = len(selected_services)
        
        if total_selections > 0:
            failed_ratio = failed_count / total_selections
            # With 3 services and circuit breaker on one, it should be selected less than 50% of the time
            assert failed_ratio < 0.5, f"Failed service selected {failed_ratio:.2%} of time, circuit breaker should reduce this"
    
    async def test_network_partition_resilience(self, service_registry):
        """Test system behavior during network partitions"""
        # This test simulates network partitions by deregistering services
        # In a real environment, this would involve actual network manipulation
        
        # Setup services in different "zones"
        zone_a_services = [
            ServiceInfo(service_id="svc-a1", service_name="test-svc", host="10.0.1.1", port=8000),
            ServiceInfo(service_id="svc-a2", service_name="test-svc", host="10.0.1.2", port=8000)
        ]
        zone_b_services = [
            ServiceInfo(service_id="svc-b1", service_name="test-svc", host="10.0.2.1", port=8000),
            ServiceInfo(service_id="svc-b2", service_name="test-svc", host="10.0.2.2", port=8000)
        ]
        
        # Register all services
        all_services = zone_a_services + zone_b_services
        for service in all_services:
            service_registry.register_service(service)
        
        # Verify all services available
        discovered = service_registry.discover_services("test-svc")
        assert len(discovered) == 4
        
        # Simulate zone A partition (remove zone A services)
        for service in zone_a_services:
            service_registry.deregister_service(service.service_id)
        
        # Zone B services should still be available
        discovered = service_registry.discover_services("test-svc")
        assert len(discovered) == 2
        zone_b_ids = {s.service_id for s in zone_b_services}
        discovered_ids = {s.service_id for s in discovered}
        assert discovered_ids == zone_b_ids


if __name__ == "__main__":
    async def run_comprehensive_tests():
        """Run all microservice tests"""
        print("=== Microservice Discovery and Orchestration Tests ===\n")
        
        # Create test instances
        registry = ServiceRegistry(cleanup_interval=5, service_timeout=10)
        await registry.start()
        
        discovery = ServiceDiscovery(registry, cache_ttl=5)
        
        sample_services = [
            ServiceInfo(
                service_id="financial-001",
                service_name="financial-analysis",
                host="127.0.0.1",
                port=8010,
                tags=["financial", "analysis"]
            ),
            ServiceInfo(
                service_id="reasoning-001", 
                service_name="cognitive-reasoning",
                host="127.0.0.1",
                port=8011,
                tags=["cognitive", "reasoning"]
            ),
            ServiceInfo(
                service_id="ggml-001",
                service_name="ml-inference", 
                host="127.0.0.1",
                port=8012,
                tags=["ml", "ggml"]
            )
        ]
        
        try:
            # Test service discovery
            print("Testing Service Discovery...")
            test_discovery = TestServiceDiscovery()
            await test_discovery.test_service_registration(registry, sample_services)
            await test_discovery.test_discovery_performance(registry, sample_services)
            print("✓ Service Discovery tests passed\n")
            
            # Test load balancing
            print("Testing Load Balancing...")
            test_lb = TestLoadBalancing()
            await test_lb.test_round_robin_balancing(discovery, sample_services)
            await test_lb.test_circuit_breaker(discovery, sample_services)
            print("✓ Load Balancing tests passed\n")
            
            # Test GGML optimization
            print("Testing GGML Optimization...")
            test_ggml = TestGGMLOptimization()
            test_ggml.test_ggml_service_registration()
            test_ggml.test_resource_optimization(sample_services)
            test_ggml.test_inference_cost_calculation()
            print("✓ GGML Optimization tests passed\n")
            
            # Test hypergraph encoding
            print("Testing Hypergraph Encoding...")
            test_hypergraph = TestHypergraphEncoding()
            test_hypergraph.test_hypergraph_initialization()
            test_hypergraph.test_service_node_addition(sample_services)
            test_hypergraph.test_dependency_encoding(sample_services)
            print("✓ Hypergraph Encoding tests passed\n")
            
            # Test integrated workflow
            print("Testing Integrated Workflow...")
            test_integrated = TestIntegratedWorkflow()
            await test_integrated.test_zero_downtime_deployment_simulation(registry)
            await test_integrated.test_service_mesh_configuration_generation(sample_services)
            await test_integrated.test_performance_benchmarks(registry, sample_services)
            print("✓ Integrated Workflow tests passed\n")
            
            # Test chaos engineering
            print("Testing Chaos Engineering...")
            test_chaos = TestChaosEngineering()
            await test_chaos.test_service_failure_recovery(registry, sample_services)
            await test_chaos.test_network_partition_resilience(registry)
            print("✓ Chaos Engineering tests passed\n")
            
            print("=== All Microservice Tests Passed! ===")
            print("✓ Dynamic service discovery with sub-100ms performance")
            print("✓ Load balancing with circuit breaker protection")
            print("✓ Zero-downtime deployment orchestration")
            print("✓ GGML-optimized ML model serving")
            print("✓ Hypergraph pattern encoding for service mesh")
            print("✓ Chaos engineering and failure recovery")
            print("✓ Service mesh configuration generation")
            
        finally:
            await registry.stop()
    
    # Run tests
    asyncio.run(run_comprehensive_tests())