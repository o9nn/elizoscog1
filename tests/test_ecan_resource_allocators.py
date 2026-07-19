#!/usr/bin/env python3
"""
ECAN Resource Allocator Test Suite

Tests for Phase 2 ECAN-inspired resource allocation implementation.
Validates performance against success criteria:
- Sub-50ms attention allocation response time
- 95% resource utilization efficiency
- Dynamic load balancing across 100+ agents
- Stable economic attention equilibrium
- Real-time adaptation to changing priorities
"""

import asyncio
import sys
import os
import time
import statistics
from typing import List, Dict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bridges.attention_bridge import AttentionBridge, ECANResourceAllocator, AttentionValue, ResourceAgent


class ECANTestSuite:
    """Comprehensive test suite for ECAN resource allocators"""
    
    def __init__(self):
        self.test_results = {
            "performance_tests": [],
            "efficiency_tests": [],
            "load_balancing_tests": [],
            "equilibrium_tests": [],
            "adaptation_tests": []
        }
    
    async def run_all_tests(self):
        """Run complete ECAN test suite"""
        print("=== ECAN Resource Allocator Test Suite ===\n")
        
        # Test 1: Performance Tests (Sub-50ms requirement)
        await self.test_allocation_performance()
        
        # Test 2: Resource Utilization Efficiency (95% requirement)  
        await self.test_resource_efficiency()
        
        # Test 3: Load Balancing (100+ agents requirement)
        await self.test_load_balancing()
        
        # Test 4: Economic Equilibrium Stability
        await self.test_economic_equilibrium()
        
        # Test 5: Real-time Adaptation
        await self.test_real_time_adaptation()
        
        # Test 6: Integration Tests
        await self.test_bridge_integration()
        
        # Generate comprehensive report
        self.generate_test_report()
    
    async def test_allocation_performance(self):
        """Test allocation performance against sub-50ms target"""
        print("Testing Allocation Performance (Target: <50ms)...")
        
        allocator = ECANResourceAllocator()
        performance_results = []
        
        # Test with varying numbers of agents
        agent_counts = [10, 25, 50, 100, 150]
        
        for count in agent_counts:
            # Generate test requests
            requests = []
            for i in range(count):
                requests.append({
                    "agent_id": f"perf_agent_{i}",
                    "importance": 0.5 + (i % 10) * 0.05,
                    "urgency": 0.3 + (i % 5) * 0.1
                })
            
            # Measure allocation time
            start_time = time.perf_counter()
            result = await allocator.allocate_resources(requests)
            end_time = time.perf_counter()
            
            allocation_time_ms = (end_time - start_time) * 1000
            performance_results.append({
                "agent_count": count,
                "allocation_time_ms": allocation_time_ms,
                "meets_target": allocation_time_ms < 50
            })
            
            print(f"  {count} agents: {allocation_time_ms:.2f}ms {'✓' if allocation_time_ms < 50 else '✗'}")
        
        # Calculate statistics
        all_times = [r["allocation_time_ms"] for r in performance_results]
        avg_time = statistics.mean(all_times)
        max_time = max(all_times)
        success_rate = sum(1 for r in performance_results if r["meets_target"]) / len(performance_results)
        
        self.test_results["performance_tests"] = {
            "results": performance_results,
            "avg_time_ms": avg_time,
            "max_time_ms": max_time,
            "success_rate": success_rate,
            "passes_requirement": success_rate >= 0.8  # 80% of tests should meet target
        }
        
        print(f"  Average: {avg_time:.2f}ms, Max: {max_time:.2f}ms, Success Rate: {success_rate*100:.1f}%")
        print(f"  Performance Test: {'✓ PASS' if success_rate >= 0.8 else '✗ FAIL'}\n")
    
    async def test_resource_efficiency(self):
        """Test resource utilization efficiency against 95% target"""
        print("Testing Resource Utilization Efficiency (Target: >95%)...")
        
        allocator = ECANResourceAllocator(max_agents=50)
        efficiency_results = []
        
        # Test scenarios with different load patterns
        scenarios = [
            {"name": "uniform_load", "agents": 30, "importance_range": (0.5, 0.8)},
            {"name": "heavy_load", "agents": 45, "importance_range": (0.7, 1.0)},
            {"name": "light_load", "agents": 15, "importance_range": (0.2, 0.6)},
            {"name": "mixed_load", "agents": 40, "importance_range": (0.1, 1.0)}
        ]
        
        for scenario in scenarios:
            requests = []
            for i in range(scenario["agents"]):
                importance = scenario["importance_range"][0] + \
                           (scenario["importance_range"][1] - scenario["importance_range"][0]) * \
                           (i / scenario["agents"])
                requests.append({
                    "agent_id": f"eff_agent_{scenario['name']}_{i}",
                    "importance": importance,
                    "urgency": 0.5
                })
            
            result = await allocator.allocate_resources(requests)
            efficiency = result["efficiency_metrics"]["utilization_rate"]
            
            efficiency_results.append({
                "scenario": scenario["name"],
                "efficiency": efficiency,
                "meets_target": efficiency >= 0.95
            })
            
            print(f"  {scenario['name']}: {efficiency*100:.1f}% {'✓' if efficiency >= 0.95 else '✗'}")
        
        avg_efficiency = statistics.mean([r["efficiency"] for r in efficiency_results])
        success_rate = sum(1 for r in efficiency_results if r["meets_target"]) / len(efficiency_results)
        
        self.test_results["efficiency_tests"] = {
            "results": efficiency_results,
            "avg_efficiency": avg_efficiency,
            "success_rate": success_rate,
            "passes_requirement": avg_efficiency >= 0.95
        }
        
        print(f"  Average Efficiency: {avg_efficiency*100:.1f}%, Success Rate: {success_rate*100:.1f}%")
        print(f"  Efficiency Test: {'✓ PASS' if avg_efficiency >= 0.95 else '✗ FAIL'}\n")
    
    async def test_load_balancing(self):
        """Test load balancing across 100+ agents"""
        print("Testing Load Balancing (Target: 100+ agents)...")
        
        allocator = ECANResourceAllocator(max_agents=120)
        
        # Create 110 agents with varied importance
        requests = []
        for i in range(110):
            requests.append({
                "agent_id": f"balance_agent_{i}",
                "importance": 0.3 + (i % 20) * 0.035,  # Varied importance
                "urgency": 0.2 + (i % 15) * 0.04
            })
        
        result = await allocator.allocate_resources(requests)
        
        # Analyze load distribution
        allocations = result["allocation_plan"]["resource_distribution"]
        allocation_values = list(allocations.values())
        
        # Calculate distribution metrics
        min_allocation = min(allocation_values)
        max_allocation = max(allocation_values)
        std_deviation = statistics.stdev(allocation_values)
        balance_ratio = min_allocation / max_allocation if max_allocation > 0 else 0
        
        # Good load balancing should have reasonable distribution
        good_balance = std_deviation < 0.1 and balance_ratio > 0.1
        
        self.test_results["load_balancing_tests"] = {
            "agent_count": len(allocations),
            "min_allocation": min_allocation,
            "max_allocation": max_allocation,
            "std_deviation": std_deviation,
            "balance_ratio": balance_ratio,
            "good_balance": good_balance,
            "passes_requirement": len(allocations) >= 100 and good_balance
        }
        
        print(f"  Agents processed: {len(allocations)}")
        print(f"  Allocation range: {min_allocation:.3f} - {max_allocation:.3f}")
        print(f"  Standard deviation: {std_deviation:.3f}")
        print(f"  Balance ratio: {balance_ratio:.3f}")
        print(f"  Load Balancing Test: {'✓ PASS' if len(allocations) >= 100 and good_balance else '✗ FAIL'}\n")
    
    async def test_economic_equilibrium(self):
        """Test economic attention equilibrium stability"""
        print("Testing Economic Equilibrium Stability...")
        
        allocator = ECANResourceAllocator(attention_bank_funds=1000.0)
        initial_bank = allocator.attention_bank
        
        equilibrium_data = []
        
        # Run multiple allocation cycles to test equilibrium
        for cycle in range(10):
            requests = []
            for i in range(20):
                requests.append({
                    "agent_id": f"equil_agent_{cycle}_{i}",
                    "importance": 0.4 + (i % 10) * 0.06,
                    "urgency": 0.3 + (cycle % 5) * 0.1
                })
            
            result = await allocator.allocate_resources(requests)
            bank_change = abs(allocator.attention_bank - initial_bank)
            
            equilibrium_data.append({
                "cycle": cycle,
                "bank_funds": allocator.attention_bank,
                "bank_change": bank_change,
                "active_agents": len(allocator.active_agents)
            })
            
            # Small delay to simulate real usage
            await asyncio.sleep(0.01)
        
        # Analyze equilibrium stability
        final_bank = allocator.attention_bank
        total_change = abs(final_bank - initial_bank)
        max_change = max(eq["bank_change"] for eq in equilibrium_data)
        
        # Stable equilibrium should have bounded changes
        stable_equilibrium = total_change < 300 and max_change < 200
        
        self.test_results["equilibrium_tests"] = {
            "initial_bank": initial_bank,
            "final_bank": final_bank,
            "total_change": total_change,
            "max_change": max_change,
            "stable_equilibrium": stable_equilibrium,
            "equilibrium_data": equilibrium_data,
            "passes_requirement": stable_equilibrium
        }
        
        print(f"  Initial bank: {initial_bank:.1f}")
        print(f"  Final bank: {final_bank:.1f}")
        print(f"  Total change: {total_change:.1f}")
        print(f"  Equilibrium Test: {'✓ PASS' if stable_equilibrium else '✗ FAIL'}\n")
    
    async def test_real_time_adaptation(self):
        """Test real-time adaptation to changing priorities"""
        print("Testing Real-time Adaptation...")
        
        bridge = AttentionBridge()
        await bridge.initialize()
        
        adaptation_results = []
        
        # Simulate changing priorities
        priority_changes = [
            {"urgency_multiplier": 1.0, "expected_change": "baseline"},
            {"urgency_multiplier": 2.0, "expected_change": "increased_urgency"},
            {"urgency_multiplier": 0.5, "expected_change": "decreased_urgency"},
            {"urgency_multiplier": 1.5, "expected_change": "moderate_urgency"}
        ]
        
        for i, change in enumerate(priority_changes):
            requests = []
            for j in range(15):
                requests.append({
                    "agent_id": f"adapt_agent_{i}_{j}",
                    "importance": 0.6,
                    "urgency": 0.4 * change["urgency_multiplier"]
                })
            
            result = await bridge.allocate_attention_resources(requests)
            
            adaptation_results.append({
                "change_scenario": change["expected_change"],
                "allocation_time": result["allocation_time_ms"],
                "efficiency": result["efficiency_metrics"]["utilization_rate"],
                "adapted_properly": result["allocation_time_ms"] < 50
            })
        
        # Check adaptation capability
        adaptation_success = all(r["adapted_properly"] for r in adaptation_results)
        
        self.test_results["adaptation_tests"] = {
            "results": adaptation_results,
            "adaptation_success": adaptation_success,
            "passes_requirement": adaptation_success
        }
        
        print(f"  Adaptation scenarios tested: {len(adaptation_results)}")
        for result in adaptation_results:
            print(f"    {result['change_scenario']}: {result['allocation_time']:.1f}ms {'✓' if result['adapted_properly'] else '✗'}")
        print(f"  Adaptation Test: {'✓ PASS' if adaptation_success else '✗ FAIL'}\n")
    
    async def test_bridge_integration(self):
        """Test full bridge integration functionality"""
        print("Testing Bridge Integration...")
        
        bridge = AttentionBridge()
        init_success = await bridge.initialize()
        
        if not init_success:
            print("  ✗ Bridge initialization failed")
            return
        
        # Test system status
        status = await bridge.get_system_status()
        
        # Test allocation
        test_requests = [
            {"agent_id": "integration_agent_1", "importance": 0.8, "urgency": 0.6},
            {"agent_id": "integration_agent_2", "importance": 0.7, "urgency": 0.4},
            {"agent_id": "integration_agent_3", "importance": 0.9, "urgency": 0.8}
        ]
        
        allocation_result = await bridge.allocate_attention_resources(test_requests)
        
        # Verify integration
        integration_success = (
            status["status"] == "active" and
            "allocation_plan" in allocation_result and
            allocation_result["allocation_time_ms"] > 0
        )
        
        print(f"  Bridge Status: {status['status']}")
        print(f"  Allocation Time: {allocation_result['allocation_time_ms']:.2f}ms")
        print(f"  Integration Test: {'✓ PASS' if integration_success else '✗ FAIL'}\n")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("=== ECAN Test Results Summary ===\n")
        
        # Count passes and fails
        total_tests = 0
        passed_tests = 0
        
        for test_category, results in self.test_results.items():
            if isinstance(results, dict) and "passes_requirement" in results:
                total_tests += 1
                if results["passes_requirement"]:
                    passed_tests += 1
        
        print(f"Tests Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        print()
        
        # Detailed results
        performance = self.test_results.get("performance_tests", {})
        if performance:
            print(f"Performance (Sub-50ms): {'✓ PASS' if performance.get('passes_requirement') else '✗ FAIL'}")
            print(f"  Average time: {performance.get('avg_time_ms', 0):.2f}ms")
        
        efficiency = self.test_results.get("efficiency_tests", {})
        if efficiency:
            print(f"Efficiency (>95%): {'✓ PASS' if efficiency.get('passes_requirement') else '✗ FAIL'}")
            print(f"  Average efficiency: {efficiency.get('avg_efficiency', 0)*100:.1f}%")
        
        load_balancing = self.test_results.get("load_balancing_tests", {})
        if load_balancing:
            print(f"Load Balancing (100+ agents): {'✓ PASS' if load_balancing.get('passes_requirement') else '✗ FAIL'}")
            print(f"  Agents handled: {load_balancing.get('agent_count', 0)}")
        
        equilibrium = self.test_results.get("equilibrium_tests", {})
        if equilibrium:
            print(f"Economic Equilibrium: {'✓ PASS' if equilibrium.get('passes_requirement') else '✗ FAIL'}")
            print(f"  Bank stability: {equilibrium.get('total_change', 0):.1f} total change")
        
        adaptation = self.test_results.get("adaptation_tests", {})
        if adaptation:
            print(f"Real-time Adaptation: {'✓ PASS' if adaptation.get('passes_requirement') else '✗ FAIL'}")
        
        print()
        
        # Success criteria evaluation
        print("=== Success Criteria Evaluation ===")
        success_criteria = {
            "Sub-50ms response time": performance.get("passes_requirement", False),
            "95% resource efficiency": efficiency.get("passes_requirement", False), 
            "100+ agent load balancing": load_balancing.get("passes_requirement", False),
            "Stable economic equilibrium": equilibrium.get("passes_requirement", False),
            "Real-time adaptation": adaptation.get("passes_requirement", False)
        }
        
        for criterion, passed in success_criteria.items():
            print(f"  {criterion}: {'✓ PASS' if passed else '✗ FAIL'}")
        
        overall_success = all(success_criteria.values())
        print(f"\n🎯 Overall ECAN Implementation: {'✅ SUCCESS' if overall_success else '❌ NEEDS IMPROVEMENT'}")
        
        if overall_success:
            print("\n🚀 ECAN Resource Allocators successfully implemented!")
            print("   All success criteria met - ready for production deployment.")
        else:
            print("\n⚠️  Some success criteria not met - optimization needed.")


async def main():
    """Run ECAN test suite"""
    test_suite = ECANTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())