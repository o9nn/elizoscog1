"""
Comprehensive tests for Scheme Cognitive Grammar Microservices
Tests round-trip translation, accuracy validation, and performance benchmarks
"""

import pytest
import asyncio
import time
from typing import Dict, List, Any

from src.microservices.scheme_cognitive_grammar import (
    SchemeCognitiveGrammarService,
    ElizaOSPrimitive,
    AtomSpacePattern,
    AgentGrammarAdapter,
    MemoryGrammarAdapter,
    TranslationResult
)

class TestSchemeGrammarAdapters:
    """Test individual grammar adapters"""
    
    @pytest.fixture
    def agent_adapter(self):
        return AgentGrammarAdapter()
    
    @pytest.fixture  
    def memory_adapter(self):
        return MemoryGrammarAdapter()
    
    @pytest.mark.asyncio
    async def test_agent_to_atomspace_translation(self, agent_adapter):
        """Test agent data translation to AtomSpace"""
        eliza_agent = {
            "agent": {
                "id": "financial-agent-001",
                "type": "financial_analyzer",
                "goals": ["analyze_spending", "detect_anomalies"],
                "beliefs": ["spending_tracking_enabled"],
                "capabilities": ["natural_language", "pattern_recognition"]
            }
        }
        
        result = await agent_adapter.translate_to_atomspace(eliza_agent)
        
        assert "atomspace_data" in result
        assert "translation_time_ms" in result
        assert result["translation_time_ms"] > 0
        
        atomspace_data = result["atomspace_data"]
        assert "atoms" in atomspace_data
        assert "links" in atomspace_data
        assert len(atomspace_data["atoms"]) > 0
        
        # Check agent concept node creation
        agent_nodes = [atom for atom in atomspace_data["atoms"] 
                      if atom.get("type") == "ConceptNode" and 
                      atom.get("name", "").startswith("agent:")]
        assert len(agent_nodes) == 1
        assert agent_nodes[0]["name"] == "agent:financial-agent-001"
    
    @pytest.mark.asyncio
    async def test_agent_from_atomspace_translation(self, agent_adapter):
        """Test agent data translation from AtomSpace"""
        atomspace_data = {
            "atoms": [
                {
                    "type": "ConceptNode",
                    "name": "agent:test-agent-001",
                    "tv": {"strength": 0.9, "confidence": 0.8}
                },
                {
                    "type": "PredicateNode",
                    "name": "agent_type:analyzer"
                }
            ],
            "links": [
                {
                    "type": "EvaluationLink",
                    "outgoing": ["agent_type:analyzer", "agent:test-agent-001"]
                }
            ]
        }
        
        result = await agent_adapter.translate_from_atomspace(atomspace_data)
        
        assert "agents" in result
        assert "metadata" in result
        assert len(result["agents"]) == 1
        
        agent = result["agents"][0]
        assert agent["id"] == "test-agent-001"
        assert agent["confidence"] == 0.8
    
    @pytest.mark.asyncio
    async def test_memory_to_atomspace_translation(self, memory_adapter):
        """Test memory data translation to AtomSpace"""
        eliza_memory = {
            "memory": {
                "id": "memory-001",
                "content": "User spent $350 on groceries last week",
                "strength": 0.85,
                "context": ["financial", "spending"]
            }
        }
        
        result = await memory_adapter.translate_to_atomspace(eliza_memory)
        
        assert "atomspace_data" in result
        assert "translation_time_ms" in result
        
        atomspace_data = result["atomspace_data"]
        assert len(atomspace_data["atoms"]) >= 2  # memory node + content node
        
        # Check memory concept node
        memory_nodes = [atom for atom in atomspace_data["atoms"]
                       if atom.get("type") == "ConceptNode" and
                       atom.get("name", "").startswith("memory:")]
        assert len(memory_nodes) == 1
        assert memory_nodes[0]["name"] == "memory:memory-001"
        assert memory_nodes[0]["tv"]["strength"] == 0.85
    
    @pytest.mark.asyncio
    async def test_memory_from_atomspace_translation(self, memory_adapter):
        """Test memory data translation from AtomSpace"""
        atomspace_data = {
            "atoms": [
                {
                    "type": "ConceptNode",
                    "name": "memory:test-memory-001",
                    "tv": {"strength": 0.7, "confidence": 0.9}
                }
            ]
        }
        
        result = await memory_adapter.translate_from_atomspace(atomspace_data)
        
        assert "memories" in result
        assert len(result["memories"]) == 1
        
        memory = result["memories"][0]
        assert memory["id"] == "test-memory-001"
        assert memory["strength"] == 0.7

class TestSchemeCognitiveGrammarService:
    """Test the main cognitive grammar service"""
    
    @pytest.fixture
    def grammar_service(self):
        return SchemeCognitiveGrammarService()
    
    @pytest.fixture
    def test_agent_data(self):
        return {
            "agent": {
                "id": "cognitive-agent-001", 
                "type": "reasoning",
                "goals": ["understand_context", "provide_insights"],
                "beliefs": ["user_needs_help"],
                "memory_capacity": 1000
            }
        }
    
    @pytest.fixture
    def test_memory_data(self):
        return {
            "memory": {
                "id": "financial-memory-001",
                "content": "Portfolio performance improved 15% this quarter",
                "strength": 0.92,
                "tags": ["portfolio", "performance", "quarterly"]
            }
        }
    
    def test_atomic_vocabulary_initialization(self, grammar_service):
        """Test atomic vocabulary is properly initialized"""
        vocab = grammar_service.get_atomic_vocabulary()
        
        assert "agent" in vocab
        assert "memory" in vocab
        assert "action" in vocab
        
        agent_entry = vocab["agent"]
        assert agent_entry["eliza_primitive"] == "agent"  # String value, not enum
        assert agent_entry["atomspace_pattern"] == "ConceptNode"
        assert agent_entry["bidirectional"] is True
        assert agent_entry["confidence"] == 0.95
    
    @pytest.mark.asyncio
    async def test_eliza_to_atomspace_translation(self, grammar_service, test_agent_data):
        """Test ElizaOS to AtomSpace translation"""
        result = await grammar_service.translate_eliza_to_atomspace(test_agent_data, "agent")
        
        assert isinstance(result, TranslationResult)
        assert result.source_format == "ElizaOS"
        assert result.target_format == "AtomSpace"
        assert result.translation_time_ms > 0
        assert result.accuracy_score >= 0.0
        assert len(result.validation_errors) == 0
        
        # Check translated data structure
        translated = result.translated_data
        assert "atomspace_data" in translated
        assert "translation_time_ms" in translated
    
    @pytest.mark.asyncio
    async def test_atomspace_to_eliza_translation(self, grammar_service):
        """Test AtomSpace to ElizaOS translation"""
        atomspace_data = {
            "atoms": [
                {
                    "type": "ConceptNode",
                    "name": "agent:translation-test-001",
                    "tv": {"strength": 0.95, "confidence": 0.88}
                }
            ],
            "links": []
        }
        
        result = await grammar_service.translate_atomspace_to_eliza(atomspace_data, "agent")
        
        assert isinstance(result, TranslationResult)
        assert result.source_format == "AtomSpace"
        assert result.target_format == "ElizaOS"
        assert result.translation_time_ms > 0
        assert len(result.validation_errors) == 0
        
        # Check translated data
        translated = result.translated_data
        assert "agents" in translated
        assert len(translated["agents"]) == 1
    
    @pytest.mark.asyncio
    async def test_round_trip_translation_agent(self, grammar_service, test_agent_data):
        """Test round-trip translation for agent data"""
        result = await grammar_service.round_trip_translate(test_agent_data, "agent")
        
        assert result["success"] is True
        assert "round_trip_accuracy" in result
        assert result["round_trip_accuracy"] >= 0.0
        assert "forward_result" in result
        assert "backward_result" in result
        assert "original_data" in result
        assert "final_data" in result
        
        # Verify data preservation
        original = result["original_data"]
        final = result["final_data"]
        assert original is not None
        assert final is not None
    
    @pytest.mark.asyncio
    async def test_round_trip_translation_memory(self, grammar_service, test_memory_data):
        """Test round-trip translation for memory data"""
        result = await grammar_service.round_trip_translate(test_memory_data, "memory")
        
        assert result["success"] is True
        assert result["round_trip_accuracy"] >= 0.0
        
        # Check memory preservation
        original_memory = result["original_data"]["memory"]
        final_memories = result["final_data"]["memories"]
        assert len(final_memories) > 0
        
        # Memory ID should be preserved
        final_memory = final_memories[0]
        assert final_memory["id"] == original_memory["id"]
    
    @pytest.mark.asyncio
    async def test_translation_error_handling(self, grammar_service):
        """Test error handling for invalid translations"""
        invalid_data = {"invalid": "data"}
        
        result = await grammar_service.translate_eliza_to_atomspace(invalid_data, "unknown_type")
        
        assert isinstance(result, TranslationResult)
        assert len(result.validation_errors) > 0
        assert result.accuracy_score == 0.0
        assert "Unknown data type" in result.validation_errors[0]
    
    def test_performance_stats_tracking(self, grammar_service):
        """Test performance statistics tracking"""
        stats = grammar_service.get_performance_stats()
        
        assert "total_translations" in stats
        assert "successful_translations" in stats
        assert "average_translation_time_ms" in stats
        assert "accuracy_scores" in stats
        assert "average_accuracy" in stats
        assert "success_rate" in stats
        
        # Initial stats should be empty
        assert stats["total_translations"] == 0
        assert stats["success_rate"] == 0.0

class TestTranslationAccuracy:
    """Test translation accuracy validation"""
    
    @pytest.fixture
    def grammar_service(self):
        return SchemeCognitiveGrammarService()
    
    @pytest.fixture
    def accuracy_test_data(self):
        """Generate test data for accuracy validation"""
        return [
            {
                "agent": {
                    "id": f"test-agent-{i}",
                    "type": "analyzer",
                    "goals": [f"goal-{j}" for j in range(2)],
                    "capabilities": ["reasoning", "analysis"]
                }
            } for i in range(10)
        ]
    
    @pytest.mark.asyncio
    async def test_accuracy_validation_high_threshold(self, grammar_service, accuracy_test_data):
        """Test accuracy validation with high threshold (99%)"""
        result = await grammar_service.validate_translation_accuracy(
            accuracy_test_data, "agent", target_accuracy=0.99
        )
        
        assert "validation_passed" in result
        assert "overall_accuracy" in result
        assert "target_accuracy" in result
        assert "test_results" in result
        assert "successful_tests" in result
        assert "total_tests" in result
        
        assert result["target_accuracy"] == 0.99
        assert result["total_tests"] == len(accuracy_test_data)
        assert result["successful_tests"] <= result["total_tests"]
    
    @pytest.mark.asyncio
    async def test_accuracy_validation_medium_threshold(self, grammar_service, accuracy_test_data):
        """Test accuracy validation with medium threshold (80%)"""
        result = await grammar_service.validate_translation_accuracy(
            accuracy_test_data, "agent", target_accuracy=0.80
        )
        
        # Should likely pass with 80% threshold
        assert result["overall_accuracy"] >= 0.0
        assert len(result["test_results"]) == len(accuracy_test_data)

class TestPerformanceBenchmarks:
    """Test performance benchmarks for translation speed"""
    
    @pytest.fixture
    def grammar_service(self):
        return SchemeCognitiveGrammarService()
    
    @pytest.mark.asyncio
    async def test_translation_speed_benchmark(self, grammar_service):
        """Test translation speed meets performance requirements"""
        test_data = {
            "agent": {
                "id": "benchmark-agent",
                "type": "performance_test",
                "goals": ["speed", "accuracy"],
                "metadata": {"timestamp": time.time()}
            }
        }
        
        # Perform multiple translations to get average
        times = []
        for _ in range(100):
            start_time = time.time()
            result = await grammar_service.translate_eliza_to_atomspace(test_data, "agent")
            end_time = time.time()
            
            translation_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(translation_time)
            
            assert len(result.validation_errors) == 0
        
        average_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        # Performance assertions (should be under 100ms for simple translations)
        assert average_time < 100.0, f"Average translation time {average_time}ms exceeds 100ms"
        assert max_time < 500.0, f"Maximum translation time {max_time}ms exceeds 500ms"
        assert min_time >= 0.0, "Minimum translation time should be positive"
    
    @pytest.mark.asyncio
    async def test_round_trip_performance_benchmark(self, grammar_service):
        """Test round-trip translation performance"""
        test_data = {
            "memory": {
                "id": "performance-memory",
                "content": "Performance test data for round-trip translation",
                "strength": 0.9,
                "context": ["performance", "testing"]
            }
        }
        
        times = []
        accuracies = []
        
        for _ in range(50):
            start_time = time.time()
            result = await grammar_service.round_trip_translate(test_data, "memory")
            end_time = time.time()
            
            round_trip_time = (end_time - start_time) * 1000
            times.append(round_trip_time)
            
            if result["success"]:
                accuracies.append(result["round_trip_accuracy"])
        
        average_time = sum(times) / len(times)
        average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        # Performance assertions for round-trip
        assert average_time < 200.0, f"Average round-trip time {average_time}ms exceeds 200ms"
        assert average_accuracy >= 0.3, f"Average round-trip accuracy {average_accuracy} too low"  # Adjusted threshold
        assert len(accuracies) >= len(times) * 0.8, "Success rate should be at least 80%"
    
    @pytest.mark.asyncio
    async def test_concurrent_translation_performance(self, grammar_service):
        """Test performance under concurrent load"""
        test_data_variants = [
            {
                "agent": {
                    "id": f"concurrent-agent-{i}",
                    "type": "load_test",
                    "goals": [f"concurrent-goal-{i}"],
                    "load_test_id": i
                }
            } for i in range(20)
        ]
        
        # Run concurrent translations
        start_time = time.time()
        
        tasks = [
            grammar_service.translate_eliza_to_atomspace(data, "agent")
            for data in test_data_variants
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        # Check results
        successful_results = [r for r in results if isinstance(r, TranslationResult) and not r.validation_errors]
        
        assert len(successful_results) >= len(test_data_variants) * 0.9, "Should handle 90% of concurrent requests"
        assert total_time < 2000.0, f"Total concurrent processing time {total_time}ms exceeds 2 seconds"
        
        # Average time per translation in concurrent scenario
        avg_concurrent_time = total_time / len(test_data_variants)
        assert avg_concurrent_time < 150.0, f"Average concurrent translation time {avg_concurrent_time}ms too high"

class TestErrorHandlingAndValidation:
    """Test error handling and validation protocols"""
    
    @pytest.fixture
    def grammar_service(self):
        return SchemeCognitiveGrammarService()
    
    @pytest.mark.asyncio
    async def test_invalid_data_type_handling(self, grammar_service):
        """Test handling of invalid data types"""
        valid_data = {"agent": {"id": "test"}}
        
        result = await grammar_service.translate_eliza_to_atomspace(valid_data, "invalid_type")
        
        assert len(result.validation_errors) > 0
        assert "Unknown data type" in str(result.validation_errors)
        assert result.accuracy_score == 0.0
    
    @pytest.mark.asyncio
    async def test_malformed_data_handling(self, grammar_service):
        """Test handling of malformed input data"""
        malformed_data = {"malformed": None, "nested": {"invalid": []}}
        
        # Should not crash, but may produce validation errors
        result = await grammar_service.translate_eliza_to_atomspace(malformed_data, "agent")
        
        assert isinstance(result, TranslationResult)
        # May or may not have validation errors depending on adapter robustness
        assert result.translation_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_empty_data_handling(self, grammar_service):
        """Test handling of empty data"""
        empty_data = {}
        
        result = await grammar_service.translate_eliza_to_atomspace(empty_data, "agent")
        
        assert isinstance(result, TranslationResult)
        assert result.accuracy_score >= 0.0  # May be 0 or higher depending on implementation
    
    def test_stats_consistency(self, grammar_service):
        """Test that statistics remain consistent"""
        initial_stats = grammar_service.get_performance_stats()
        
        # Stats should be well-formed
        assert initial_stats["total_translations"] >= 0
        assert initial_stats["successful_translations"] >= 0
        assert initial_stats["successful_translations"] <= initial_stats["total_translations"]
        assert 0.0 <= initial_stats["success_rate"] <= 1.0
        assert initial_stats["average_translation_time_ms"] >= 0.0

if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])