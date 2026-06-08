#!/usr/bin/env python3
"""
Neural-Symbolic Bridge for AtomSpace Integration
Phase 3 Implementation: GGML Kernels ↔ AtomSpace Bridge

Provides seamless integration between GGML symbolic kernels and OpenCog AtomSpace
for neural-symbolic computation and inference.
"""

import asyncio
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from abc import ABC, abstractmethod

from .ggml_symbolic_kernels import (
    SymbolicTensor, GGMLSymbolicKernelManager, SymbolicOperation,
    KernelArchitecture, get_kernel_manager
)
from .atomspace_bindings import AtomSpaceCore

logger = logging.getLogger(__name__)


class AtomType(Enum):
    """Extended atom types for neural-symbolic bridge"""
    CONCEPT_NODE = "ConceptNode"
    PREDICATE_NODE = "PredicateNode"
    NUMBER_NODE = "NumberNode"
    TENSOR_NODE = "TensorNode"
    KERNEL_NODE = "KernelNode"
    SYMBOL_NODE = "SymbolNode"
    
    # Link types
    LIST_LINK = "ListLink"
    EVALUATION_LINK = "EvaluationLink"
    IMPLICATION_LINK = "ImplicationLink"
    TENSOR_LINK = "TensorLink"
    INFERENCE_LINK = "InferenceLink"
    SYMBOLIC_LINK = "SymbolicLink"


@dataclass
class TruthValue:
    """Truth value representation for atoms"""
    strength: float  # [0.0, 1.0]
    confidence: float  # [0.0, 1.0]
    
    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class Atom:
    """Enhanced atom representation with neural-symbolic support"""
    atom_id: str
    atom_type: AtomType
    name: Optional[str] = None
    value: Optional[Any] = None
    truth_value: Optional[TruthValue] = None
    tensor_data: Optional[SymbolicTensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.truth_value is None:
            self.truth_value = TruthValue(strength=1.0, confidence=1.0)


@dataclass
class Link:
    """Enhanced link representation with neural support"""
    link_id: str
    link_type: AtomType
    outgoing: List[str]  # List of atom IDs
    truth_value: Optional[TruthValue] = None
    tensor_data: Optional[SymbolicTensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.truth_value is None:
            self.truth_value = TruthValue(strength=1.0, confidence=1.0)


class InferencePattern:
    """Pattern for neural-symbolic inference"""
    
    def __init__(self, pattern_id: str, pattern_type: str):
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type
        self.conditions: List[Dict[str, Any]] = []
        self.conclusions: List[Dict[str, Any]] = []
        self.confidence_threshold: float = 0.7
        self.kernel_operations: List[SymbolicOperation] = []
    
    def add_condition(self, atom_pattern: Dict[str, Any]):
        """Add condition pattern for inference"""
        self.conditions.append(atom_pattern)
    
    def add_conclusion(self, atom_pattern: Dict[str, Any]):
        """Add conclusion pattern for inference"""
        self.conclusions.append(atom_pattern)
    
    def add_kernel_operation(self, operation: SymbolicOperation):
        """Add kernel operation for neural processing"""
        self.kernel_operations.append(operation)


class NeuralSymbolicBridge:
    """Bridge between GGML kernels and AtomSpace"""
    
    def __init__(self, atomspace: Optional[AtomSpaceCore] = None):
        self.atomspace = atomspace or AtomSpaceCore()
        self.kernel_manager = get_kernel_manager()
        self.atoms: Dict[str, Atom] = {}
        self.links: Dict[str, Link] = {}
        self.inference_patterns: Dict[str, InferencePattern] = {}
        self.tensor_cache: Dict[str, SymbolicTensor] = {}
        self.next_atom_id = 1
        self.next_link_id = 1
        self.performance_stats = {
            'conversions': 0,
            'inferences': 0,
            'cache_hits': 0,
            'total_time': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize neural-symbolic bridge"""
        try:
            logger.info("Initializing Neural-Symbolic Bridge...")
            
            # Initialize AtomSpace
            await self.atomspace.initialize()
            
            # Create core symbolic inference patterns
            await self._create_core_patterns()
            
            logger.info("✅ Neural-Symbolic Bridge initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize bridge: {e}")
            return False
    
    async def atom_to_tensor(self, atom_id: str, 
                           architecture: KernelArchitecture = KernelArchitecture.CPU_X86_64) -> Optional[SymbolicTensor]:
        """Convert atom to symbolic tensor representation"""
        start_time = time.perf_counter()
        
        try:
            atom = self.atoms.get(atom_id)
            if not atom:
                logger.warning(f"Atom {atom_id} not found")
                return None
            
            # Check cache first
            cache_key = f"atom_{atom_id}_{architecture.value}"
            if cache_key in self.tensor_cache:
                self.performance_stats['cache_hits'] += 1
                return self.tensor_cache[cache_key]
            
            # Convert atom to tensor based on type
            if atom.tensor_data:
                # Already has tensor data
                result = atom.tensor_data
            else:
                # Generate tensor from atom properties
                result = await self._generate_atom_tensor(atom, architecture)
            
            # Cache result
            self.tensor_cache[cache_key] = result
            
            # Update stats
            self.performance_stats['conversions'] += 1
            self.performance_stats['total_time'] += time.perf_counter() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to convert atom to tensor: {e}")
            return None
    
    async def tensor_to_atom(self, tensor: SymbolicTensor, 
                           atom_type: AtomType = AtomType.TENSOR_NODE,
                           name: Optional[str] = None) -> str:
        """Convert symbolic tensor to atom"""
        start_time = time.perf_counter()
        
        try:
            # Generate atom ID
            atom_id = f"atom_{self.next_atom_id}"
            self.next_atom_id += 1
            
            # Extract symbolic information from tensor
            symbols = tensor.symbols.copy()
            metadata = tensor.metadata.copy()
            
            # Calculate truth value from tensor properties
            if tensor.data.size > 0:
                strength = float(np.mean(np.abs(tensor.data)))
                confidence = 1.0 - float(np.std(tensor.data) / (np.mean(np.abs(tensor.data)) + 1e-6))
                confidence = max(0.0, min(1.0, confidence))
            else:
                strength = 0.5
                confidence = 0.5
            
            # Create atom
            atom = Atom(
                atom_id=atom_id,
                atom_type=atom_type,
                name=name or f"tensor_{len(self.atoms)}",
                truth_value=TruthValue(strength=strength, confidence=confidence),
                tensor_data=tensor,
                metadata={
                    **metadata,
                    'tensor_shape': tensor.data.shape,
                    'tensor_dtype': str(tensor.data.dtype),
                    'symbol_count': len(symbols),
                    'created_from_tensor': True
                }
            )
            
            # Store atom
            self.atoms[atom_id] = atom
            
            # Update stats
            self.performance_stats['conversions'] += 1
            self.performance_stats['total_time'] += time.perf_counter() - start_time
            
            logger.debug(f"Created atom {atom_id} from tensor")
            return atom_id
            
        except Exception as e:
            logger.error(f"Failed to convert tensor to atom: {e}")
            return ""
    
    async def create_symbolic_link(self, link_type: AtomType, 
                                 outgoing_atoms: List[str],
                                 tensor_operation: Optional[SymbolicOperation] = None,
                                 architecture: KernelArchitecture = KernelArchitecture.CPU_X86_64) -> str:
        """Create link with neural-symbolic computation"""
        try:
            # Generate link ID
            link_id = f"link_{self.next_link_id}"
            self.next_link_id += 1
            
            # Validate outgoing atoms exist
            for atom_id in outgoing_atoms:
                if atom_id not in self.atoms:
                    raise ValueError(f"Atom {atom_id} not found")
            
            # Perform tensor operation if specified
            tensor_data = None
            if tensor_operation is not None and len(outgoing_atoms) >= 1:
                # Get tensors from outgoing atoms
                tensors = []
                for atom_id in outgoing_atoms:
                    atom_tensor = await self.atom_to_tensor(atom_id, architecture)
                    if atom_tensor:
                        tensors.append(atom_tensor)
                
                if tensors:
                    # Execute kernel operation
                    try:
                        tensor_data = await self.kernel_manager.execute_operation(
                            tensor_operation, tensors, architecture=architecture
                        )
                    except Exception as e:
                        logger.warning(f"Tensor operation failed in link creation: {e}")
                        tensor_data = None
            
            # Calculate link truth value
            atom_strengths = []
            atom_confidences = []
            for atom_id in outgoing_atoms:
                atom = self.atoms[atom_id]
                atom_strengths.append(atom.truth_value.strength)
                atom_confidences.append(atom.truth_value.confidence)
            
            # Aggregate truth values
            link_strength = np.mean(atom_strengths) if atom_strengths else 1.0
            link_confidence = np.mean(atom_confidences) if atom_confidences else 1.0
            
            # Create link
            link = Link(
                link_id=link_id,
                link_type=link_type,
                outgoing=outgoing_atoms,
                truth_value=TruthValue(strength=link_strength, confidence=link_confidence),
                tensor_data=tensor_data,
                metadata={
                    'tensor_operation': tensor_operation.name if tensor_operation else None,
                    'architecture': architecture.value,
                    'outgoing_count': len(outgoing_atoms)
                }
            )
            
            # Store link
            self.links[link_id] = link
            
            logger.debug(f"Created symbolic link {link_id} with {len(outgoing_atoms)} atoms")
            return link_id
            
        except Exception as e:
            logger.error(f"Failed to create symbolic link: {e}")
            return ""
    
    async def neural_inference(self, pattern_id: str, 
                              input_atoms: List[str],
                              architecture: KernelArchitecture = KernelArchitecture.CPU_X86_64) -> List[str]:
        """Perform neural-symbolic inference using patterns"""
        start_time = time.perf_counter()
        
        try:
            pattern = self.inference_patterns.get(pattern_id)
            if not pattern:
                raise ValueError(f"Pattern {pattern_id} not found")
            
            # Check if input atoms match pattern conditions
            if not await self._match_pattern_conditions(pattern, input_atoms):
                logger.debug(f"Input atoms don't match pattern {pattern_id} conditions")
                return []
            
            # Get tensors from input atoms
            input_tensors = []
            for atom_id in input_atoms:
                tensor = await self.atom_to_tensor(atom_id, architecture)
                if tensor:
                    input_tensors.append(tensor)
            
            if not input_tensors:
                logger.warning("No valid tensors from input atoms")
                return []
            
            # Execute kernel operations in sequence
            current_tensor = input_tensors[0]
            for operation in pattern.kernel_operations:
                if operation == SymbolicOperation.SYMBOL_ADD and len(input_tensors) > 1:
                    current_tensor = await self.kernel_manager.execute_operation(
                        operation, input_tensors, architecture=architecture
                    )
                elif operation == SymbolicOperation.PATTERN_RECOGNITION:
                    current_tensor = await self.kernel_manager.execute_operation(
                        operation, [current_tensor], architecture=architecture
                    )
                elif operation == SymbolicOperation.TENSOR_TO_SYMBOL:
                    current_tensor = await self.kernel_manager.execute_operation(
                        operation, [current_tensor], architecture=architecture
                    )
                # Add more operation handling as needed
            
            # Generate conclusion atoms
            conclusion_atoms = []
            for conclusion_pattern in pattern.conclusions:
                # Create atom from pattern and processed tensor
                atom_type = AtomType(conclusion_pattern.get('type', AtomType.CONCEPT_NODE.value))
                atom_name = conclusion_pattern.get('name', f"inferred_{len(self.atoms)}")
                
                atom_id = await self.tensor_to_atom(current_tensor, atom_type, atom_name)
                if atom_id:
                    conclusion_atoms.append(atom_id)
            
            # Update stats
            self.performance_stats['inferences'] += 1
            self.performance_stats['total_time'] += time.perf_counter() - start_time
            
            logger.info(f"Neural inference completed: {len(conclusion_atoms)} conclusions")
            return conclusion_atoms
            
        except Exception as e:
            logger.error(f"Neural inference failed: {e}")
            return []
    
    async def create_inference_pattern(self, pattern_id: str, pattern_type: str) -> InferencePattern:
        """Create new inference pattern"""
        pattern = InferencePattern(pattern_id, pattern_type)
        self.inference_patterns[pattern_id] = pattern
        logger.debug(f"Created inference pattern {pattern_id}")
        return pattern
    
    async def query_atoms_by_pattern(self, atom_pattern: Dict[str, Any]) -> List[str]:
        """Query atoms matching a pattern"""
        matching_atoms = []
        
        for atom_id, atom in self.atoms.items():
            if self._atom_matches_pattern(atom, atom_pattern):
                matching_atoms.append(atom_id)
        
        return matching_atoms
    
    async def get_atom_neighbors(self, atom_id: str, max_depth: int = 2) -> Set[str]:
        """Get neighboring atoms connected through links"""
        neighbors = set()
        current_level = {atom_id}
        
        for depth in range(max_depth):
            next_level = set()
            for current_atom in current_level:
                # Find links containing this atom
                for link in self.links.values():
                    if current_atom in link.outgoing:
                        # Add all other atoms in the link
                        for linked_atom in link.outgoing:
                            if linked_atom != current_atom:
                                neighbors.add(linked_atom)
                                next_level.add(linked_atom)
            current_level = next_level
        
        return neighbors
    
    async def compute_atom_similarity(self, atom1_id: str, atom2_id: str,
                                    architecture: KernelArchitecture = KernelArchitecture.CPU_X86_64) -> float:
        """Compute similarity between two atoms using neural kernels"""
        try:
            tensor1 = await self.atom_to_tensor(atom1_id, architecture)
            tensor2 = await self.atom_to_tensor(atom2_id, architecture)
            
            if not tensor1 or not tensor2:
                return 0.0
            
            # Use pattern recognition to find similarity
            combined_tensors = [tensor1, tensor2]
            similarity_tensor = await self.kernel_manager.execute_operation(
                SymbolicOperation.PATTERN_RECOGNITION, combined_tensors, architecture=architecture
            )
            
            # Extract similarity score from result
            if 'similarity' in similarity_tensor.symbols:
                return float(similarity_tensor.symbols['similarity'])
            
            # Fallback: compute cosine similarity
            if tensor1.data.size == tensor2.data.size:
                flat1 = tensor1.data.flatten()
                flat2 = tensor2.data.flatten()
                cosine_sim = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2) + 1e-8)
                return float(cosine_sim)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to compute atom similarity: {e}")
            return 0.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the bridge"""
        avg_time = (self.performance_stats['total_time'] / 
                   max(1, self.performance_stats['conversions'] + self.performance_stats['inferences']))
        
        return {
            'atoms_count': len(self.atoms),
            'links_count': len(self.links),
            'patterns_count': len(self.inference_patterns),
            'cached_tensors': len(self.tensor_cache),
            'total_conversions': self.performance_stats['conversions'],
            'total_inferences': self.performance_stats['inferences'],
            'cache_hits': self.performance_stats['cache_hits'],
            'cache_hit_rate': (self.performance_stats['cache_hits'] / 
                              max(1, self.performance_stats['conversions'])),
            'avg_operation_time_ms': avg_time * 1000,
            'total_time_seconds': self.performance_stats['total_time']
        }
    
    async def _generate_atom_tensor(self, atom: Atom, 
                                  architecture: KernelArchitecture) -> SymbolicTensor:
        """Generate tensor representation for an atom"""
        # Create base tensor from atom properties
        if atom.atom_type == AtomType.NUMBER_NODE and isinstance(atom.value, (int, float)):
            # Numeric atom: create tensor with the value
            data = np.array([atom.value], dtype=np.float32)
        elif atom.name:
            # Named atom: create hash-based tensor
            hash_val = hash(atom.name) % 1000000
            data = np.array([hash_val / 1000000.0], dtype=np.float32)
        else:
            # Default: unit tensor
            data = np.array([1.0], dtype=np.float32)
        
        # Add truth value information
        truth_data = np.array([atom.truth_value.strength, atom.truth_value.confidence])
        data = np.concatenate([data, truth_data])
        
        # Create symbolic representation
        symbols = {
            'atom_id': atom.atom_id,
            'atom_type': atom.atom_type.value,
            'truth_strength': atom.truth_value.strength,
            'truth_confidence': atom.truth_value.confidence
        }
        
        if atom.name:
            symbols['name'] = atom.name
        if atom.value is not None:
            symbols['value'] = atom.value
        
        # Add metadata
        metadata = {
            'source': 'atom_conversion',
            'atom_id': atom.atom_id,
            **atom.metadata
        }
        
        return SymbolicTensor(data=data, symbols=symbols, metadata=metadata)
    
    async def _match_pattern_conditions(self, pattern: InferencePattern, 
                                      input_atoms: List[str]) -> bool:
        """Check if input atoms match pattern conditions"""
        if not pattern.conditions:
            return True  # No conditions means always match
        
        # Simple pattern matching - can be extended
        for condition in pattern.conditions:
            matched = False
            for atom_id in input_atoms:
                atom = self.atoms.get(atom_id)
                if atom and self._atom_matches_pattern(atom, condition):
                    matched = True
                    break
            if not matched:
                return False
        
        return True
    
    def _atom_matches_pattern(self, atom: Atom, pattern: Dict[str, Any]) -> bool:
        """Check if atom matches a pattern"""
        if 'type' in pattern:
            if atom.atom_type.value != pattern['type']:
                return False
        
        if 'name' in pattern:
            if atom.name != pattern['name']:
                return False
        
        if 'min_strength' in pattern:
            if atom.truth_value.strength < pattern['min_strength']:
                return False
        
        if 'min_confidence' in pattern:
            if atom.truth_value.confidence < pattern['min_confidence']:
                return False
        
        return True
    
    async def _create_core_patterns(self):
        """Create core symbolic inference patterns"""
        # Financial reasoning pattern
        financial_pattern = await self.create_inference_pattern(
            "financial_analysis", "cognitive_reasoning"
        )
        financial_pattern.add_condition({
            'type': AtomType.CONCEPT_NODE.value,
            'min_strength': 0.5
        })
        financial_pattern.add_conclusion({
            'type': AtomType.CONCEPT_NODE.value,
            'name': 'financial_insight'
        })
        financial_pattern.add_kernel_operation(SymbolicOperation.PATTERN_RECOGNITION)
        financial_pattern.add_kernel_operation(SymbolicOperation.TENSOR_TO_SYMBOL)
        
        # Pattern recognition pattern
        pattern_pattern = await self.create_inference_pattern(
            "pattern_recognition", "cognitive_analysis"
        )
        pattern_pattern.add_condition({
            'type': AtomType.TENSOR_NODE.value
        })
        pattern_pattern.add_conclusion({
            'type': AtomType.CONCEPT_NODE.value,
            'name': 'recognized_pattern'
        })
        pattern_pattern.add_kernel_operation(SymbolicOperation.PATTERN_RECOGNITION)
        
        # Similarity analysis pattern
        similarity_pattern = await self.create_inference_pattern(
            "similarity_analysis", "comparative_reasoning"
        )
        similarity_pattern.add_condition({
            'min_strength': 0.3
        })
        similarity_pattern.add_conclusion({
            'type': AtomType.CONCEPT_NODE.value,
            'name': 'similarity_result'
        })
        similarity_pattern.add_kernel_operation(SymbolicOperation.SYMBOL_ADD)
        similarity_pattern.add_kernel_operation(SymbolicOperation.PATTERN_RECOGNITION)
        
        logger.info("Created core inference patterns")


# Global bridge instance
_neural_symbolic_bridge = None

def get_neural_symbolic_bridge(atomspace: Optional[AtomSpaceCore] = None) -> NeuralSymbolicBridge:
    """Get global neural-symbolic bridge instance"""
    global _neural_symbolic_bridge
    if _neural_symbolic_bridge is None:
        _neural_symbolic_bridge = NeuralSymbolicBridge(atomspace)
    return _neural_symbolic_bridge


# Convenience functions for neural-symbolic operations
async def convert_atom_to_tensor(atom_id: str, 
                               architecture: KernelArchitecture = KernelArchitecture.CPU_X86_64) -> Optional[SymbolicTensor]:
    """Convert atom to tensor using neural-symbolic bridge"""
    bridge = get_neural_symbolic_bridge()
    return await bridge.atom_to_tensor(atom_id, architecture)


async def convert_tensor_to_atom(tensor: SymbolicTensor, 
                               atom_type: AtomType = AtomType.TENSOR_NODE,
                               name: Optional[str] = None) -> str:
    """Convert tensor to atom using neural-symbolic bridge"""
    bridge = get_neural_symbolic_bridge()
    return await bridge.tensor_to_atom(tensor, atom_type, name)


async def perform_neural_inference(pattern_id: str, input_atoms: List[str],
                                 architecture: KernelArchitecture = KernelArchitecture.CPU_X86_64) -> List[str]:
    """Perform neural-symbolic inference"""
    bridge = get_neural_symbolic_bridge()
    return await bridge.neural_inference(pattern_id, input_atoms, architecture)