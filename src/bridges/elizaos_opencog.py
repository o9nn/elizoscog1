"""
ElizaOS-OpenCog Bridge Implementation

Connects ElizaOS agents to OpenCog AtomSpace via:
1. Real HTTP calls to a running atomspace-restful server when reachable.
2. HTTP calls to a running ElizaOS server for agent actions.
3. Graceful fallback to in-process simulation so unit tests work offline.

Environment variables (all optional):
  ATOMSPACE_HOST / ATOMSPACE_PORT    – atomspace-restful server (default localhost:5000)
  ELIZAOS_HOST  / ELIZAOS_PORT       – ElizaOS API server      (default localhost:3000)
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional

try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False

import sys
import os as _os

# Add financial package to path
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))

try:
    from financial import FinancialReasoningEngine
except ImportError:
    FinancialReasoningEngine = None

logger = logging.getLogger(__name__)

_ATOMSPACE_BASE = (
    f"http://{os.environ.get('ATOMSPACE_HOST', 'localhost')}"
    f":{os.environ.get('ATOMSPACE_PORT', '5000')}"
)
_ELIZAOS_BASE = (
    f"http://{os.environ.get('ELIZAOS_HOST', 'localhost')}"
    f":{os.environ.get('ELIZAOS_PORT', '3000')}"
)


async def _http_get(url: str, timeout: float = 5.0) -> Optional[Any]:
    """Perform an async GET; return parsed JSON or None on any error."""
    if not _AIOHTTP_AVAILABLE:
        return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception as exc:
        logger.debug(f"GET {url} failed: {exc}")
    return None


async def _http_post(url: str, payload: Dict, timeout: float = 5.0) -> Optional[Any]:
    """Perform an async POST; return parsed JSON or None on any error."""
    if not _AIOHTTP_AVAILABLE:
        return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status in (200, 201):
                    return await resp.json()
    except Exception as exc:
        logger.debug(f"POST {url} failed: {exc}")
    return None


class AtomSpaceProvider:
    """ElizaOS provider for OpenCog AtomSpace operations.

    Issues real HTTP requests to atomspace-restful when the server is
    reachable.  Falls back to an in-process dict store otherwise.
    """

    def __init__(self, atomspace_config: Dict[str, Any]):
        self.config = atomspace_config
        self._base_url = (
            f"http://{atomspace_config.get('host', 'localhost')}"
            f":{atomspace_config.get('port', 5000)}"
        )
        self.atomspace = None
        self._server_available = False

    async def initialize(self):
        """Initialize connection to AtomSpace."""
        # Try live server
        result = await _http_get(f"{self._base_url}/api/v1.1/atomspaces", timeout=2.0)
        if result is not None:
            self._server_available = True
            logger.info(f"AtomSpaceProvider: connected to live server at {self._base_url}")
        else:
            logger.info("AtomSpaceProvider: server not reachable — using in-process fallback")

        # Always keep a local cache for offline operation
        self.atomspace = {'atoms': {}, 'links': {}, 'next_id': 1}

    async def store_knowledge(self, knowledge: Dict[str, Any]) -> bool:
        """Store knowledge in AtomSpace format."""
        if not self.atomspace:
            await self.initialize()

        # Build a human-readable name from the knowledge content
        name = knowledge.get('type', 'Knowledge')
        content_preview = str(knowledge.get('content', ''))[:40]
        atom_name = f"{name}-{content_preview}" if content_preview else name

        # Try persisting to live server
        if self._server_available:
            payload = {
                "type": "ConceptNode",
                "name": atom_name,
                "truthvalue": {"type": "simple", "details": {"strength": 0.9, "count": 0.9}}
            }
            result = await _http_post(f"{self._base_url}/api/v1.1/atoms", payload)
            if result:
                logger.debug(f"Stored atom on server: {atom_name}")

        # Always mirror locally
        atom_id = self.atomspace['next_id']
        self.atomspace['next_id'] += 1
        self.atomspace['atoms'][atom_id] = {
            'id': atom_id,
            'type': 'ConceptNode',
            'name': atom_name,
            'data': knowledge,
            'created_at': str(knowledge.get('timestamp', 'unknown'))
        }
        return True

    async def query_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Query AtomSpace using pattern matching."""
        results = []
        query_lower = query.lower()

        # Try live server first
        if self._server_available:
            server_result = await _http_get(
                f"{self._base_url}/api/v1.1/atoms?name={query}&filterby=name"
            )
            if isinstance(server_result, dict) and "result" in server_result:
                for atom in server_result["result"].get("atoms", []):
                    results.append({
                        'atom_id': atom.get("handle"),
                        'type': atom.get("type"),
                        'name': atom.get("name", ""),
                        'data': {},
                        'confidence': 0.85,
                        'source': 'server'
                    })
                if results:
                    return results

        # Local fallback
        if self.atomspace:
            for atom_id, atom in self.atomspace['atoms'].items():
                atom_data = atom.get('data', {})
                content = str(atom_data.get('content', '')).lower()
                if query_lower in content or query_lower in atom.get('name', '').lower():
                    results.append({
                        'atom_id': atom_id,
                        'type': atom['type'],
                        'name': atom['name'],
                        'data': atom_data,
                        'confidence': 0.8
                    })

        logger.debug(f"Query '{query}' returned {len(results)} results")
        return results

    async def reason_about(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply PLN reasoning to context.

        When a live ElizaOS server is reachable the reasoning request is
        forwarded; otherwise simple rule-based conclusions are returned.
        """
        context_type = context.get('type', 'unknown')

        # Try forwarding to ElizaOS reasoning endpoint
        if _AIOHTTP_AVAILABLE:
            result = await _http_post(
                f"{_ELIZAOS_BASE}/api/reasoning",
                {"context": context},
                timeout=3.0
            )
            if result:
                return result

        # Local rule-based fallback
        reasoning_result: Dict[str, Any] = {
            'conclusions': [],
            'confidence': 0.7,
            'reasoning_steps': []
        }
        if context_type == 'financial':
            reasoning_result['conclusions'].append(
                "Financial context detected — applying financial reasoning patterns"
            )
            reasoning_result['reasoning_steps'].append("pattern_match_financial")
        elif context_type == 'message':
            reasoning_result['conclusions'].append(
                "Message context detected — applying conversational reasoning"
            )
            reasoning_result['reasoning_steps'].append("pattern_match_conversation")

        return reasoning_result


class CogServerAction:
    """ElizaOS action for CogServer communication.

    Sends real HTTP requests to the CogServer REST API when available;
    degrades gracefully if no server is running.
    """

    def __init__(self, cogserver_url: str):
        self.cogserver_url = cogserver_url
        self.subscribed_events: List[str] = []

    async def execute(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action through CogServer."""
        action_type = action_data.get('type', 'unknown')

        # Try real CogServer REST endpoint
        server_result = await _http_post(
            f"{self.cogserver_url}/api/execute",
            action_data
        )
        if server_result:
            logger.debug(f"CogServer executed action: {action_type}")
            return server_result

        # Fallback response
        result: Dict[str, Any] = {
            'status': 'success',
            'action_type': action_type,
            'cogserver_response': f"Executed {action_type} action (local fallback)",
            'timestamp': str(action_data.get('timestamp', 'unknown'))
        }
        if action_type == 'query':
            result['result'] = {'query_results': []}
        elif action_type == 'reasoning':
            result['result'] = {'reasoning_output': 'local_fallback_result'}
        else:
            result['result'] = {'output': 'generic_action_completed'}

        logger.debug(f"CogServer action '{action_type}' completed via fallback")
        return result

    async def subscribe_to_events(self, event_types: List[str]):
        """Subscribe to CogServer events."""
        # Try real subscription via WebSocket/REST
        if _AIOHTTP_AVAILABLE:
            await _http_post(
                f"{self.cogserver_url}/api/subscribe",
                {"events": event_types}
            )
        self.subscribed_events = event_types
        logger.debug(f"Subscribed to CogServer events: {event_types}")


class PLNReasoner:
    """ElizaOS reasoning service using PLN (Probabilistic Logic Networks)"""
    
    def __init__(self, pln_config: Dict[str, Any]):
        self.config = pln_config
        
    async def infer(self, premises: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform PLN inference on premises"""
        conclusions = []
        
        # Enhanced PLN inference implementation
        for premise in premises:
            confidence = premise.get('confidence', 0.5)
            premise_type = premise.get('type', 'unknown')
            
            # Financial reasoning rules
            if premise_type == 'financial_pattern':
                conclusions.extend(await self._infer_financial_patterns(premise, confidence))
            elif premise_type == 'conversation' or premise_type == 'message':
                conclusions.extend(await self._infer_conversation_intent(premise, confidence))
            elif premise_type == 'transaction':
                conclusions.extend(await self._infer_transaction_patterns(premise, confidence))
            elif premise_type == 'spending_behavior':
                conclusions.extend(await self._infer_spending_insights(premise, confidence))
            else:
                # Generic inference for unknown types
                conclusions.append({
                    'type': 'generic_inference',
                    'content': f"Processed {premise_type} premise with general reasoning",
                    'confidence': confidence * 0.6,
                    'source_premise': premise.get('content', 'unknown')
                })
        
        # Apply cross-premise reasoning
        if len(premises) > 1:
            cross_conclusions = await self._cross_premise_reasoning(premises)
            conclusions.extend(cross_conclusions)
        
        logger.debug(f"PLN inference: {len(premises)} premises -> {len(conclusions)} conclusions")
        return conclusions
    
    async def _infer_financial_patterns(self, premise: Dict[str, Any], confidence: float) -> List[Dict[str, Any]]:
        """Apply financial pattern reasoning"""
        conclusions = []
        pattern_name = premise.get('name', 'unknown')
        
        # Trend prediction
        conclusions.append({
            'type': 'financial_prediction',
            'content': f"Based on pattern {pattern_name}, predict similar future behavior",
            'confidence': confidence * 0.8,
            'source_premise': pattern_name,
            'reasoning_type': 'trend_analysis'
        })
        
        # Budget impact analysis
        if 'expense' in pattern_name.lower() or 'spending' in pattern_name.lower():
            conclusions.append({
                'type': 'budget_impact',
                'content': f"Pattern {pattern_name} may impact budget allocation",
                'confidence': confidence * 0.7,
                'source_premise': pattern_name,
                'reasoning_type': 'budget_analysis'
            })
        
        return conclusions
    
    async def _infer_conversation_intent(self, premise: Dict[str, Any], confidence: float) -> List[Dict[str, Any]]:
        """Apply conversation intent reasoning"""
        conclusions = []
        content = premise.get('content', '').lower()
        category = premise.get('category', 'general')
        
        # Intent classification
        if any(word in content for word in ['spend', 'expense', 'cost', 'money', 'budget']):
            conclusions.append({
                'type': 'financial_intent',
                'content': f"User query indicates financial interest - category: {category}",
                'confidence': confidence * 0.9,
                'source_premise': content[:50] + "...",
                'reasoning_type': 'intent_classification'
            })
        
        # Response strategy
        conclusions.append({
            'type': 'response_strategy',
            'content': f"Recommend {category} response with analytical approach",
            'confidence': confidence * 0.8,
            'source_premise': category,
            'reasoning_type': 'response_planning'
        })
        
        return conclusions
    
    async def _infer_transaction_patterns(self, premise: Dict[str, Any], confidence: float) -> List[Dict[str, Any]]:
        """Apply transaction pattern reasoning"""
        conclusions = []
        amount = premise.get('amount', 0)
        description = premise.get('description', '')
        
        # Spending pattern analysis
        if amount < 0:  # Expense
            conclusions.append({
                'type': 'spending_pattern',
                'content': f"Expense pattern detected: {description} - amount: {abs(amount)}",
                'confidence': confidence * 0.8,
                'source_premise': description,
                'reasoning_type': 'pattern_detection'
            })
        
        # Anomaly detection
        if abs(amount) > 1000:  # Large transaction
            conclusions.append({
                'type': 'anomaly_alert',
                'content': f"Large transaction detected: {description} - review recommended",
                'confidence': confidence * 0.9,
                'source_premise': f"Amount: {amount}",
                'reasoning_type': 'anomaly_detection'
            })
        
        return conclusions
    
    async def _infer_spending_insights(self, premise: Dict[str, Any], confidence: float) -> List[Dict[str, Any]]:
        """Apply spending behavior reasoning"""
        conclusions = []
        behavior_data = premise.get('data', {})
        
        # Behavioral insights
        conclusions.append({
            'type': 'behavioral_insight',
            'content': f"Spending behavior analysis reveals patterns in {behavior_data.get('category', 'general')} spending",
            'confidence': confidence * 0.8,
            'source_premise': str(behavior_data),
            'reasoning_type': 'behavioral_analysis'
        })
        
        return conclusions
    
    async def _cross_premise_reasoning(self, premises: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply reasoning across multiple premises"""
        conclusions = []
        
        # Find correlations between premises
        financial_premises = [p for p in premises if 'financial' in p.get('type', '')]
        conversation_premises = [p for p in premises if 'conversation' in p.get('type', '') or 'message' in p.get('type', '')]
        
        if financial_premises and conversation_premises:
            conclusions.append({
                'type': 'contextual_synthesis',
                'content': f"Financial data context enhances conversation understanding - {len(financial_premises)} financial patterns inform response",
                'confidence': 0.85,
                'source_premise': 'cross_premise_analysis',
                'reasoning_type': 'contextual_integration'
            })
        
        return conclusions
        
    async def validate_reasoning(self, conclusion: Dict[str, Any]) -> float:
        """Validate reasoning conclusion and return confidence"""
        # Mock confidence calculation based on conclusion type and content
        base_confidence = conclusion.get('confidence', 0.5)
        
        validation_factors = {
            'financial_prediction': 0.7,
            'response_intent': 0.8,
            'pattern_match': 0.9,
            'unknown': 0.5
        }
        
        conclusion_type = conclusion.get('type', 'unknown')
        validation_factor = validation_factors.get(conclusion_type, 0.5)
        
        final_confidence = base_confidence * validation_factor
        logger.debug(f"Validated conclusion type '{conclusion_type}' with confidence: {final_confidence}")
        
        return final_confidence


class OpenCogAgentTemplate:
    """Template for creating ElizaOS agents backed by OpenCog"""
    
    def __init__(self, agent_config: Dict[str, Any]):
        self.config = agent_config
        self.atomspace_provider = AtomSpaceProvider(agent_config.get('atomspace', {}))
        self.pln_reasoner = PLNReasoner(agent_config.get('pln', {}))
        
        # Initialize financial reasoning if available
        self.financial_engine = None
        if FinancialReasoningEngine and agent_config.get('gnucash_file'):
            self.financial_engine = FinancialReasoningEngine(
                agent_config['gnucash_file'],
                agent_config.get('atomspace', {})
            )
        
    async def initialize(self):
        """Initialize all components"""
        await self.atomspace_provider.initialize()
        
        if self.financial_engine:
            await self.financial_engine.initialize()
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> str:
        """Process message using OpenCog cognitive capabilities"""
        
        # Check if this is a financial query
        if self.financial_engine and self._is_financial_query(message):
            return await self._process_financial_query(message, context)
        
        # Store message in AtomSpace
        await self.atomspace_provider.store_knowledge({
            'type': 'message',
            'content': message,
            'context': context
        })
        
        # Query relevant knowledge
        relevant_knowledge = await self.atomspace_provider.query_knowledge(message)
        
        # Apply reasoning
        reasoning_result = await self.pln_reasoner.infer(relevant_knowledge)
        
        # Generate response
        return self._generate_response(reasoning_result)
        
    def _is_financial_query(self, message: str) -> bool:
        """Determine if message is a financial query"""
        financial_keywords = [
            'spend', 'spending', 'budget', 'money', 'expense', 'expenses',
            'transaction', 'account', 'balance', 'income', 'cost', 'price',
            'financial', 'groceries', 'utilities', 'rent', 'mortgage'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in financial_keywords)
        
    async def _process_financial_query(self, message: str, context: Dict[str, Any]) -> str:
        """Process financial queries using the financial reasoning engine"""
        try:
            result = await self.financial_engine.answer_financial_question(message, context)
            return result['answer']
        except Exception as e:
            return f"I encountered an issue processing your financial query: {str(e)}"
        
    def _generate_response(self, reasoning_result: List[Dict[str, Any]]) -> str:
        """Generate natural language response from reasoning result"""
        if not reasoning_result:
            return "I understand your message and have processed it cognitively."
        
        # Extract conclusions from reasoning result
        conclusions = [r.get('content', '') for r in reasoning_result if r.get('content')]
        
        if conclusions:
            primary_conclusion = conclusions[0]
            if len(conclusions) > 1:
                return f"Based on my analysis: {primary_conclusion}. I also considered {len(conclusions)-1} additional factors."
            else:
                return f"Based on my cognitive analysis: {primary_conclusion}"
        else:
            return "I have processed your message through cognitive reasoning systems."


# Integration utility functions

def convert_eliza_memory_to_atoms(memory_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert ElizaOS memory format to AtomSpace Atoms"""
    atoms = []
    
    # Convert memories to ConceptNodes and EvaluationLinks
    for key, value in memory_data.items():
        # Create concept node for the memory item
        concept_atom = {
            'type': 'ConceptNode',
            'name': f"Memory-{key}",
            'data': value
        }
        atoms.append(concept_atom)
        
        # Create evaluation link for the memory content
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                eval_atom = {
                    'type': 'EvaluationLink',
                    'predicate': sub_key,
                    'subject': f"Memory-{key}",
                    'object': str(sub_value)
                }
                atoms.append(eval_atom)
    
    logger.debug(f"Converted ElizaOS memory to {len(atoms)} atoms")
    return atoms

def convert_atoms_to_eliza_memory(atoms: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert AtomSpace Atoms to ElizaOS memory format"""
    memory_data = {}
    
    # Extract memory data from atoms
    for atom in atoms:
        if atom.get('type') == 'ConceptNode' and atom.get('name', '').startswith('Memory-'):
            memory_key = atom['name'].replace('Memory-', '')
            memory_data[memory_key] = atom.get('data', {})
    
    logger.debug(f"Converted {len(atoms)} atoms to ElizaOS memory format")
    return memory_data

def create_atomese_query(eliza_query: str) -> str:
    """Convert ElizaOS query to Atomese pattern"""
    # Simple query conversion - extract key terms and create basic Atomese pattern
    query_terms = eliza_query.lower().split()
    
    if 'financial' in query_terms or 'money' in query_terms or 'expense' in query_terms:
        return """(GetLink 
                    (VariableNode "$x") 
                    (EvaluationLink 
                        (PredicateNode "financial_data") 
                        (ListLink (VariableNode "$x"))))"""
    elif 'memory' in query_terms or 'remember' in query_terms:
        return """(GetLink 
                    (VariableNode "$x") 
                    (InheritanceLink 
                        (VariableNode "$x") 
                        (ConceptNode "Memory")))"""
    else:
        # Generic pattern for any concept
        return f"""(GetLink 
                     (VariableNode "$x") 
                     (EvaluationLink 
                         (PredicateNode "relates_to") 
                         (ListLink 
                             (VariableNode "$x") 
                             (ConceptNode "{query_terms[0] if query_terms else 'query'}"))))"""