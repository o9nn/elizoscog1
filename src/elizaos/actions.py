"""
ElizaOS Action System

Provides an extensible action framework for agents to perform various tasks
including financial operations, data analysis, and system integrations.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
from dataclasses import dataclass
import json
import inspect

logger = logging.getLogger(__name__)

@dataclass
class ActionResult:
    """Result of an action execution"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'metadata': self.metadata or {},
            'execution_time': self.execution_time
        }

@dataclass
class ActionParameter:
    """Definition of an action parameter"""
    name: str
    type: type
    description: str
    required: bool = True
    default: Any = None
    validation_regex: Optional[str] = None

class BaseAction(ABC):
    """Base class for all actions"""
    
    def __init__(self, name: str, description: str, parameters: List[ActionParameter] = None):
        self.name = name
        self.description = description
        self.parameters = parameters or []
        self.cognitive_framework = None
        self.execution_count = 0
        self.total_execution_time = 0.0
        
    @abstractmethod
    async def execute(self, **kwargs) -> ActionResult:
        """Execute the action with given parameters"""
        pass
    
    def validate_parameters(self, params: Dict[str, Any]) -> List[str]:
        """Validate input parameters and return list of errors"""
        errors = []
        
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                errors.append(f"Required parameter '{param.name}' is missing")
            
            # Type checking
            if param.name in params:
                value = params[param.name]
                if not isinstance(value, param.type) and value is not None:
                    errors.append(f"Parameter '{param.name}' must be of type {param.type.__name__}")
        
        return errors
    
    def set_cognitive_framework(self, framework):
        """Set the cognitive framework for enhanced processing"""
        self.cognitive_framework = framework
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get action execution statistics"""
        avg_time = self.total_execution_time / max(1, self.execution_count)
        return {
            'name': self.name,
            'execution_count': self.execution_count,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': avg_time
        }

class FinancialQueryAction(BaseAction):
    """Action for querying financial data"""
    
    def __init__(self):
        parameters = [
            ActionParameter('query', str, 'Natural language financial query'),
            ActionParameter('account_filter', str, 'Optional account filter', required=False),
            ActionParameter('date_range', dict, 'Optional date range filter', required=False)
        ]
        super().__init__(
            name='financial_query',
            description='Query financial data using natural language',
            parameters=parameters
        )
    
    async def execute(self, **kwargs) -> ActionResult:
        """Execute financial query"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Validate parameters
            errors = self.validate_parameters(kwargs)
            if errors:
                return ActionResult(
                    success=False,
                    error=f"Parameter validation failed: {', '.join(errors)}"
                )
            
            query = kwargs['query']
            account_filter = kwargs.get('account_filter')
            date_range = kwargs.get('date_range')
            
            logger.info(f"🔍 Executing financial query: {query}")
            
            # Use cognitive framework for enhanced query processing
            if self.cognitive_framework:
                # Route to financial chat agent
                agent = self.cognitive_framework.cognitive_agents.get('financial_chat_agent')
                if agent:
                    context = {
                        'action': 'financial_query',
                        'account_filter': account_filter,
                        'date_range': date_range
                    }
                    
                    result = await agent.process_message(query, context)
                    
                    execution_time = asyncio.get_event_loop().time() - start_time
                    self.execution_count += 1
                    self.total_execution_time += execution_time
                    
                    return ActionResult(
                        success=True,
                        data=result,
                        metadata={'query': query, 'context': context},
                        execution_time=execution_time
                    )
            
            # Fallback implementation
            result = f"[MOCK] Financial query result for: {query}"
            
            execution_time = asyncio.get_event_loop().time() - start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            
            return ActionResult(
                success=True,
                data=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"❌ Financial query action failed: {e}")
            return ActionResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class BudgetAnalysisAction(BaseAction):
    """Action for budget analysis and planning"""
    
    def __init__(self):
        parameters = [
            ActionParameter('analysis_type', str, 'Type of analysis: summary, trends, forecast'),
            ActionParameter('time_period', str, 'Time period: monthly, quarterly, yearly', required=False, default='monthly'),
            ActionParameter('categories', list, 'Expense categories to analyze', required=False)
        ]
        super().__init__(
            name='budget_analysis',
            description='Perform budget analysis and generate insights',
            parameters=parameters
        )
    
    async def execute(self, **kwargs) -> ActionResult:
        """Execute budget analysis"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            errors = self.validate_parameters(kwargs)
            if errors:
                return ActionResult(success=False, error=f"Validation failed: {', '.join(errors)}")
            
            analysis_type = kwargs['analysis_type']
            time_period = kwargs.get('time_period', 'monthly')
            categories = kwargs.get('categories', [])
            
            logger.info(f"📊 Executing budget analysis: {analysis_type}")
            
            if self.cognitive_framework:
                agent = self.cognitive_framework.cognitive_agents.get('budget_planning_agent')
                if agent:
                    context = {
                        'action': 'budget_analysis',
                        'analysis_type': analysis_type,
                        'time_period': time_period,
                        'categories': categories
                    }
                    
                    prompt = f"Perform {analysis_type} budget analysis for {time_period} period"
                    if categories:
                        prompt += f" focusing on categories: {', '.join(categories)}"
                    
                    result = await agent.process_message(prompt, context)
                    
                    execution_time = asyncio.get_event_loop().time() - start_time
                    self.execution_count += 1
                    self.total_execution_time += execution_time
                    
                    return ActionResult(
                        success=True,
                        data=result,
                        metadata=context,
                        execution_time=execution_time
                    )
            
            # Mock result
            result = {
                'analysis_type': analysis_type,
                'time_period': time_period,
                'summary': f'Budget analysis for {time_period} period completed',
                'insights': ['Mock insight 1', 'Mock insight 2'],
                'recommendations': ['Mock recommendation 1']
            }
            
            execution_time = asyncio.get_event_loop().time() - start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            
            return ActionResult(
                success=True,
                data=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"❌ Budget analysis action failed: {e}")
            return ActionResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class TransactionAnalysisAction(BaseAction):
    """Action for analyzing transaction patterns"""
    
    def __init__(self):
        parameters = [
            ActionParameter('transaction_id', str, 'Specific transaction ID to analyze', required=False),
            ActionParameter('pattern_type', str, 'Pattern type: spending, income, anomaly', required=False, default='spending'),
            ActionParameter('lookback_days', int, 'Number of days to look back', required=False, default=30)
        ]
        super().__init__(
            name='transaction_analysis',
            description='Analyze transaction patterns and detect anomalies',
            parameters=parameters
        )
    
    async def execute(self, **kwargs) -> ActionResult:
        """Execute transaction analysis"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            errors = self.validate_parameters(kwargs)
            if errors:
                return ActionResult(success=False, error=f"Validation failed: {', '.join(errors)}")
            
            transaction_id = kwargs.get('transaction_id')
            pattern_type = kwargs.get('pattern_type', 'spending')
            lookback_days = kwargs.get('lookback_days', 30)
            
            logger.info(f"🔍 Executing transaction analysis: {pattern_type}")
            
            if self.cognitive_framework:
                agent = self.cognitive_framework.cognitive_agents.get('transaction_analysis_agent')
                if agent:
                    context = {
                        'action': 'transaction_analysis',
                        'transaction_id': transaction_id,
                        'pattern_type': pattern_type,
                        'lookback_days': lookback_days
                    }
                    
                    if transaction_id:
                        prompt = f"Analyze transaction {transaction_id}"
                    else:
                        prompt = f"Analyze {pattern_type} patterns over the last {lookback_days} days"
                    
                    result = await agent.process_message(prompt, context)
                    
                    execution_time = asyncio.get_event_loop().time() - start_time
                    self.execution_count += 1
                    self.total_execution_time += execution_time
                    
                    return ActionResult(
                        success=True,
                        data=result,
                        metadata=context,
                        execution_time=execution_time
                    )
            
            # Mock implementation
            result = {
                'pattern_type': pattern_type,
                'lookback_days': lookback_days,
                'patterns_found': 3,
                'anomalies_detected': 1 if pattern_type == 'anomaly' else 0,
                'summary': f'Transaction analysis completed for {pattern_type} patterns'
            }
            
            execution_time = asyncio.get_event_loop().time() - start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            
            return ActionResult(
                success=True,
                data=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"❌ Transaction analysis action failed: {e}")
            return ActionResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class SendMessageAction(BaseAction):
    """Action for sending messages through various platforms"""
    
    def __init__(self):
        parameters = [
            ActionParameter('platform', str, 'Platform to send message to: discord, telegram'),
            ActionParameter('channel_id', str, 'Channel or chat ID'),
            ActionParameter('message', str, 'Message content to send'),
            ActionParameter('message_type', str, 'Message type: text, embed, file', required=False, default='text')
        ]
        super().__init__(
            name='send_message',
            description='Send message through social platforms',
            parameters=parameters
        )
    
    async def execute(self, **kwargs) -> ActionResult:
        """Execute send message action"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            errors = self.validate_parameters(kwargs)
            if errors:
                return ActionResult(success=False, error=f"Validation failed: {', '.join(errors)}")
            
            platform = kwargs['platform']
            channel_id = kwargs['channel_id']
            message = kwargs['message']
            message_type = kwargs.get('message_type', 'text')
            
            logger.info(f"📤 Sending message to {platform}:{channel_id}")
            
            # Get connector from cognitive framework
            if self.cognitive_framework and hasattr(self.cognitive_framework, 'connector_manager'):
                connector = self.cognitive_framework.connector_manager.get_connector(platform)
                if connector:
                    success = await connector.send_message(channel_id, message, message_type=message_type)
                    
                    execution_time = asyncio.get_event_loop().time() - start_time
                    self.execution_count += 1
                    self.total_execution_time += execution_time
                    
                    return ActionResult(
                        success=success,
                        data={'message_sent': success, 'platform': platform, 'channel_id': channel_id},
                        metadata={'platform': platform, 'message_type': message_type},
                        execution_time=execution_time
                    )
            
            # Mock implementation
            logger.info(f"[MOCK] Message sent to {platform}:{channel_id}: {message[:50]}...")
            
            execution_time = asyncio.get_event_loop().time() - start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            
            return ActionResult(
                success=True,
                data={'message_sent': True, 'platform': platform, 'channel_id': channel_id},
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"❌ Send message action failed: {e}")
            return ActionResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class WebSearchAction(BaseAction):
    """Action for searching the web"""
    
    def __init__(self):
        parameters = [
            ActionParameter('query', str, 'Search query'),
            ActionParameter('max_results', int, 'Maximum number of results', required=False, default=5),
            ActionParameter('site_filter', str, 'Filter to specific site', required=False)
        ]
        super().__init__(
            name='web_search',
            description='Search the web for information',
            parameters=parameters
        )
    
    async def execute(self, **kwargs) -> ActionResult:
        """Execute web search"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            errors = self.validate_parameters(kwargs)
            if errors:
                return ActionResult(success=False, error=f"Validation failed: {', '.join(errors)}")
            
            query = kwargs['query']
            max_results = kwargs.get('max_results', 5)
            site_filter = kwargs.get('site_filter')
            
            logger.info(f"🔍 Searching web for: {query}")
            
            # Mock search results
            results = []
            for i in range(min(max_results, 3)):
                results.append({
                    'title': f'Search result {i+1} for {query}',
                    'url': f'https://example.com/result{i+1}',
                    'snippet': f'This is a mock snippet for result {i+1} about {query}',
                    'relevance_score': 0.9 - (i * 0.1)
                })
            
            execution_time = asyncio.get_event_loop().time() - start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            
            return ActionResult(
                success=True,
                data={
                    'query': query,
                    'results': results,
                    'total_results': len(results)
                },
                metadata={'site_filter': site_filter},
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"❌ Web search action failed: {e}")
            return ActionResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class ActionRegistry:
    """Registry for managing actions"""
    
    def __init__(self):
        self.actions: Dict[str, BaseAction] = {}
        self.action_history: List[Dict[str, Any]] = []
        self.cognitive_framework = None
        
        # Register built-in actions
        self._register_builtin_actions()
    
    def _register_builtin_actions(self):
        """Register built-in actions"""
        builtin_actions = [
            FinancialQueryAction(),
            BudgetAnalysisAction(),
            TransactionAnalysisAction(),
            SendMessageAction(),
            WebSearchAction()
        ]
        
        for action in builtin_actions:
            self.register_action(action)
    
    def register_action(self, action: BaseAction):
        """Register a new action"""
        self.actions[action.name] = action
        if self.cognitive_framework:
            action.set_cognitive_framework(self.cognitive_framework)
        logger.info(f"📝 Registered action: {action.name}")
    
    def unregister_action(self, name: str) -> bool:
        """Unregister an action"""
        if name in self.actions:
            del self.actions[name]
            logger.info(f"🗑️ Unregistered action: {name}")
            return True
        return False
    
    def get_action(self, name: str) -> Optional[BaseAction]:
        """Get an action by name"""
        return self.actions.get(name)
    
    def list_actions(self) -> List[Dict[str, Any]]:
        """List all registered actions"""
        return [
            {
                'name': action.name,
                'description': action.description,
                'parameters': [
                    {
                        'name': param.name,
                        'type': param.type.__name__,
                        'description': param.description,
                        'required': param.required,
                        'default': param.default
                    }
                    for param in action.parameters
                ]
            }
            for action in self.actions.values()
        ]
    
    def set_cognitive_framework(self, framework):
        """Set cognitive framework for all actions"""
        self.cognitive_framework = framework
        for action in self.actions.values():
            action.set_cognitive_framework(framework)
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics for all actions"""
        stats = {}
        for name, action in self.actions.items():
            stats[name] = action.get_statistics()
        return stats

class ActionExecutor:
    """Executes actions with proper error handling and logging"""
    
    def __init__(self, registry: ActionRegistry):
        self.registry = registry
        self.execution_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
    
    async def execute_action(self, action_name: str, parameters: Dict[str, Any] = None) -> ActionResult:
        """Execute an action by name"""
        start_time = datetime.now()
        
        try:
            action = self.registry.get_action(action_name)
            if not action:
                return ActionResult(
                    success=False,
                    error=f"Action '{action_name}' not found"
                )
            
            parameters = parameters or {}
            logger.info(f"⚡ Executing action: {action_name} with params: {parameters}")
            
            # Execute the action
            result = await action.execute(**parameters)
            
            # Log execution
            execution_record = {
                'action_name': action_name,
                'parameters': parameters,
                'result': result.to_dict(),
                'timestamp': start_time.isoformat(),
                'execution_time': result.execution_time
            }
            
            self.execution_history.append(execution_record)
            
            # Maintain history size
            if len(self.execution_history) > self.max_history_size:
                self.execution_history = self.execution_history[-self.max_history_size:]
            
            if result.success:
                logger.info(f"✅ Action {action_name} completed successfully in {result.execution_time:.3f}s")
            else:
                logger.error(f"❌ Action {action_name} failed: {result.error}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Action execution failed: {e}")
            return ActionResult(
                success=False,
                error=f"Execution error: {str(e)}"
            )
    
    async def execute_action_chain(self, actions: List[Dict[str, Any]]) -> List[ActionResult]:
        """Execute a chain of actions"""
        results = []
        
        for action_config in actions:
            action_name = action_config.get('name')
            parameters = action_config.get('parameters', {})
            
            if not action_name:
                results.append(ActionResult(
                    success=False,
                    error="Action name not specified in chain"
                ))
                continue
            
            result = await self.execute_action(action_name, parameters)
            results.append(result)
            
            # Stop chain on failure if configured
            if not result.success and action_config.get('stop_on_failure', False):
                logger.warning(f"Action chain stopped at {action_name} due to failure")
                break
        
        return results
    
    def get_execution_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for record in self.execution_history if record['result']['success'])
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'failure_rate': (total_executions - successful_executions) / total_executions,
            'average_execution_time': sum(record['execution_time'] for record in self.execution_history) / total_executions
        }

# Decorator for creating actions from functions
def action(name: str, description: str, parameters: List[ActionParameter] = None):
    """Decorator to create actions from functions"""
    def decorator(func: Callable) -> BaseAction:
        
        class FunctionAction(BaseAction):
            def __init__(self):
                super().__init__(name, description, parameters)
                self.func = func
            
            async def execute(self, **kwargs) -> ActionResult:
                start_time = asyncio.get_event_loop().time()
                
                try:
                    errors = self.validate_parameters(kwargs)
                    if errors:
                        return ActionResult(success=False, error=f"Validation failed: {', '.join(errors)}")
                    
                    # Call the function
                    if inspect.iscoroutinefunction(func):
                        result = await func(**kwargs)
                    else:
                        result = func(**kwargs)
                    
                    execution_time = asyncio.get_event_loop().time() - start_time
                    self.execution_count += 1
                    self.total_execution_time += execution_time
                    
                    return ActionResult(
                        success=True,
                        data=result,
                        execution_time=execution_time
                    )
                    
                except Exception as e:
                    execution_time = asyncio.get_event_loop().time() - start_time
                    return ActionResult(
                        success=False,
                        error=str(e),
                        execution_time=execution_time
                    )
        
        return FunctionAction()
    
    return decorator

# Example usage of the decorator
@action(
    name="calculate_sum",
    description="Calculate sum of two numbers",
    parameters=[
        ActionParameter('a', float, 'First number'),
        ActionParameter('b', float, 'Second number')
    ]
)
async def calculate_sum(a: float, b: float) -> float:
    """Example action function"""
    return a + b


# ==================== OpenCog PLN Reasoning Actions ====================

class PLNReasoningAction(BaseAction):
    """Action that uses OpenCog PLN (Probabilistic Logic Networks) for reasoning"""
    
    def __init__(self):
        parameters = [
            ActionParameter('premises', list, 'List of premises for reasoning'),
            ActionParameter('target_conclusion', str, 'Target conclusion to reason about', required=False),
            ActionParameter('confidence_threshold', float, 'Minimum confidence threshold', required=False, default=0.5),
        ]
        super().__init__(
            name="pln_reasoning",
            description="Perform PLN reasoning on given premises",
            parameters=parameters
        )
        self.pln_available = False
        try:
            # Try to import OpenCog PLN
            from opencog.pln import *
            self.pln_available = True
        except ImportError:
            logger.warning("OpenCog PLN not available, using fallback reasoning")
    
    async def execute(self, premises: List[str], target_conclusion: str = None, 
                     confidence_threshold: float = 0.5, **kwargs) -> ActionResult:
        """Execute PLN reasoning"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if self.pln_available:
                result = await self._execute_pln_reasoning(premises, target_conclusion, confidence_threshold)
            else:
                result = await self._fallback_reasoning(premises, target_conclusion, confidence_threshold)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            
            return ActionResult(
                success=True,
                data=result,
                execution_time=execution_time,
                metadata={'reasoning_type': 'PLN', 'premises_count': len(premises)}
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"PLN reasoning failed: {e}")
            return ActionResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _execute_pln_reasoning(self, premises: List[str], target: str, threshold: float) -> Dict:
        """Execute actual PLN reasoning using OpenCog"""
        # This would use OpenCog's PLN inference engine
        # Simplified implementation for now
        conclusions = []
        
        for premise in premises:
            # In real implementation, this would:
            # 1. Convert premises to Atoms
            # 2. Apply PLN inference rules
            # 3. Calculate confidence values
            # 4. Return high-confidence conclusions
            
            conclusion = {
                'statement': f"Inferred from: {premise}",
                'confidence': 0.8,
                'strength': 0.9
            }
            conclusions.append(conclusion)
        
        return {
            'conclusions': conclusions,
            'reasoning_method': 'PLN',
            'total_inferences': len(conclusions)
        }
    
    async def _fallback_reasoning(self, premises: List[str], target: str, threshold: float) -> Dict:
        """Fallback reasoning when PLN is not available"""
        # Simple rule-based reasoning
        conclusions = []
        
        for i, premise in enumerate(premises):
            conclusion = {
                'statement': f"Simple inference {i+1}: Based on {premise[:50]}...",
                'confidence': 0.6,
                'strength': 0.7,
                'method': 'fallback'
            }
            conclusions.append(conclusion)
        
        return {
            'conclusions': conclusions,
            'reasoning_method': 'fallback',
            'total_inferences': len(conclusions),
            'note': 'Using fallback reasoning - OpenCog PLN not available'
        }


class CognitiveActionPlanner(BaseAction):
    """
    Cognitive action planning using OpenCog reasoning
    Plans sequences of actions to achieve goals
    """
    
    def __init__(self, action_registry: 'ActionRegistry'):
        parameters = [
            ActionParameter('goal', str, 'Goal to achieve'),
            ActionParameter('context', dict, 'Current context and state', required=False),
            ActionParameter('max_steps', int, 'Maximum planning steps', required=False, default=10),
        ]
        super().__init__(
            name="cognitive_action_planner",
            description="Plan action sequences using cognitive reasoning",
            parameters=parameters
        )
        self.action_registry = action_registry
    
    async def execute(self, goal: str, context: Dict = None, max_steps: int = 10, **kwargs) -> ActionResult:
        """Execute cognitive action planning"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Generate action plan
            plan = await self._generate_action_plan(goal, context or {}, max_steps)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            
            return ActionResult(
                success=True,
                data=plan,
                execution_time=execution_time,
                metadata={'goal': goal, 'plan_steps': len(plan.get('steps', []))}
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Action planning failed: {e}")
            return ActionResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _generate_action_plan(self, goal: str, context: Dict, max_steps: int) -> Dict:
        """Generate a cognitive action plan"""
        # This would use OpenCog's planning algorithms
        # For now, implementing a simplified version
        
        steps = []
        available_actions = self.action_registry.get_all_actions()
        
        # Analyze goal
        goal_analysis = self._analyze_goal(goal, context)
        
        # Select relevant actions based on goal
        relevant_actions = self._select_relevant_actions(goal_analysis, available_actions)
        
        # Build action sequence
        for i, action in enumerate(relevant_actions[:max_steps]):
            step = {
                'step_number': i + 1,
                'action': action.name,
                'description': action.description,
                'parameters': {},  # Would be filled based on context
                'expected_outcome': f"Progress toward: {goal}",
                'confidence': 0.8 - (i * 0.05)  # Confidence decreases with plan length
            }
            steps.append(step)
        
        plan = {
            'goal': goal,
            'steps': steps,
            'total_steps': len(steps),
            'estimated_success_probability': 0.7,
            'alternative_plans': []  # Could generate multiple plans
        }
        
        return plan
    
    def _analyze_goal(self, goal: str, context: Dict) -> Dict:
        """Analyze goal to determine requirements"""
        # Simple keyword-based analysis
        # Real implementation would use NLP and cognitive reasoning
        
        analysis = {
            'goal_type': 'unknown',
            'required_capabilities': [],
            'constraints': [],
            'context_requirements': []
        }
        
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ['calculate', 'compute', 'analyze']):
            analysis['goal_type'] = 'analytical'
            analysis['required_capabilities'].append('computation')
        
        if any(word in goal_lower for word in ['find', 'search', 'locate']):
            analysis['goal_type'] = 'retrieval'
            analysis['required_capabilities'].append('search')
        
        if any(word in goal_lower for word in ['financial', 'money', 'transaction']):
            analysis['required_capabilities'].append('financial_access')
        
        return analysis
    
    def _select_relevant_actions(self, goal_analysis: Dict, available_actions: List[BaseAction]) -> List[BaseAction]:
        """Select actions relevant to the goal"""
        relevant = []
        
        goal_type = goal_analysis.get('goal_type', '')
        
        for action in available_actions:
            # Match actions to goal type
            if goal_type in action.description.lower():
                relevant.append(action)
            elif any(cap in action.description.lower() for cap in goal_analysis.get('required_capabilities', [])):
                relevant.append(action)
        
        # If no specific matches, return a subset of all actions
        if not relevant:
            relevant = available_actions[:3]
        
        return relevant


class PLNActionSelector(BaseAction):
    """
    Use PLN reasoning to select the best action for a given situation
    """
    
    def __init__(self, action_registry: 'ActionRegistry'):
        parameters = [
            ActionParameter('situation', str, 'Current situation description'),
            ActionParameter('available_actions', list, 'List of available action names', required=False),
            ActionParameter('preferences', dict, 'User preferences for action selection', required=False),
        ]
        super().__init__(
            name="pln_action_selector",
            description="Select best action using PLN reasoning",
            parameters=parameters
        )
        self.action_registry = action_registry
    
    async def execute(self, situation: str, available_actions: List[str] = None,
                     preferences: Dict = None, **kwargs) -> ActionResult:
        """Select action using PLN reasoning"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Get available actions
            if available_actions:
                actions = [self.action_registry.get_action(name) for name in available_actions]
                actions = [a for a in actions if a is not None]
            else:
                actions = self.action_registry.get_all_actions()
            
            # Evaluate each action
            evaluations = []
            for action in actions:
                score = await self._evaluate_action(action, situation, preferences or {})
                evaluations.append({
                    'action': action.name,
                    'score': score,
                    'description': action.description
                })
            
            # Sort by score
            evaluations.sort(key=lambda x: x['score'], reverse=True)
            
            # Select best action
            best_action = evaluations[0] if evaluations else None
            
            execution_time = asyncio.get_event_loop().time() - start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            
            return ActionResult(
                success=True,
                data={
                    'selected_action': best_action,
                    'all_evaluations': evaluations,
                    'reasoning': f"Selected based on situation: {situation[:100]}"
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Action selection failed: {e}")
            return ActionResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _evaluate_action(self, action: BaseAction, situation: str, preferences: Dict) -> float:
        """Evaluate how well an action fits the situation"""
        # This would use PLN to reason about action appropriateness
        # Simplified scoring for now
        
        score = 0.5  # Base score
        
        # Check if action description matches situation keywords
        situation_words = set(situation.lower().split())
        action_words = set(action.description.lower().split())
        
        overlap = situation_words & action_words
        score += len(overlap) * 0.1
        
        # Check preferences
        if preferences.get('prefer_fast', False):
            avg_time = action.total_execution_time / max(1, action.execution_count)
            if avg_time < 1.0:
                score += 0.2
        
        if preferences.get('prefer_reliable', False):
            # Higher execution count suggests reliability
            if action.execution_count > 10:
                score += 0.2
        
        # Cap score at 1.0
        return min(1.0, score)
    return a + b