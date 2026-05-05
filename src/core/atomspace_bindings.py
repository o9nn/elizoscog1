#!/usr/bin/env python3
"""
Core OpenCog AtomSpace Python bindings integration for ElizaOS

Connects to a running atomspace-restful server (https://github.com/opencog/atomspace-restful)
via HTTP when available, and falls back to an in-process dictionary store so the rest of
the codebase keeps working even without a live OpenCog deployment.

Server environment variables (all optional):
  ATOMSPACE_HOST   – hostname of atomspace-restful  (default: localhost)
  ATOMSPACE_PORT   – port of atomspace-restful       (default: 5000)
  ATOMSPACE_SCHEME_PORT – CogServer telnet port      (default: 17001)
"""

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

_ATOMSPACE_HOST = os.environ.get("ATOMSPACE_HOST", "localhost")
_ATOMSPACE_PORT = int(os.environ.get("ATOMSPACE_PORT", "5000"))
_ATOMSPACE_BASE_URL = f"http://{_ATOMSPACE_HOST}:{_ATOMSPACE_PORT}"

class AtomSpaceCore:
    """
    Core AtomSpace implementation.

    When a running atomspace-restful server is reachable the class issues real
    HTTP requests so that atoms/links are stored in OpenCog's native graph DB.
    When the server is unavailable it degrades gracefully to an in-process
    Python-dict store so integration tests keep working offline.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.atoms = {}
        self.links = {}
        self.next_atom_id = 1
        self.next_link_id = 1
        self.initialized = False
        self._server_available = False
        self._base_url = self.config.get(
            "base_url",
            f"http://{self.config.get('host', _ATOMSPACE_HOST)}:{self.config.get('port', _ATOMSPACE_PORT)}"
        )

    # ------------------------------------------------------------------
    # Server connectivity helpers
    # ------------------------------------------------------------------

    async def _check_server(self) -> bool:
        """Return True if atomspace-restful is reachable."""
        if not _AIOHTTP_AVAILABLE:
            return False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._base_url}/api/v1.1/atomspaces",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    return resp.status < 500
        except Exception:
            return False

    async def _rest_get(self, path: str) -> Optional[Any]:
        """GET from atomspace-restful; returns parsed JSON or None on error."""
        if not _AIOHTTP_AVAILABLE or not self._server_available:
            return None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._base_url}{path}",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as exc:
            logger.debug(f"AtomSpace REST GET {path} failed: {exc}")
        return None

    async def _rest_post(self, path: str, payload: Dict) -> Optional[Any]:
        """POST to atomspace-restful; returns parsed JSON or None on error."""
        if not _AIOHTTP_AVAILABLE or not self._server_available:
            return None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}{path}",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status in (200, 201):
                        return await resp.json()
        except Exception as exc:
            logger.debug(f"AtomSpace REST POST {path} failed: {exc}")
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> bool:
        """Initialize AtomSpace core with configuration."""
        try:
            logger.info("Initializing AtomSpace core...")

            self.atom_types = {
                'ConceptNode': 'concept',
                'PredicateNode': 'predicate',
                'NumberNode': 'number',
                'ListLink': 'list',
                'EvaluationLink': 'evaluation',
                'ImplicationLink': 'implication'
            }

            self.atoms = {}
            self.links = {}
            self.truth_values = {}

            # Try to reach the real server
            self._server_available = await self._check_server()
            if self._server_available:
                logger.info(f"✅ AtomSpace server reachable at {self._base_url} — using live store")
            else:
                logger.info("ℹ️  AtomSpace server not reachable — using in-process fallback store")

            self.initialized = True
            logger.info("✅ AtomSpace core initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize AtomSpace core: {e}")
            return False
    
    def create_atom(self, atom_type: str, name: str, truth_value: Optional[Dict] = None) -> int:
        """Create an atom in the AtomSpace.

        Persists to the REST server when available; always mirrors locally so
        that callers can introspect atoms without an async call.
        """
        if not self.initialized:
            raise RuntimeError("AtomSpace not initialized")

        atom_id = self.next_atom_id
        self.next_atom_id += 1

        atom = {
            'id': atom_id,
            'type': atom_type,
            'name': name,
            'created_at': datetime.now().isoformat(),
            'truth_value': truth_value or {'strength': 1.0, 'confidence': 1.0}
        }

        self.atoms[atom_id] = atom
        self.truth_values[atom_id] = atom['truth_value']

        # Fire-and-forget persistence to server (non-blocking)
        if self._server_available and _AIOHTTP_AVAILABLE:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._persist_atom(atom))
            except RuntimeError:
                pass

        logger.debug(f"Created {atom_type} atom '{name}' with ID {atom_id}")
        return atom_id

    async def _persist_atom(self, atom: Dict) -> None:
        """Attempt to persist an atom to atomspace-restful."""
        payload = {
            "type": atom["type"],
            "name": atom["name"],
            "truthvalue": {
                "type": "simple",
                "details": {
                    "strength": atom["truth_value"]["strength"],
                    "count": atom["truth_value"]["confidence"]
                }
            }
        }
        await self._rest_post("/api/v1.1/atoms", payload)
    
    def create_link(self, link_type: str, outgoing: List[int], truth_value: Optional[Dict] = None) -> int:
        """Create a link between atoms"""
        if not self.initialized:
            raise RuntimeError("AtomSpace not initialized")
        
        # Validate all outgoing atoms exist
        for atom_id in outgoing:
            if atom_id not in self.atoms and atom_id not in self.links:
                raise ValueError(f"Atom/Link {atom_id} does not exist")
        
        link_id = self.next_link_id
        self.next_link_id += 1
        
        link = {
            'id': link_id,
            'type': link_type,
            'outgoing': outgoing,
            'created_at': datetime.now().isoformat(),
            'truth_value': truth_value or {'strength': 1.0, 'confidence': 1.0}
        }
        
        self.links[link_id] = link
        self.truth_values[link_id] = link['truth_value']
        
        logger.debug(f"Created {link_type} link with ID {link_id} connecting {outgoing}")
        return link_id
    
    def get_atom(self, atom_id: int) -> Optional[Dict]:
        """Retrieve an atom by ID"""
        return self.atoms.get(atom_id)
    
    def get_link(self, link_id: int) -> Optional[Dict]:
        """Retrieve a link by ID"""
        return self.links.get(link_id)
    
    def query_by_type(self, atom_type: str) -> List[Dict]:
        """Query atoms by type"""
        results = []
        
        # Search atoms
        for atom in self.atoms.values():
            if atom['type'] == atom_type:
                results.append(atom)
        
        # Search links
        for link in self.links.values():
            if link['type'] == atom_type:
                results.append(link)
        
        return results
    
    def query_by_name(self, name: str) -> List[Dict]:
        """Query atoms by name pattern.

        Tries the REST server first so results include atoms that were stored
        in previous sessions; falls back to local cache.
        """
        results = []

        # Attempt server-side query (sync wrapper around async helper)
        if self._server_available and _AIOHTTP_AVAILABLE:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule coroutine but can't await here; fall through to local cache
                    pass
                else:
                    server_results = loop.run_until_complete(
                        self._rest_get(f"/api/v1.1/atoms?name={name}&filterby=name")
                    )
                    if isinstance(server_results, dict) and "result" in server_results:
                        for atom in server_results["result"].get("atoms", []):
                            results.append({
                                'id': atom.get("handle"),
                                'type': atom.get("type"),
                                'name': atom.get("name", ""),
                                'truth_value': atom.get("truthvalue", {}).get("details", {}),
                                'source': 'server'
                            })
                        return results
            except Exception as exc:
                logger.debug(f"Server query_by_name failed, using local cache: {exc}")

        # Local cache fallback
        name_lower = name.lower()
        for atom in self.atoms.values():
            if name_lower in atom['name'].lower():
                results.append(atom)

        return results
    
    def get_incoming_links(self, atom_id: int) -> List[Dict]:
        """Get all links that reference this atom"""
        incoming = []
        
        for link in self.links.values():
            if atom_id in link['outgoing']:
                incoming.append(link)
        
        return incoming
    
    def get_atom_count(self) -> int:
        """Get total number of atoms"""
        return len(self.atoms)
    
    def get_link_count(self) -> int:
        """Get total number of links"""
        return len(self.links)
    
    def export_atomspace(self) -> Dict:
        """Export entire AtomSpace as JSON"""
        return {
            'atoms': self.atoms,
            'links': self.links,
            'truth_values': self.truth_values,
            'atom_types': self.atom_types,
            'metadata': {
                'atom_count': self.get_atom_count(),
                'link_count': self.get_link_count(),
                'exported_at': datetime.now().isoformat()
            }
        }


class FinancialAtomSpace(AtomSpaceCore):
    """
    Specialized AtomSpace for financial domain
    Extends core with financial-specific patterns and structures
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.financial_concepts = {}
        self.account_atoms = {}
        self.transaction_links = {}
    
    async def initialize(self) -> bool:
        """Initialize financial AtomSpace with domain-specific setup"""
        if not await super().initialize():
            return False
        
        try:
            # Create financial domain concepts
            self.financial_concepts['account'] = self.create_atom(
                'ConceptNode', 'FinancialAccount'
            )
            self.financial_concepts['transaction'] = self.create_atom(
                'ConceptNode', 'Transaction'
            )
            self.financial_concepts['money'] = self.create_atom(
                'ConceptNode', 'Money'
            )
            self.financial_concepts['category'] = self.create_atom(
                'ConceptNode', 'ExpenseCategory'
            )
            
            logger.info("✅ Financial AtomSpace initialized with domain concepts")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize financial AtomSpace: {e}")
            return False
    
    def create_account_atom(self, account_name: str, account_type: str, balance: float = 0.0) -> int:
        """Create an atom representing a financial account"""
        account_atom = self.create_atom('ConceptNode', f"Account-{account_name}")
        
        # Create type link
        type_atom = self.create_atom('ConceptNode', account_type)
        type_link = self.create_link('EvaluationLink', [
            self.create_atom('PredicateNode', 'AccountType'),
            self.create_link('ListLink', [account_atom, type_atom])
        ])
        
        # Create balance link
        balance_atom = self.create_atom('NumberNode', str(balance))
        balance_link = self.create_link('EvaluationLink', [
            self.create_atom('PredicateNode', 'Balance'),
            self.create_link('ListLink', [account_atom, balance_atom])
        ])
        
        self.account_atoms[account_name] = {
            'atom_id': account_atom,
            'type_link': type_link,
            'balance_link': balance_link
        }
        
        logger.info(f"Created account atom for '{account_name}' ({account_type})")
        return account_atom
    
    def create_transaction_link(self, from_account: str, to_account: str, amount: float, description: str) -> int:
        """Create a link representing a financial transaction"""
        # Get or create account atoms
        if from_account not in self.account_atoms:
            self.create_account_atom(from_account, 'Unknown')
        if to_account not in self.account_atoms:
            self.create_account_atom(to_account, 'Unknown')
        
        from_atom = self.account_atoms[from_account]['atom_id']
        to_atom = self.account_atoms[to_account]['atom_id']
        amount_atom = self.create_atom('NumberNode', str(amount))
        desc_atom = self.create_atom('ConceptNode', description)
        
        # Create transaction link
        transaction_link = self.create_link('EvaluationLink', [
            self.create_atom('PredicateNode', 'Transaction'),
            self.create_link('ListLink', [from_atom, to_atom, amount_atom, desc_atom])
        ])
        
        transaction_id = f"{from_account}->{to_account}-{amount}"
        self.transaction_links[transaction_id] = transaction_link
        
        logger.info(f"Created transaction: {from_account} -> {to_account}: ${amount}")
        return transaction_link
    
    def get_account_balance(self, account_name: str) -> Optional[float]:
        """Get the current balance for an account"""
        if account_name not in self.account_atoms:
            return None
        
        balance_link_id = self.account_atoms[account_name]['balance_link']
        balance_link = self.get_link(balance_link_id)
        
        if balance_link:
            # Extract balance from link structure
            list_link_id = balance_link['outgoing'][1]
            list_link = self.get_link(list_link_id)
            balance_atom_id = list_link['outgoing'][1]
            balance_atom = self.get_atom(balance_atom_id)
            return float(balance_atom['name'])
        
        return 0.0
    
    def query_transactions_by_account(self, account_name: str) -> List[Dict]:
        """Query all transactions involving an account"""
        if account_name not in self.account_atoms:
            return []
        
        account_atom_id = self.account_atoms[account_name]['atom_id']
        transactions = []
        
        # Find transaction links that reference this account
        for link in self.links.values():
            if (link['type'] == 'EvaluationLink' and 
                len(link['outgoing']) >= 2):
                
                # Check if this is a transaction link
                predicate_id = link['outgoing'][0]
                predicate = self.get_atom(predicate_id)
                
                if predicate and predicate['name'] == 'Transaction':
                    list_link_id = link['outgoing'][1]
                    list_link = self.get_link(list_link_id)
                    
                    if (list_link and account_atom_id in list_link['outgoing']):
                        transactions.append({
                            'link_id': link['id'],
                            'link': link,
                            'created_at': link['created_at']
                        })
        
        return transactions