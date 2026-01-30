"""
ElizaOS Enhanced Memory Management

Provides advanced memory storage, retrieval, and management capabilities
integrated with the OpenCog AtomSpace and GnuCash financial data.
"""

import asyncio
import logging
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import pickle

logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """Represents a single memory item"""
    id: str
    content: str
    content_type: str  # 'text', 'conversation', 'financial', 'document'
    timestamp: datetime
    source: str  # 'discord', 'telegram', 'api', 'financial_agent'
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    importance_score: float = 0.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None

class EnhancedMemoryManager:
    """Advanced memory management system for ElizaOS"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = Path(config.get('db_path', 'data/memory.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.connection = None
        self.embedding_provider = None
        self.cognitive_framework = None
        
        # Memory configuration
        self.max_memory_items = config.get('max_memory_items', 10000)
        self.retention_days = config.get('retention_days', 365)
        self.importance_threshold = config.get('importance_threshold', 0.5)
        
        # Memory categories
        self.memory_categories = {
            'conversation': 'User conversations and interactions',
            'financial': 'Financial data and analysis',
            'document': 'Document content and summaries',
            'system': 'System events and logs',
            'learning': 'Learned patterns and insights'
        }
    
    async def initialize(self) -> bool:
        """Initialize the memory management system"""
        try:
            await self._setup_database()
            logger.info("✅ Enhanced Memory Manager initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize memory manager: {e}")
            return False
    
    async def _setup_database(self):
        """Setup SQLite database for memory storage"""
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row
        
        # Create tables
        cursor = self.connection.cursor()
        
        # Main memories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_type TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                source TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB,
                importance_score REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                last_accessed DATETIME
            )
        ''')
        
        # Memory relationships table (for linking related memories)  
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_memory_id TEXT,
                target_memory_id TEXT,
                relationship_type TEXT,
                strength REAL DEFAULT 1.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_memory_id) REFERENCES memories (id),
                FOREIGN KEY (target_memory_id) REFERENCES memories (id)
            )
        ''')
        
        # Memory tags table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT,
                tag TEXT,
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (memory_id) REFERENCES memories (id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories (timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_content_type ON memories (content_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_source ON memories (source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories (importance_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags (tag)')
        
        self.connection.commit()
    
    def set_embedding_provider(self, provider):
        """Set the embedding provider for vector similarity"""
        self.embedding_provider = provider
    
    def set_cognitive_framework(self, framework):
        """Set the cognitive framework for enhanced processing"""
        self.cognitive_framework = framework
    
    async def store_memory(self, content: str, content_type: str = 'text', 
                          source: str = 'unknown', metadata: Dict[str, Any] = None) -> str:
        """Store a new memory item"""
        try:
            # Generate unique ID
            memory_id = self._generate_memory_id(content, source)
            
            # Calculate importance score
            importance_score = await self._calculate_importance(content, content_type, metadata or {})
            
            # Generate embedding if provider available
            embedding = None
            if self.embedding_provider:
                try:
                    embedding_vec = await self.embedding_provider.generate_embedding(content)
                    embedding = pickle.dumps(embedding_vec)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")
            
            # Create memory item
            memory_item = MemoryItem(
                id=memory_id,
                content=content,
                content_type=content_type,
                timestamp=datetime.now(),
                source=source,
                metadata=metadata or {},
                embedding=embedding,
                importance_score=importance_score
            )
            
            # Store in database
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO memories 
                (id, content, content_type, timestamp, source, metadata, embedding, importance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory_item.id,
                memory_item.content,
                memory_item.content_type,
                memory_item.timestamp,
                memory_item.source,
                json.dumps(memory_item.metadata),
                embedding,
                memory_item.importance_score
            ))
            self.connection.commit()
            
            # Auto-generate tags
            await self._auto_tag_memory(memory_id, content, content_type)
            
            # Link to related memories
            await self._link_related_memories(memory_id, content, content_type)
            
            logger.info(f"📝 Stored memory: {memory_id} (importance: {importance_score:.2f})")
            return memory_id
            
        except Exception as e:
            logger.error(f"❌ Failed to store memory: {e}")
            return ""
    
    async def retrieve_memories(self, query: str = None, content_type: str = None,
                               source: str = None, limit: int = 10,
                               similarity_threshold: float = 0.7) -> List[MemoryItem]:
        """Retrieve memories based on query and filters"""
        try:
            cursor = self.connection.cursor()
            
            # Build SQL query
            sql_parts = ["SELECT * FROM memories WHERE 1=1"]
            params = []
            
            if content_type:
                sql_parts.append("AND content_type = ?")
                params.append(content_type)
            
            if source:
                sql_parts.append("AND source = ?")
                params.append(source)
            
            # Text search if query provided
            if query:
                sql_parts.append("AND (content LIKE ? OR id IN (SELECT memory_id FROM memory_tags WHERE tag LIKE ?))")
                params.extend([f"%{query}%", f"%{query}%"])
            
            sql_parts.append("ORDER BY importance_score DESC, timestamp DESC LIMIT ?")
            params.append(limit)
            
            sql = " ".join(sql_parts)
            cursor.execute(sql, params)
            
            memories = []
            for row in cursor.fetchall():
                memory = self._row_to_memory_item(row)
                
                # Update access tracking
                await self._update_access_tracking(memory.id)
                
                memories.append(memory)
            
            # If we have embeddings and a query, do semantic similarity ranking
            if query and self.embedding_provider and memories:
                memories = await self._rank_by_similarity(query, memories, similarity_threshold)
            
            logger.info(f"🔍 Retrieved {len(memories)} memories for query: {query}")
            return memories
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve memories: {e}")
            return []
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        """Get a specific memory by ID"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            
            if row:
                await self._update_access_tracking(memory_id)
                return self._row_to_memory_item(row)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get memory by ID: {e}")
            return None
    
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing memory"""
        try:
            cursor = self.connection.cursor()
            
            # Build update query
            update_parts = []
            params = []
            
            allowed_fields = ['content', 'content_type', 'metadata', 'importance_score']
            for field, value in updates.items():
                if field in allowed_fields:
                    if field == 'metadata':
                        value = json.dumps(value)
                    update_parts.append(f"{field} = ?")
                    params.append(value)
            
            if not update_parts:
                return False
            
            params.append(memory_id)
            sql = f"UPDATE memories SET {', '.join(update_parts)} WHERE id = ?"
            
            cursor.execute(sql, params)
            self.connection.commit()
            
            logger.info(f"📝 Updated memory: {memory_id}")
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to update memory: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory and its relationships"""
        try:
            cursor = self.connection.cursor()
            
            # Delete relationships
            cursor.execute("DELETE FROM memory_relationships WHERE source_memory_id = ? OR target_memory_id = ?", 
                          (memory_id, memory_id))
            
            # Delete tags
            cursor.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))
            
            # Delete memory
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            
            self.connection.commit()
            
            logger.info(f"🗑️ Deleted memory: {memory_id}")
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to delete memory: {e}")
            return False
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            cursor = self.connection.cursor()
            
            # Total memories
            cursor.execute("SELECT COUNT(*) FROM memories")
            total_memories = cursor.fetchone()[0]
            
            # Memories by type
            cursor.execute("SELECT content_type, COUNT(*) FROM memories GROUP BY content_type")
            by_type = dict(cursor.fetchall())
            
            # Memories by source
            cursor.execute("SELECT source, COUNT(*) FROM memories GROUP BY source")
            by_source = dict(cursor.fetchall())
            
            # Average importance
            cursor.execute("SELECT AVG(importance_score) FROM memories")
            avg_importance = cursor.fetchone()[0] or 0
            
            # Recent activity (last 24 hours)
            yesterday = datetime.now() - timedelta(days=1)
            cursor.execute("SELECT COUNT(*) FROM memories WHERE timestamp > ?", (yesterday,))
            recent_memories = cursor.fetchone()[0]
            
            return {
                'total_memories': total_memories,
                'by_type': by_type,
                'by_source': by_source,
                'average_importance': round(avg_importance, 2),
                'recent_memories_24h': recent_memories,
                'max_capacity': self.max_memory_items,
                'capacity_used_percent': round((total_memories / self.max_memory_items) * 100, 1)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get memory statistics: {e}")
            return {}
    
    async def cleanup_old_memories(self) -> int:
        """Clean up old, low-importance memories"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            cursor = self.connection.cursor()
            
            # Delete old, low-importance memories
            cursor.execute('''
                DELETE FROM memories 
                WHERE timestamp < ? AND importance_score < ? AND access_count < 2
            ''', (cutoff_date, self.importance_threshold))
            
            deleted_count = cursor.rowcount
            
            # Clean up orphaned relationships and tags
            cursor.execute('''
                DELETE FROM memory_relationships 
                WHERE source_memory_id NOT IN (SELECT id FROM memories)
                   OR target_memory_id NOT IN (SELECT id FROM memories)
            ''')
            
            cursor.execute('''
                DELETE FROM memory_tags 
                WHERE memory_id NOT IN (SELECT id FROM memories)
            ''')
            
            self.connection.commit()
            
            logger.info(f"🧹 Cleaned up {deleted_count} old memories")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup memories: {e}")
            return 0
    
    async def get_related_memories(self, memory_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get memories related to a specific memory"""
        try:
            cursor = self.connection.cursor()
            
            # Get related memories through relationships
            cursor.execute('''
                SELECT m.*, r.relationship_type, r.strength
                FROM memories m
                JOIN memory_relationships r ON (
                    (r.source_memory_id = ? AND r.target_memory_id = m.id) OR
                    (r.target_memory_id = ? AND r.source_memory_id = m.id)
                )
                ORDER BY r.strength DESC, m.importance_score DESC
                LIMIT ?
            ''', (memory_id, memory_id, limit))
            
            related = []
            for row in cursor.fetchall():
                memory = self._row_to_memory_item(row)
                related.append({
                    'memory': memory,
                    'relationship_type': row['relationship_type'],
                    'strength': row['strength']
                })
            
            return related
            
        except Exception as e:
            logger.error(f"❌ Failed to get related memories: {e}")
            return []
    
    # Private helper methods
    
    def _generate_memory_id(self, content: str, source: str) -> str:
        """Generate unique memory ID"""
        hash_input = f"{content[:100]}{source}{datetime.now().isoformat()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    async def _calculate_importance(self, content: str, content_type: str, metadata: Dict) -> float:
        """Calculate importance score for memory"""
        score = 0.5  # Base score
        
        # Content type weights
        type_weights = {
            'financial': 0.8,
            'conversation': 0.6,
            'document': 0.7,
            'system': 0.3,
            'learning': 0.9
        }
        score *= type_weights.get(content_type, 0.5)
        
        # Length factor (longer content often more important)
        if len(content) > 200:
            score += 0.1
        if len(content) > 500:
            score += 0.1
        
        # Financial context boost
        financial_keywords = ['budget', 'money', 'transaction', 'account', 'balance', 'payment']
        if any(keyword in content.lower() for keyword in financial_keywords):
            score += 0.2
        
        # User interaction boost
        if metadata.get('user_id') or metadata.get('author'):
            score += 0.1
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, score))
    
    async def _auto_tag_memory(self, memory_id: str, content: str, content_type: str):
        """Automatically generate tags for memory"""
        try:
            tags = set()
            
            # Content type tag
            tags.add(content_type)
            
            # Extract financial tags
            financial_terms = ['budget', 'expense', 'income', 'transaction', 'account', 'balance']
            for term in financial_terms:
                if term in content.lower():
                    tags.add(f'financial_{term}')
            
            # Extract entity tags (simple keyword matching)
            common_entities = ['discord', 'telegram', 'openai', 'claude', 'gemini']
            for entity in common_entities:
                if entity in content.lower():
                    tags.add(entity)
            
            # Store tags
            cursor = self.connection.cursor()
            for tag in tags:
                cursor.execute('''
                    INSERT OR IGNORE INTO memory_tags (memory_id, tag, confidence)
                    VALUES (?, ?, ?)
                ''', (memory_id, tag, 0.8))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error auto-tagging memory: {e}")
    
    async def _link_related_memories(self, memory_id: str, content: str, content_type: str):
        """Find and link related memories"""
        try:
            # Simple similarity based on shared tags
            cursor = self.connection.cursor()
            
            # Find memories with shared tags
            cursor.execute('''
                SELECT DISTINCT m.id, COUNT(*) as shared_tags
                FROM memories m
                JOIN memory_tags mt1 ON m.id = mt1.memory_id
                JOIN memory_tags mt2 ON mt1.tag = mt2.tag
                WHERE mt2.memory_id = ? AND m.id != ?
                GROUP BY m.id
                HAVING shared_tags >= 2
                ORDER BY shared_tags DESC
                LIMIT 5
            ''', (memory_id, memory_id))
            
            for row in cursor.fetchall():
                related_id = row['id']
                strength = min(1.0, row['shared_tags'] / 5.0)  # Normalize strength
                
                # Create bidirectional relationship
                cursor.execute('''
                    INSERT OR IGNORE INTO memory_relationships 
                    (source_memory_id, target_memory_id, relationship_type, strength)
                    VALUES (?, ?, ?, ?)
                ''', (memory_id, related_id, 'similar_tags', strength))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error linking related memories: {e}")
    
    async def _update_access_tracking(self, memory_id: str):
        """Update access tracking for memory"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                UPDATE memories 
                SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (memory_id,))
            self.connection.commit()
        except Exception as e:
            logger.error(f"Error updating access tracking: {e}")
    
    def _row_to_memory_item(self, row) -> MemoryItem:
        """Convert database row to MemoryItem"""
        embedding = None
        if row['embedding']:
            try:
                embedding = pickle.loads(row['embedding'])
            except:
                pass
        
        return MemoryItem(
            id=row['id'],
            content=row['content'],
            content_type=row['content_type'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            source=row['source'],
            metadata=json.loads(row['metadata'] or '{}'),
            embedding=embedding,
            importance_score=row['importance_score'],
            access_count=row['access_count'],
            last_accessed=datetime.fromisoformat(row['last_accessed']) if row['last_accessed'] else None
        )
    
    async def _rank_by_similarity(self, query: str, memories: List[MemoryItem], 
                                 threshold: float) -> List[MemoryItem]:
        """Rank memories by semantic similarity to query"""
        try:
            if not self.embedding_provider:
                return memories
            
            # Generate query embedding
            query_embedding = await self.embedding_provider.generate_embedding(query)
            
            # Calculate similarities
            similarities = []
            for memory in memories:
                if memory.embedding:
                    similarity = self._cosine_similarity(query_embedding, memory.embedding)
                    if similarity >= threshold:
                        similarities.append((memory, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return [memory for memory, _ in similarities]
            
        except Exception as e:
            logger.error(f"Error ranking by similarity: {e}")
            return memories
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0
    
    async def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


class AtomSpaceMemoryBackend:
    """
    AtomSpace-based memory backend for ElizaOS
    Provides hypergraph memory storage using OpenCog AtomSpace
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.atomspace = None
        self.initialized = False
        
        # Import AtomSpace bindings if available
        try:
            from opencog.atomspace import AtomSpace, types
            from opencog.type_constructors import *
            self.AtomSpace = AtomSpace
            self.types = types
            self.atomspace_available = True
        except ImportError:
            logger.warning("OpenCog AtomSpace not available. Using fallback storage.")
            self.atomspace_available = False
    
    async def initialize(self) -> bool:
        """Initialize AtomSpace memory backend"""
        try:
            if self.atomspace_available:
                self.atomspace = self.AtomSpace()
                logger.info("✅ AtomSpace memory backend initialized")
            else:
                logger.warning("⚠️ AtomSpace not available, using fallback")
            
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize AtomSpace backend: {e}")
            return False
    
    async def store_memory_as_atoms(self, memory_item: MemoryItem) -> bool:
        """Store a memory item as atoms in AtomSpace"""
        if not self.initialized or not self.atomspace_available:
            return False
        
        try:
            from opencog.type_constructors import ConceptNode, PredicateNode, InheritanceLink
            
            # Create concept node for the memory
            memory_node = ConceptNode(f"Memory:{memory_item.id}")
            
            # Create nodes for content type and source
            content_type_node = ConceptNode(f"ContentType:{memory_item.content_type}")
            source_node = ConceptNode(f"Source:{memory_item.source}")
            
            # Create inheritance links
            InheritanceLink(memory_node, content_type_node)
            InheritanceLink(memory_node, source_node)
            
            # Store content as predicate
            content_pred = PredicateNode("has_content")
            # Note: In real implementation, content would be stored in a proper format
            
            # Store metadata as atoms
            for key, value in memory_item.metadata.items():
                meta_pred = PredicateNode(f"meta:{key}")
                meta_value = ConceptNode(str(value))
                # Create evaluation link for metadata
            
            # Store importance score
            memory_node.tv = self.atomspace.create_truth_value(
                memory_item.importance_score, 
                0.9  # confidence
            )
            
            logger.debug(f"Stored memory {memory_item.id} in AtomSpace")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory in AtomSpace: {e}")
            return False
    
    async def retrieve_memories_from_atomspace(self, pattern: Dict) -> List[Dict]:
        """Retrieve memories from AtomSpace using pattern matching"""
        if not self.initialized or not self.atomspace_available:
            return []
        
        try:
            from opencog.bindlink import execute_atom
            from opencog.type_constructors import BindLink, VariableNode, ConceptNode
            
            # Build pattern matching query
            # This is a simplified example - real implementation would be more sophisticated
            results = []
            
            # Get all memory nodes
            memory_nodes = self.atomspace.get_atoms_by_type(self.types.ConceptNode)
            
            for node in memory_nodes:
                if node.name.startswith("Memory:"):
                    memory_data = {
                        'id': node.name.replace("Memory:", ""),
                        'importance': node.tv.mean if hasattr(node, 'tv') else 0.0
                    }
                    results.append(memory_data)
            
            logger.debug(f"Retrieved {len(results)} memories from AtomSpace")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve from AtomSpace: {e}")
            return []
    
    async def create_hypergraph_relationship(self, source_id: str, target_id: str, 
                                            relationship_type: str, strength: float = 1.0):
        """Create hypergraph relationship between memories"""
        if not self.initialized or not self.atomspace_available:
            return False
        
        try:
            from opencog.type_constructors import ConceptNode, PredicateNode, EvaluationLink, ListLink
            
            # Create nodes for source and target memories
            source_node = ConceptNode(f"Memory:{source_id}")
            target_node = ConceptNode(f"Memory:{target_id}")
            
            # Create predicate for relationship type
            relationship_pred = PredicateNode(relationship_type)
            
            # Create evaluation link with strength
            eval_link = EvaluationLink(
                relationship_pred,
                ListLink(source_node, target_node)
            )
            
            # Set strength as truth value
            eval_link.tv = self.atomspace.create_truth_value(strength, 0.9)
            
            logger.debug(f"Created hypergraph link: {source_id} --[{relationship_type}]--> {target_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create hypergraph relationship: {e}")
            return False
    
    async def query_hypergraph(self, query_pattern: str) -> List[Dict]:
        """Query hypergraph using pattern matching"""
        if not self.initialized or not self.atomspace_available:
            return []
        
        try:
            # This would use OpenCog's pattern matcher for sophisticated queries
            # Simplified implementation for now
            results = []
            
            logger.debug(f"Executed hypergraph query: {query_pattern}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to query hypergraph: {e}")
            return []


class HybridMemoryManager(EnhancedMemoryManager):
    """
    Hybrid memory manager combining SQL storage with AtomSpace hypergraph
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.atomspace_backend = AtomSpaceMemoryBackend(config)
        self.use_atomspace = config.get('use_atomspace', True)
    
    async def initialize(self) -> bool:
        """Initialize both SQL and AtomSpace backends"""
        sql_init = await super().initialize()
        
        if self.use_atomspace:
            atomspace_init = await self.atomspace_backend.initialize()
            self.use_atomspace = atomspace_init
        
        return sql_init
    
    async def store_memory(self, content: str, content_type: str = 'text',
                          source: str = 'default', metadata: Optional[Dict] = None,
                          importance_score: float = 0.5) -> str:
        """Store memory in both SQL and AtomSpace"""
        # Store in SQL first
        memory_id = await super().store_memory(content, content_type, source, metadata, importance_score)
        
        # Also store in AtomSpace if available
        if memory_id and self.use_atomspace:
            memory_item = await self.get_memory_by_id(memory_id)
            if memory_item:
                await self.atomspace_backend.store_memory_as_atoms(memory_item)
        
        return memory_id
    
    async def create_memory_relationship(self, source_id: str, target_id: str,
                                        relationship_type: str, strength: float = 1.0):
        """Create relationship in both SQL and AtomSpace"""
        # Create in SQL
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO memory_relationships 
                (source_memory_id, target_memory_id, relationship_type, strength)
                VALUES (?, ?, ?, ?)
            ''', (source_id, target_id, relationship_type, strength))
            self.connection.commit()
        except Exception as e:
            logger.error(f"Failed to create SQL relationship: {e}")
        
        # Create in AtomSpace
        if self.use_atomspace:
            await self.atomspace_backend.create_hypergraph_relationship(
                source_id, target_id, relationship_type, strength
            )
    
    async def query_hypergraph_memories(self, pattern: str) -> List[MemoryItem]:
        """Query memories using hypergraph pattern matching"""
        if not self.use_atomspace:
            # Fallback to regular query
            return await self.retrieve_memories(query=pattern)
        
        # Query AtomSpace
        atom_results = await self.atomspace_backend.query_hypergraph(pattern)
        
        # Convert atom results to memory items
        memories = []
        for result in atom_results:
            memory_id = result.get('id')
            if memory_id:
                memory = await self.get_memory_by_id(memory_id)
                if memory:
                    memories.append(memory)
        
        return memories
            logger.info("📴 Memory manager connection closed")