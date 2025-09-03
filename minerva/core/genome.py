import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import zlib

@dataclass
class KnowledgeGene:
    """Represents a unit of knowledge in the evolutionary system."""
    id: str
    semantic_pattern: np.ndarray
    connections: Dict[str, float]
    strength: float = 1.0
    activation: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def evolve(self, mutation_rate: float) -> 'KnowledgeGene':
        """Apply evolutionary mutation to this gene."""
        new_pattern = self.semantic_pattern.copy()
        mask = np.random.random(len(new_pattern)) < mutation_rate
        new_pattern[mask] += np.random.normal(0, 0.1, np.sum(mask))
        new_pattern = new_pattern / np.linalg.norm(new_pattern)
        
        return KnowledgeGene(
            id=f"{self.id}_mut",
            semantic_pattern=new_pattern,
            connections=self.connections.copy(),
            strength=self.strength * 0.95,
            metadata=self.metadata.copy()
        )
    
    def to_compressed(self) -> bytes:
        """Convert gene to compressed binary representation."""
        data = {
            'id': self.id,
            'pattern': self.semantic_pattern.tolist(),
            'connections': self.connections,
            'strength': self.strength,
            'metadata': self.metadata
        }
        return zlib.compress(json.dumps(data).encode(), level=9)
    
    @classmethod
    def from_compressed(cls, data: bytes) -> 'KnowledgeGene':
        """Reconstruct gene from compressed binary."""
        decompressed = json.loads(zlib.decompress(data).decode())
        return cls(
            id=decompressed['id'],
            semantic_pattern=np.array(decompressed['pattern']),
            connections=decompressed['connections'],
            strength=decompressed['strength'],
            metadata=decompressed['metadata']
        )


class NeuralGenePool:
    """Main evolutionary knowledge repository."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.genes: Dict[str, KnowledgeGene] = {}
        self.graph = nx.DiGraph()
        self.embedder = None
        self._init_embedder()
    
    def _init_embedder(self):
        """Initialize the embedding model."""
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_gene(self, gene: KnowledgeGene):
        """Add a gene to the pool."""
        self.genes[gene.id] = gene
        self.graph.add_node(gene.id, gene=gene)
        
        # Add connections to other genes
        for other_id, strength in gene.connections.items():
            if other_id in self.genes:
                self.graph.add_edge(gene.id, other_id, weight=strength)
    
    def natural_selection(self):
        """Apply evolutionary pressure to the gene pool."""
        # Remove weak genes
        to_remove = [
            gene_id for gene_id, gene in self.genes.items() 
            if gene.strength < 0.1
        ]
        
        for gene_id in to_remove:
            del self.genes[gene_id]
            self.graph.remove_node(gene_id)
        
        # Strengthen frequently used genes
        for gene in self.genes.values():
            if gene.activation > 0.5:
                gene.strength = min(1.0, gene.strength * 1.1)
    
    def activate_genes(self, query: str) -> List[KnowledgeGene]:
        """Find genes relevant to the query."""
        query_embedding = self.embedder.encode([query])[0]
        activated_genes = []
        
        for gene in self.genes.values():
            similarity = np.dot(gene.semantic_pattern, query_embedding)
            gene.activation = max(0, similarity - 0.3)  # Activation threshold
            if gene.activation > 0:
                activated_genes.append(gene)
        
        return sorted(activated_genes, key=lambda x: x.activation, reverse=True)
    
    def evolve_activation(self, genes: List[KnowledgeGene]) -> List[KnowledgeGene]:
        """Evolve the activated genes through crossover and mutation."""
        new_generation = []
        
        for i, gene in enumerate(genes):
            # Mutation
            if np.random.random() < self.config['evolution']['mutation_rate']:
                new_generation.append(gene.evolve(0.1))
            
            # Crossover with other strong genes
            if i < len(genes) - 1 and np.random.random() < self.config['evolution']['crossover_rate']:
                child = self._crossover(gene, genes[i + 1])
                new_generation.append(child)
        
        return new_generation[:self.config['retrieval']['top_k']]
    
    def _crossover(self, gene1: KnowledgeGene, gene2: KnowledgeGene) -> KnowledgeGene:
        """Create a new gene through crossover of two parent genes."""
        # Blend semantic patterns
        alpha = np.random.random()
        new_pattern = alpha * gene1.semantic_pattern + (1 - alpha) * gene2.semantic_pattern
        new_pattern = new_pattern / np.linalg.norm(new_pattern)
        
        # Merge connections
        new_connections = gene1.connections.copy()
        for key, value in gene2.connections.items():
            new_connections[key] = max(value, new_connections.get(key, 0))
        
        return KnowledgeGene(
            id=f"{gene1.id}_{gene2.id}_child",
            semantic_pattern=new_pattern,
            connections=new_connections,
            strength=(gene1.strength + gene2.strength) / 2,
            metadata={**gene1.metadata, **gene2.metadata}
        )
