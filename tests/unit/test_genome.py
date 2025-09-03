import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import numpy as np
from minerva.core.genome import KnowledgeGene, NeuralGenePool

class TestKnowledgeGene:
    def test_gene_creation(self):
        # Create a gene pool to get the correct embedding dimension
        pool = NeuralGenePool({})
        pattern = np.random.randn(pool.embedding_dim)  # Use correct dimension
        
        gene = KnowledgeGene(
            id="test_gene",
            semantic_pattern=pattern,
            connections={"other_gene": 0.5},
            strength=0.8
        )
        
        assert gene.id == "test_gene"
        assert np.array_equal(gene.semantic_pattern, pattern)
        assert gene.connections == {"other_gene": 0.5}
        assert gene.strength == 0.8
    
    def test_gene_compression(self):
        pool = NeuralGenePool({})
        pattern = np.random.randn(pool.embedding_dim)  # Use correct dimension
        
        gene = KnowledgeGene(
            id="test_gene",
            semantic_pattern=pattern,
            connections={"other_gene": 0.5}
        )
        
        compressed = gene.to_compressed()
        reconstructed = KnowledgeGene.from_compressed(compressed)
        
        assert reconstructed.id == gene.id
        assert np.allclose(reconstructed.semantic_pattern, gene.semantic_pattern, atol=1e-6)
        assert reconstructed.connections == gene.connections

class TestNeuralGenePool:
    def test_pool_initialization(self):
        config = {"evolution": {"mutation_rate": 0.1}}
        pool = NeuralGenePool(config)
        
        assert pool.genes == {}
        assert pool.graph.number_of_nodes() == 0
        assert pool.embedder is not None
        assert hasattr(pool, 'embedding_dim')
        assert pool.embedding_dim == 384  # all-MiniLM-L6-v2 has 384 dimensions
    
    def test_gene_addition(self):
        pool = NeuralGenePool({})
        
        # Use the helper method to create a gene with correct dimensions
        gene = pool.create_test_gene("test_concept")
        
        pool.add_gene(gene)
        
        assert gene.id in pool.genes
        assert pool.graph.has_node(gene.id)
        assert "test_concept" in pool.genes[gene.id].metadata.get('text', '')
    
    def test_create_test_gene(self):
        pool = NeuralGenePool({})
        gene = pool.create_test_gene("test concept")
        
        assert gene.id.startswith("gene_")
        assert len(gene.semantic_pattern) == pool.embedding_dim
        assert gene.metadata['text'] == "test concept"
        assert gene.strength == 1.0
    
    def test_embedding_dimension(self):
        pool = NeuralGenePool({})
        dim = pool.get_embedding_dimension()
        
        assert dim == 384  # all-MiniLM-L6-v2 dimension
        assert dim == pool.embedding_dim
