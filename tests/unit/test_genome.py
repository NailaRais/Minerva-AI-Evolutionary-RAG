import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import numpy as np
from minerva.core.genome import KnowledgeGene, NeuralGenePool

class TestKnowledgeGene:
    def test_gene_creation(self):
        pattern = np.random.randn(768)
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
        pattern = np.random.randn(768)
        gene = KnowledgeGene(
            id="test_gene",
            semantic_pattern=pattern,
            connections={"other_gene": 0.5}
        )
        
        compressed = gene.to_compressed()
        reconstructed = KnowledgeGene.from_compressed(compressed)
        
        assert reconstructed.id == gene.id
        assert np.allclose(reconstructed.semantic_pattern, gene.semantic_pattern)
        assert reconstructed.connections == gene.connections

class TestNeuralGenePool:
    def test_pool_initialization(self):
        config = {"evolution": {"mutation_rate": 0.1}}
        pool = NeuralGenePool(config)
        
        assert pool.genes == {}
        assert pool.graph.number_of_nodes() == 0
        assert pool.embedder is not None
    
    def test_gene_addition(self):
        pool = NeuralGenePool({})
        pattern = np.random.randn(768)
        
        gene = KnowledgeGene(
            id="test_gene",
            semantic_pattern=pattern,
            connections={}
        )
        
        pool.add_gene(gene)
        
        assert "test_gene" in pool.genes
        assert pool.graph.has_node("test_gene")
