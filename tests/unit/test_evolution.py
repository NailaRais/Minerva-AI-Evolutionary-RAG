import pytest
import numpy as np
from minerva.core.evolution import EvolutionaryOptimizer
from minerva.core.genome import NeuralGenePool, KnowledgeGene

class TestEvolutionaryOptimizer:
    def test_optimizer_initialization(self):
        config = {"evolution": {"mutation_rate": 0.1}}
        gene_pool = NeuralGenePool(config)
        optimizer = EvolutionaryOptimizer(gene_pool)
        
        assert optimizer.gene_pool == gene_pool
        assert optimizer.generation == 0
        
    def test_run_generation(self):
        config = {"evolution": {"mutation_rate": 0.1, "depth": 1}}
        gene_pool = NeuralGenePool(config)
        optimizer = EvolutionaryOptimizer(gene_pool)
        
        # Add some test genes
        for i in range(5):
            gene = KnowledgeGene(
                id=f"test_gene_{i}",
                semantic_pattern=np.random.randn(768),
                connections={}
            )
            gene_pool.add_gene(gene)
        
        initial_count = len(gene_pool.genes)
        optimizer.run_generation(["test query"])
        
        # Should have processed genes
        assert optimizer.generation == 1
        # Gene count might change due to evolution
        assert len(gene_pool.genes) <= initial_count + 2  # Allow for some mutation
        
    def test_optimize_memory(self):
        config = {"evolution": {"mutation_rate": 0.1}}
        gene_pool = NeuralGenePool(config)
        optimizer = EvolutionaryOptimizer(gene_pool)
        
        # Add many genes to exceed limit
        for i in range(100):
            gene = KnowledgeGene(
                id=f"test_gene_{i}",
                semantic_pattern=np.random.randn(768),
                connections={}
            )
            gene_pool.add_gene(gene)
        
       
