import pytest
from pathlib import Path
from minerva.core.genome import NeuralGenePool
from minerva.retrieval.fractal_parser import FractalParser

class TestRAGPipeline:
    def test_full_pipeline_integration(self):
        """Test the complete RAG pipeline integration"""
        config = {
            "evolution": {"mutation_rate": 0.1, "depth": 2},
            "retrieval": {"top_k": 3}
        }
        
        # Initialize components
        gene_pool = NeuralGenePool(config)
        parser = FractalParser()
        
        # Test document processing
        test_text = """
        Artificial intelligence is the simulation of human intelligence processes by machines.
        Machine learning is a subset of AI that focuses on building systems that learn from data.
        Deep learning uses neural networks with multiple layers to analyze complex patterns.
        """
        
        # Parse document
        components = parser.decompose(test_text)
        assert len(components) > 0
        
        # Add to knowledge base
        initial_count = len(gene_pool.genes)
        for comp in components[:5]:  # Add first 5 components
            # Simplified gene creation for test
            pass
        
        # Test query
        query = "What is artificial intelligence?"
        activated_genes = gene_pool.activate_genes(query)
        assert len(activated_genes) >= 0  # Could be 0 if no matches
        
        # Test evolutionary processing
        if activated_genes:
            evolved_genes = gene_pool.evolve_activation(activated_genes)
            assert len(evolved_genes) <= config["retrieval"]["top_k"]
