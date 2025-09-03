import numpy as np
from typing import List
from .genome import KnowledgeGene, NeuralGenePool

class EvolutionaryOptimizer:
    """Manages the evolutionary process for the knowledge genome."""
    
    def __init__(self, gene_pool: NeuralGenePool):
        self.gene_pool = gene_pool
        self.generation = 0
    
    def run_generation(self, queries: List[str] = None):
        """Run one generation of evolutionary optimization."""
        self.generation += 1
        
        # If no queries provided, use random activation
        if queries is None:
            queries = self._generate_random_queries()
        
        # Process each query through evolutionary cycle
        for query in queries:
            activated = self.gene_pool.activate_genes(query)
            
            for _ in range(self.gene_pool.config['evolution']['depth']):
                activated = self.gene_pool.evolve_activation(activated)
        
        # Apply natural selection
        self.gene_pool.natural_selection()
    
    def _generate_random_queries(self) -> List[str]:
        """Generate random queries to maintain genetic diversity."""
        # This would be enhanced with actual content from genes
        return ["science", "technology", "philosophy", "art", "nature"]
    
    def optimize_memory(self, target_size: int):
        """Optimize memory usage to meet target size."""
        current_size = self._estimate_memory_usage()
        
        while current_size > target_size:
            # Increase evolutionary pressure to reduce population
            old_rate = self.gene_pool.config['evolution']['mutation_rate']
            self.gene_pool.config['evolution']['mutation_rate'] = min(0.5, old_rate * 1.1)
            
            self.run_generation()
            current_size = self._estimate_memory_usage()
    
    def _estimate_memory_usage(self) -> int:
        """Estimate current memory usage in bytes."""
        size = 0
        for gene in self.gene_pool.genes.values():
            size += len(gene.to_compressed())
        return size
