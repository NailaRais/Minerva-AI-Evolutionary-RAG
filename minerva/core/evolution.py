import numpy as np
from typing import List, Optional
from .genome import KnowledgeGene, NeuralGenePool

class EvolutionaryOptimizer:
    """Manages the evolutionary process for the knowledge genome."""
    
    def __init__(self, gene_pool: NeuralGenePool):
        self.gene_pool = gene_pool
        self.generation = 0
    
    def run_generation(self, queries: Optional[List[str]] = None):
        """Run one generation of evolutionary optimization."""
        self.generation += 1
        
        # If no queries provided, use random activation
        if queries is None:
            queries = self._generate_random_queries()
        
        # Process each query through evolutionary cycle
        for query in queries:
            try:
                activated = self.gene_pool.activate_genes(query)
                
                # Get depth from config with default value
                depth = self.gene_pool.config.get('evolution', {}).get('depth', 3)
                for _ in range(depth):
                    if activated:  # Only evolve if there are activated genes
                        activated = self.gene_pool.evolve_activation(activated)
            except Exception as e:
                # Log error but continue with other queries
                print(f"Error processing query '{query}': {e}")
                continue
        
        # Apply natural selection
        self.gene_pool.natural_selection()
    
    def _generate_random_queries(self) -> List[str]:
        """Generate random queries to maintain genetic diversity."""
        # Use more diverse queries for better genetic diversity
        return [
            "artificial intelligence", "machine learning", "deep learning",
            "natural language processing", "computer vision", "data science",
            "neural networks", "quantum computing", "robotics", "ethics in AI"
        ]
    
    def optimize_memory(self, target_size: int, max_iterations: int = 100):
        """Optimize memory usage to meet target size."""
        current_size = self._estimate_memory_usage()
        iterations = 0
        
        while current_size > target_size and iterations < max_iterations:
            iterations += 1
            
            # Increase evolutionary pressure to reduce population
            evolution_config = self.gene_pool.config.setdefault('evolution', {})
            old_rate = evolution_config.get('mutation_rate', 0.1)
            evolution_config['mutation_rate'] = min(0.5, old_rate * 1.1)
            
            self.run_generation()
            current_size = self._estimate_memory_usage()
            
            # Early stopping if we're not making progress
            if iterations > 10 and current_size >= self._estimate_memory_usage():
                break
    
    def _estimate_memory_usage(self) -> int:
        """Estimate current memory usage in bytes."""
        size = 0
        for gene in self.gene_pool.genes.values():
            try:
                size += len(gene.to_compressed())
            except Exception as e:
                print(f"Error estimating memory for gene {gene.id}: {e}")
                # Add approximate size for failed genes
                size += 1000  # Approximate size in bytes
        return size
    
    def get_stats(self) -> dict:
        """Get statistics about the current state."""
        return {
            'generation': self.generation,
            'gene_count': len(self.gene_pool.genes),
            'memory_usage_bytes': self._estimate_memory_usage(),
            'memory_usage_mb': self._estimate_memory_usage() / (1024 * 1024),
            'average_gene_strength': np.mean([gene.strength for gene in self.gene_pool.genes.values()]) 
                if self.gene_pool.genes else 0
        }
