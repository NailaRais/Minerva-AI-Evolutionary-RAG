import numpy as np
from typing import List

class HolographicMemory:
    """Implements holographic memory compression for semantic patterns."""
    
    def __init__(self, compression_size: int = 128):
        self.compression_size = compression_size
        self.basis = None
        self._init_basis()
    
    def _init_basis(self):
        """Initialize the basis functions for holographic compression."""
        # Use random orthogonal basis (could be learned instead)
        random_basis = np.random.randn(self.compression_size, 768)
        q, r = np.linalg.qr(random_basis.T)
        self.basis = q.T
    
    def compress(self, vector: np.ndarray) -> np.ndarray:
        """Compress a vector using holographic projection."""
        projection = self.basis @ vector
        return projection / np.linalg.norm(projection)
    
    def decompress(self, compressed: np.ndarray) -> np.ndarray:
        """Decompress a vector from holographic representation."""
        return self.basis.T @ compressed
    
    def similarity_search(self, query: np.ndarray, vectors: List[np.ndarray], top_k: int = 5):
        """Efficient similarity search using compressed representations."""
        compressed_query = self.compress(query)
        similarities = []
        
        for vec in vectors:
            compressed_vec = self.compress(vec)
            similarity = np.dot(compressed_query, compressed_vec)
            similarities.append(similarity)
        
        # Get top_k indices
        indices = np.argsort(similarities)[-top_k:][::-1]
        return indices, [similarities[i] for i in indices]
