import numpy as np
from typing import List, Optional

class HolographicMemory:
    """Implements holographic memory compression for semantic patterns."""
    
    def __init__(self, compression_size: int = 128, embedding_dim: int = 384):
        self.compression_size = compression_size
        self.embedding_dim = embedding_dim  # all-MiniLM-L6-v2 has 384 dimensions
        self.basis = None
        self._init_basis()
    
    def _init_basis(self):
        """Initialize the basis functions for holographic compression."""
        # Use random orthogonal basis with proper dimensions
        random_basis = np.random.randn(self.compression_size, self.embedding_dim)
        q, r = np.linalg.qr(random_basis.T)
        self.basis = q.T
    
    def compress(self, vector: np.ndarray) -> np.ndarray:
        """Compress a vector using holographic projection."""
        if len(vector) != self.embedding_dim:
            raise ValueError(f"Vector dimension {len(vector)} doesn't match expected {self.embedding_dim}")
        
        projection = self.basis @ vector
        norm = np.linalg.norm(projection)
        if norm == 0:
            return projection
        return projection / norm
    
    def decompress(self, compressed: np.ndarray) -> np.ndarray:
        """Decompress a vector from holographic representation."""
        if len(compressed) != self.compression_size:
            raise ValueError(f"Compressed dimension {len(compressed)} doesn't match expected {self.compression_size}")
        
        return self.basis.T @ compressed
    
    def similarity_search(self, query: np.ndarray, vectors: List[np.ndarray], top_k: int = 5):
        """Efficient similarity search using compressed representations."""
        if len(query) != self.embedding_dim:
            raise ValueError(f"Query dimension {len(query)} doesn't match expected {self.embedding_dim}")
        
        compressed_query = self.compress(query)
        similarities = []
        valid_vectors = []
        
        for i, vec in enumerate(vectors):
            if len(vec) != self.embedding_dim:
                continue
            try:
                compressed_vec = self.compress(vec)
                similarity = np.dot(compressed_query, compressed_vec)
                similarities.append(similarity)
                valid_vectors.append(i)
            except Exception as e:
                print(f"Error processing vector {i}: {e}")
                continue
        
        if not similarities:
            return [], []
        
        # Get top_k indices from valid vectors
        indices = np.argsort(similarities)[-top_k:][::-1]
        original_indices = [valid_vectors[i] for i in indices]
        return original_indices, [similarities[i] for i in indices]
    
    def batch_compress(self, vectors: List[np.ndarray]) -> List[np.ndarray]:
        """Compress multiple vectors efficiently."""
        compressed_vectors = []
        for vec in vectors:
            if len(vec) == self.embedding_dim:
                compressed_vectors.append(self.compress(vec))
        return compressed_vectors
    
    def get_compression_ratio(self) -> float:
        """Get the compression ratio."""
        return self.embedding_dim / self.compression_size
    
    def reconstruction_accuracy(self, original: np.ndarray, compressed: Optional[np.ndarray] = None) -> float:
        """Calculate reconstruction accuracy."""
        if compressed is None:
            compressed = self.compress(original)
        
        reconstructed = self.decompress(compressed)
        
        # Calculate cosine similarity between original and reconstructed
        norm_original = np.linalg.norm(original)
        norm_reconstructed = np.linalg.norm(reconstructed)
        
        if norm_original == 0 or norm_reconstructed == 0:
            return 0.0
        
        cosine_sim = np.dot(original, reconstructed) / (norm_original * norm_reconstructed)
        return float(cosine_sim)
