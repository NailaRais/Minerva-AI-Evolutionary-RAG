import pytest
import numpy as np
from minerva.retrieval.holographic import HolographicMemory

class TestHolographicMemory:
    def test_initialization(self):
        memory = HolographicMemory(compression_size=128)
        assert memory.compression_size == 128
        assert memory.basis is not None
        assert memory.basis.shape == (128, 768)
        
    def test_compression_decompression(self):
        memory = HolographicMemory(compression_size=64)
        test_vector = np.random.randn(768)
        
        compressed = memory.compress(test_vector)
        decompressed = memory.decompress(compressed)
        
        assert compressed.shape == (64,)
        assert decompressed.shape == (768,)
        # Should preserve direction (approximately)
        cosine_sim = np.dot(test_vector, decompressed) / (
            np.linalg.norm(test_vector) * np.linalg.norm(decompressed)
        )
        assert abs(cosine_sim) > 0.9  # Should be similar direction
        
    def test_similarity_search(self):
        memory = HolographicMemory(compression_size=32)
        query_vector = np.random.randn(768)
        test_vectors = [np.random.randn(768) for _ in range(10)]
        
        indices, similarities = memory.similarity_search(query_vector, test_vectors, top_k=3)
        
        assert len(indices) == 3
        assert len(similarities) == 3
        assert all(0 <= idx < len(test_vectors) for idx in indices)
        # Similarities should be in descending order
        assert similarities == sorted(similarities, reverse=True)
