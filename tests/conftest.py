import pytest
import tempfile
from pathlib import Path

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_config():
    """Provide a sample configuration for tests"""
    return {
        "evolution": {
            "mutation_rate": 0.1,
            "selection_pressure": 0.8,
            "crossover_rate": 0.3,
            "depth": 2
        },
        "retrieval": {
            "holographic_compression": 64,
            "superposition_collapse": "adaptive",
            "top_k": 3
        },
        "llm": {
            "model": "phi3:3.8b-q4_0",
            "temperature": 0.1
        }
    }

@pytest.fixture
def test_gene():
    """Create a test KnowledgeGene"""
    import numpy as np
    from minerva.core.genome import KnowledgeGene
    
    return KnowledgeGene(
        id="test_gene_1",
        semantic_pattern=np.random.randn(768),
        connections={"other_gene": 0.5},
        strength=0.8,
        metadata={"type": "concept", "source": "test"}
    )
