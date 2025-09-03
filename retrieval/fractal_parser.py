import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np

class FractalParser:
    """Parses documents into hierarchical semantic chunks."""
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.patterns = {
            'concept': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'relationship': r'\b(?:is|are|has|have|contains|belongs to)\b',
            'statement': r'[^.!?]+[.!?]'
        }
    
    def decompose(self, text: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """Break down text into fractal semantic components."""
        concepts = self._extract_concepts(text)
        statements = self._extract_statements(text)
        
        # Create hierarchical structure
        components = []
        for statement in statements:
            stmt_components = self._analyze_statement(statement, concepts, max_depth)
            components.extend(stmt_components)
        
        return components
    
    def _extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract key concepts from text."""
        concepts = []
        matches = re.finditer(self.patterns['concept'], text)
        
        for match in matches:
            concept_text = match.group(0)
            embedding = self.embedder.encode([concept_text])[0]
            
            concepts.append({
                'text': concept_text,
                'embedding': embedding,
                'position': match.start(),
                'type': 'concept'
            })
        
        return concepts
    
    def _extract_statements(self, text: str) -> List[str]:
        """Split text into individual statements."""
        return re.findall(self.patterns['statement'], text)
    
    def _analyze_statement(self, statement: str, concepts: List[Dict[str, Any]], 
                          depth: int) -> List[Dict[str, Any]]:
        """Recursively analyze a statement at multiple semantic levels."""
        if depth <= 0:
            return []
        
        components = []
        stmt_embedding = self.embedder.encode([statement])[0]
        
        # Find concepts mentioned in this statement
        stmt_concepts = [
            concept for concept in concepts 
            if concept['text'] in statement
        ]
        
        # Create component for the full statement
        components.append({
            'text': statement,
            'embedding': stmt_embedding,
            'type': 'statement',
            'depth': depth,
            'contains': [c['text'] for c in stmt_concepts]
        })
        
        # Recursively analyze sub-components
        if len(stmt_concepts) > 1 and depth > 1:
            for i in range(len(stmt_concepts) - 1):
                relationship = self._find_relationship(
                    stmt_concepts[i], stmt_concepts[i + 1], statement
                )
                if relationship:
                    components.append(relationship)
        
        return components
    
    def _find_relationship(self, concept1: Dict[str, Any], concept2: Dict[str, Any], 
                          statement: str) -> Dict[str, Any]:
        """Find relationships between concepts within a statement."""
        # Look for relationship patterns between the concepts
        between_text = statement[concept1['position'] + len(concept1['text']):concept2['position']]
        
        if re.search(self.patterns['relationship'], between_text):
            relationship_embedding = self.embedder.encode([between_text])[0]
            
            return {
                'type': 'relationship',
                'concept1': concept1['text'],
                'concept2': concept2['text'],
                'text': between_text.strip(),
                'embedding': relationship_embedding,
                'strength': 0.5  # Default strength
            }
        
        return None
