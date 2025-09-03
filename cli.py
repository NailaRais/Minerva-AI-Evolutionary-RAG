#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path
from core.genome import NeuralGenePool, KnowledgeGene
from core.evolution import EvolutionaryOptimizer
from retrieval.fractal_parser import FractalParser
from retrieval.holographic import HolographicMemory
import ollama

class MinervaCLI:
    """Command-line interface for Minerva."""
    
    def __init__(self):
        self.config = self._load_config()
        self.gene_pool = NeuralGenePool(self.config)
        self.parser = FractalParser()
        self.optimizer = EvolutionaryOptimizer(self.gene_pool)
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        config_path = Path('config.yaml')
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def ingest(self, path: str):
        """Ingest documents from a file or directory."""
        path = Path(path)
        documents = []
        
        if path.is_file():
            documents = self._parse_file(path)
        elif path.is_dir():
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    documents.extend(self._parse_file(file_path))
        
        for doc in documents:
            components = self.parser.decompose(doc)
            self._components_to_genes(components)
        
        print(f"Ingested {len(documents)} documents")
    
    def _parse_file(self, file_path: Path) -> List[str]:
        """Parse a single file into text content."""
        # Simple implementation - would use unstructured in real version
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [f.read()]
        except:
            return []
    
    def _components_to_genes(self, components: List[dict]):
        """Convert semantic components to knowledge genes."""
        for comp in components:
            gene = KnowledgeGene(
                id=f"gene_{hash(comp['text']) % 1000000}",
                semantic_pattern=comp['embedding'],
                connections={},
                strength=1.0,
                metadata={
                    'type': comp['type'],
                    'text': comp['text'][:1000],  # Truncate very long text
                    'depth': comp.get('depth', 0)
                }
            )
            self.gene_pool.add_gene(gene)
    
    def query(self, question: str, interactive: bool = False):
        """Process a query through the evolutionary RAG system."""
        # Activate relevant genes
        activated_genes = self.gene_pool.activate_genes(question)
        
        # Evolutionary processing
        for _ in range(self.config['evolution']['depth']):
            activated_genes = self.gene_pool.evolve_activation(activated_genes)
        
        # Generate context from top genes
        context = self._genes_to_context(activated_genes)
        
        # Generate response using local LLM
        response = self._generate_response(question, context)
        
        if interactive:
            print(f"\nMinerva: {response}\n")
        else:
            print(response)
    
    def _genes_to_context(self, genes: List[KnowledgeGene]) -> str:
        """Convert activated genes to context text."""
        context_parts = []
        for gene in genes[:self.config['retrieval']['top_k']]:
            if 'text' in gene.metadata:
                context_parts.append(gene.metadata['text'])
        return "\n\n".join(context_parts)
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate response using local LLM."""
        prompt = f"""Based on the following context, answer the question. If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = ollama.generate(
                model=self.config['llm']['model'],
                prompt=prompt,
                options={
                    'temperature': self.config['llm']['temperature']
                }
            )
            return response['response']
        except:
            return "Error generating response. Please check your Ollama setup."

def main():
    """Main CLI entry point."""
    cli = MinervaCLI()
    parser = argparse.ArgumentParser(description="Minerva Evolutionary RAG System")
    subparsers = parser.add_subparsers(dest='command')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('path', help='Path to file or directory')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the knowledge base')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--interactive', '-i', action='store_true', 
                             help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.command == 'ingest':
        cli.ingest(args.path)
    elif args.command == 'query':
        cli.query(args.question, args.interactive)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
