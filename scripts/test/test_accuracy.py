#!/usr/bin/env python3
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from minerva.core.genome import NeuralGenePool
from minerva.retrieval.fractal_parser import FractalParser

class AccuracyTester:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.gene_pool = NeuralGenePool(self.config)
        self.parser = FractalParser()
        self.results = []
    
    def _load_config(self, config_path: str) -> Dict:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_test_data(self, data_path: str) -> List[Dict]:
        """Load test questions and answers."""
        with open(data_path, 'r') as f:
            return json.load(f)
    
    def test_retrieval_accuracy(self, test_data: List[Dict], top_k: int = 3) -> float:
        """Test retrieval accuracy."""
        correct = 0
        total = len(test_data)
        
        for i, item in enumerate(test_data):
            question = item['question']
            expected_answer = item['answer']
            
            # Activate genes and get top results
            activated_genes = self.gene_pool.activate_genes(question)
            top_genes = activated_genes[:top_k]
            
            # Check if any relevant context was found
            context_found = any(
                expected_answer.lower() in gene.metadata.get('text', '').lower()
                for gene in top_genes
            )
            
            if context_found:
                correct += 1
            
            self.results.append({
                'question': question,
                'expected': expected_answer,
                'found': context_found,
                'contexts': [gene.metadata.get('text', '')[:100] for gene in top_genes]
            })
        
        accuracy = correct / total
        return accuracy
    
    def generate_report(self, output_path: str):
        """Generate detailed accuracy report."""
        report = {
            'summary': {
                'total_questions': len(self.results),
                'correct_retrievals': sum(1 for r in self.results if r['found']),
                'accuracy': sum(1 for r in self.results if r['found']) / len(self.results)
            },
            'detailed_results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Test Minerva accuracy")
    parser.add_argument('--data', default='benchmark_data/test_set.json', help='Test data path')
    parser.add_argument('--output', default='accuracy_report.json', help='Output report path')
    parser.add_argument('--quick', action='store_true', help='Run quick test with sample data')
    
    args = parser.parse_args()
    
    tester = AccuracyTester()
    
    if args.quick:
        # Create sample test data
        test_data = [
            {'question': 'What is AI?', 'answer': 'artificial intelligence'},
            {'question': 'Explain machine learning', 'answer': 'algorithm training'}
        ]
    else:
        test_data = tester.load_test_data(args.data)
    
    accuracy = tester.test_retrieval_accuracy(test_data)
    report = tester.generate_report(args.output)
    
    print(f"Retrieval Accuracy: {accuracy:.3f}")
    print(f"Report saved to: {args.output}")
    
    return accuracy

if __name__ == '__main__':
    main()
