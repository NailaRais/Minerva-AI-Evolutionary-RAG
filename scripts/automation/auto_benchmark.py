#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from minerva.core.genome import NeuralGenePool
from minerva.retrieval.fractal_parser import FractalParser

class AutoBenchmark:
    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now().isoformat()
    
    def run_benchmark_suite(self, full_suite: bool = False):
        """Run comprehensive benchmark tests."""
        print("🚀 Starting Minerva Benchmark Suite")
        
        benchmarks = {
            'ingestion_speed': self.benchmark_ingestion,
            'query_latency': self.benchmark_query_latency,
            'memory_usage': self.benchmark_memory_usage,
            'accuracy': self.benchmark_accuracy
        }
        
        if full_suite:
            benchmarks.update({
                'concurrent_queries': self.benchmark_concurrency,
                'scalability': self.benchmark_scalability
            })
        
        for name, benchmark_func in benchmarks.items():
            print(f"\n📊 Running {name}...")
            try:
                result = benchmark_func()
                self.results[name] = result
                print(f"   Result: {result}")
            except Exception as e:
                print(f"   Failed: {e}")
                self.results[name] = {'error': str(e)}
        
        self.save_results()
        return self.results
    
    def benchmark_ingestion(self) -> dict:
        """Benchmark document ingestion speed."""
        test_docs = ["sample_document.txt"] * 100  # Simulate 100 docs
        
        start_time = time.time()
        parser = FractalParser()
        gene_pool = NeuralGenePool({})
        
        for doc in test_docs:
            components = parser.decompose(f"Sample content {doc}")
            # Simulate gene creation
            for comp in components[:10]:  # Limit for benchmark
                # Simplified gene creation
                pass
        
        end_time = time.time()
        
        return {
            'documents_processed': len(test_docs),
            'total_time_seconds': end_time - start_time,
            'docs_per_second': len(test_docs) / (end_time - start_time)
        }
    
    def benchmark_query_latency(self) -> dict:
        """Benchmark query response times."""
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning",
            "How does neural network work?",
            "What is deep learning?",
            "Describe natural language processing"
        ]
        
        latencies = []
        gene_pool = NeuralGenePool({})
        
        for query in test_queries:
            start_time = time.time()
            genes = gene_pool.activate_genes(query)
            end_time = time.time()
            latencies.append(end_time - start_time)
        
        return {
            'avg_latency_seconds': sum(latencies) / len(latencies),
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'p95_latency': sorted(latencies)[int(len(latencies) * 0.95)]
        }
    
    def save_results(self):
        """Save benchmark results to file."""
        results_dir = Path("benchmark_results")
        results_dir.mkdir(exist_ok=True)
        
        filename = results_dir / f"benchmark_{self.timestamp.replace(':', '-')}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': self.timestamp,
                'results': self.results
            }, f, indent=2)
        
        print(f"\n📁 Results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Run Minerva benchmarks")
    parser.add_argument('--full-suite', action='store_true', help='Run comprehensive benchmarks')
    
    args = parser.parse_args()
    
    benchmark = AutoBenchmark()
    results = benchmark.run_benchmark_suite(args.full_suite)
    
    print(f"\n🎯 Benchmark completed!")
    print(f"Results: {json.dumps(results, indent=2)}")

if __name__ == '__main__':
    main()
