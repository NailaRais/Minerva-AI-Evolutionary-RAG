#!/usr/bin/env python3
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

class BenchmarkRunner:
    def __init__(self):
        self.results = {}
        self.start_time = None
        
    def start_timer(self):
        self.start_time = time.time()
        
    def stop_timer(self, benchmark_name):
        elapsed = time.time() - self.start_time
        self.results[benchmark_name] = elapsed
        return elapsed
        
    def memory_benchmark(self):
        """Benchmark memory usage patterns"""
        import psutil
        process = psutil.Process()
        
        memory_before = process.memory_info().rss
        # Create test data
        test_data = [np.random.rand(768) for _ in range(1000)]
        memory_after = process.memory_info().rss
        
        return {
            'memory_used_bytes': memory_after - memory_before,
            'memory_used_mb': (memory_after - memory_before) / 1024 / 1024
        }
        
    def storage_benchmark(self, test_dir="benchmark_data"):
        """Benchmark storage efficiency"""
        Path(test_dir).mkdir(exist_ok=True)
        
        # Create test files
        file_sizes = []
        for i in range(10):
            content = "x" * (1024 * 1024)  # 1MB files
            file_path = Path(test_dir) / f"test_{i}.txt"
            file_path.write_text(content)
            file_sizes.append(file_path.stat().st_size)
            
        total_size = sum(file_sizes)
        
        # Cleanup
        for file_path in Path(test_dir).glob("test_*.txt"):
            file_path.unlink()
            
        return {
            'total_size_bytes': total_size,
            'average_file_size': total_size / len(file_sizes),
            'file_count': len(file_sizes)
        }
        
    def run_comprehensive_benchmark(self):
        """Run all benchmarks"""
        print("🚀 Starting comprehensive benchmark...")
        
        # Memory benchmark
        memory_results = self.memory_benchmark()
        self.results['memory'] = memory_results
        print(f"📦 Memory usage: {memory_results['memory_used_mb']:.2f} MB")
        
        # Storage benchmark
        storage_results = self.storage_benchmark()
        self.results['storage'] = storage_results
        print(f"💾 Storage test: {storage_results['file_count']} files, {storage_results['total_size_bytes']/1024/1024:.2f} MB total")
        
        # Performance benchmarks
        benchmarks = [
            ('array_operations', self.benchmark_array_ops),
            ('string_processing', self.benchmark_string_processing),
            ('file_io', self.benchmark_file_io)
        ]
        
        for name, benchmark_func in benchmarks:
            self.start_timer()
            result = benchmark_func()
            elapsed = self.stop_timer(name)
            self.results[name] = {'time_seconds': elapsed, 'details': result}
            print(f"⏱️  {name}: {elapsed:.3f}s")
            
        return self.results
        
    def benchmark_array_ops(self):
        """Benchmark numpy array operations"""
        large_array = np.random.rand(10000, 768)
        # Various operations
        result = {
            'mean': np.mean(large_array),
            'std': np.std(large_array),
            'dot_product': np.dot(large_array[0], large_array[1])
        }
        return result
        
    def benchmark_string_processing(self):
        """Benchmark string manipulation"""
        large_text = " ".join(["minerva"] * 10000)
        operations = {
            'split_count': len(large_text.split()),
            'find_word': large_text.find("minerva"),
            'replace': large_text.replace("minerva", "athena")
        }
        return operations
        
    def benchmark_file_io(self):
        """Benchmark file I/O operations"""
        test_file = Path("temp_benchmark.txt")
        test_data = "x" * 1024 * 1024  # 1MB data
        
        # Write benchmark
        with open(test_file, 'w') as f:
            f.write(test_data)
            
        # Read benchmark
        with open(test_file, 'r') as f:
            content = f.read()
            
        test_file.unlink()
        
        return {
            'write_success': len(test_data) == len(content),
            'data_size': len(test_data)
        }
        
    def save_results(self, output_file=None):
        """Save benchmark results to file"""
        if output_file is None:
            output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        results_dir = Path("benchmark_results")
        results_dir.mkdir(exist_ok=True)
        
        output_path = results_dir / output_file
        
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': self.results
            }, f, indent=2)
            
        print(f"💾 Results saved to: {output_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive benchmarks")
    parser.add_argument('--output', '-o', help='Output file name')
    parser.add_argument('--quick', '-q', action='store_true', help='Run quick benchmark')
    
    args = parser.parse_args()
    
    benchmark = BenchmarkRunner()
    results = benchmark.run_comprehensive_benchmark()
    benchmark.save_results(args.output)
    
    print(f"\n🎯 Benchmark completed!")
    print(f"Total benchmarks run: {len(results)}")

if __name__ == '__main__':
    main()
