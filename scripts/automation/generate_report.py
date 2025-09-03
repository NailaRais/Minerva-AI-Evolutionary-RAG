#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

class ReportGenerator:
    def __init__(self):
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
    def load_benchmark_data(self, benchmark_file):
        """Load benchmark data from file"""
        with open(benchmark_file, 'r') as f:
            return json.load(f)
            
    def generate_comprehensive_report(self, benchmark_files):
        """Generate comprehensive report from multiple benchmark runs"""
        all_data = []
        
        for file in benchmark_files:
            if file.exists():
                data = self.load_benchmark_data(file)
                all_data.append(data)
                
        if not all_data:
            print("❌ No benchmark data found")
            return
            
        # Generate summary report
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_runs': len(all_data),
            'summary': self._generate_summary(all_data),
            'trends': self._analyze_trends(all_data)
        }
        
        # Save report
        report_file = self.reports_dir / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Generate charts
        self._generate_charts(all_data)
        
        print(f"📊 Comprehensive report generated: {report_file}")
        return report
        
    def _generate_summary(self, all_data):
        """Generate summary statistics"""
        summary = {}
        
        # Extract metrics from all runs
        for metric in ['ingestion_speed', 'query_latency', 'memory_usage']:
            values = []
            for data in all_data:
                if metric in data.get('results', {}):
                    values.append(data['results'][metric])
                    
            if values:
                summary[metric] = {
                    'average': np.mean(values),
                    'std_dev': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
                
        return summary
        
    def _analyze_trends(self, all_data):
        """Analyze trends across multiple runs"""
        trends = {}
        
        # Example trend analysis - could be expanded
        timestamps = [datetime.fromisoformat(data['timestamp']) for data in all_data]
        performance_metrics = []
        
        for data in all_data:
            if 'results' in data and 'query_latency' in data['results']:
                performance_metrics.append(data['results']['query_latency']['avg_latency_seconds'])
                
        if performance_metrics and len(performance_metrics) > 1:
            trends['performance_trend'] = {
                'improvement': performance_metrics[-1] < performance_metrics[0],
                'percent_change': ((performance_metrics[-1] - performance_metrics[0]) / performance_metrics[0]) * 100
            }
            
        return trends
        
    def _generate_charts(self, all_data):
        """Generate visualization charts"""
        # CPU usage over time
        timestamps = []
        cpu_usage = []
        
        for data in all_data:
            if 'results' in data and 'system' in data['results']:
                timestamps.append(datetime.fromisoformat(data['timestamp']))
                cpu_usage.append(data['results']['system']['cpu_percent'])
                
        if timestamps and cpu_usage:
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, cpu_usage, 'b-', marker='o')
            plt.title('CPU Usage Over Time')
            plt.xlabel('Time')
            plt.ylabel('CPU Percentage')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.reports_dir / 'cpu_usage_trend.png')
            plt.close()
            
        # Memory usage
        memory_usage = []
        for data in all_data:
            if 'results' in data and 'memory' in data['results']:
                memory_usage.append(data['results']['memory']['memory_used_mb'])
                
        if memory_usage:
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(memory_usage)), memory_usage, 'r-', marker='s')
            plt.title('Memory Usage Trend')
            plt.xlabel('Run Number')
            plt.ylabel('Memory Usage (MB)')
            plt.tight_layout()
            plt.savefig(self.reports_dir / 'memory_usage_trend.png')
            plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate performance reports")
    parser.add_argument('--benchmark-dir', default='benchmark_results', help='Directory containing benchmark files')
    parser.add_argument('--pattern', default='benchmark_*.json', help='File pattern to match')
    
    args = parser.parse_args()
    
    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.exists():
        print(f"❌ Benchmark directory not found: {benchmark_dir}")
        return
        
    benchmark_files = list(benchmark_dir.glob(args.pattern))
    
    if not benchmark_files:
        print(f"❌ No benchmark files found matching: {args.pattern}")
        return
        
    print(f"📈 Found {len(benchmark_files)} benchmark files")
    
    generator = ReportGenerator()
    report = generator.generate_comprehensive_report(benchmark_files)
    
    if report:
        print("🎉 Report generation completed!")
        print(f"   Total runs analyzed: {report['total_runs']}")
        
        if 'summary' in report and 'query_latency' in report['summary']:
            latency = report['summary']['query_latency']
            print(f"   Average query latency: {latency['average']:.3f}s ± {latency['std_dev']:.3f}s")

if __name__ == '__main__':
    main()
