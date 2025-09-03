#!/usr/bin/env python3
import time
import psutil
import json
from datetime import datetime
from pathlib import Path

class PerformanceMonitor:
    def __init__(self, interval=1.0, log_file="performance.log"):
        self.interval = interval
        self.log_file = Path(log_file)
        self.running = False
        
    def get_system_stats(self):
        """Get current system performance statistics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=None),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'disk_usage': psutil.disk_usage('/').percent,
            'process_memory_mb': psutil.Process().memory_info().rss / (1024**2)
        }
        
    def log_stats(self, stats):
        """Log statistics to file"""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(stats) + '\n')
            
    def monitor_loop(self):
        """Main monitoring loop"""
        print(f"📊 Starting performance monitoring (interval: {self.interval}s)")
        print("Press Ctrl+C to stop...")
        
        self.running = True
        try:
            while self.running:
                stats = self.get_system_stats()
                self.log_stats(stats)
                
                # Print to console
                print(f"[{stats['timestamp']}] CPU: {stats['cpu_percent']}% | "
                      f"Mem: {stats['memory_used_gb']:.1f}GB | "
                      f"Process: {stats['process_memory_mb']:.1f}MB")
                      
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped")
            
    def generate_report(self, input_file=None, output_file="performance_report.json"):
        """Generate performance report from logs"""
        if input_file is None:
            input_file = self.log_file
            
        if not Path(input_file).exists():
            print(f"❌ Log file not found: {input_file}")
            return
            
        stats = []
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    stats.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
                    
        if not stats:
            print("❌ No valid data found in log file")
            return
            
        # Calculate statistics
        cpu_values = [s['cpu_percent'] for s in stats]
        memory_values = [s['memory_used_gb'] for s in stats]
        
        report = {
            'period': {
                'start': stats[0]['timestamp'],
                'end': stats[-1]['timestamp'],
                'duration_seconds': (datetime.fromisoformat(stats[-1]['timestamp']) - 
                                   datetime.fromisoformat(stats[0]['timestamp'])).total_seconds()
            },
            'cpu': {
                'average': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'average_gb': sum(memory_values) / len(memory_values),
                'max_gb': max(memory_values),
                'min_gb': min(memory_values)
            },
            'sample_count': len(stats)
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"📈 Performance report generated: {output_file}")
        print(f"   Duration: {report['period']['duration_seconds']:.0f}s")
        print(f"   Avg CPU: {report['cpu']['average']:.1f}%")
        print(f"   Avg Memory: {report['memory']['average_gb']:.1f}GB")
        
        return report

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor system performance")
    parser.add_argument('action', choices=['monitor', 'report'], help='Action to perform')
    parser.add_argument('--interval', '-i', type=float, default=1.0, help='Monitoring interval in seconds')
    parser.add_argument('--log-file', default='performance.log', help='Log file path')
    parser.add_argument('--input-file', help='Input log file for report generation')
    parser.add_argument('--output-file', default='performance_report.json', help='Output report file')
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(interval=args.interval, log_file=args.log_file)
    
    if args.action == 'monitor':
        monitor.monitor_loop()
    elif args.action == 'report':
        monitor.generate_report(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
