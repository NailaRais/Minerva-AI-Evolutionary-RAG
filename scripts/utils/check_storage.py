#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    for entry in path.rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total

def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def main():
    parser = argparse.ArgumentParser(description="Check storage footprint")
    parser.add_argument('--max-size', type=float, default=3.0, help='Maximum allowed size in GB')
    parser.add_argument('--path', default='.', help='Path to check')
    
    args = parser.parse_args()
    
    base_path = Path(args.path)
    total_size = get_directory_size(base_path)
    total_gb = total_size / (1024 ** 3)
    
    print(f"Storage footprint analysis for: {base_path}")
    print(f"Total size: {format_size(total_size)} ({total_gb:.2f} GB)")
    
    # Check individual components
    components = {
        'Models': base_path / 'models',
        'Data': base_path / 'data',
        'Index': base_path / 'index',
        'Cache': base_path / '.cache'
    }
    
    for name, path in components.items():
        if path.exists():
            size = get_directory_size(path)
            print(f"{name}: {format_size(size)}")
    
    if total_gb > args.max_size:
        print(f"❌ ERROR: Storage exceeds {args.max_size} GB limit!")
        sys.exit(1)
    else:
        print(f"✅ Storage within {args.max_size} GB limit")
        sys.exit(0)

if __name__ == '__main__':
    main()
