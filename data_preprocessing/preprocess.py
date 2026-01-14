#!/usr/bin/env python3
"""
FALCON Data Preprocessing Script.
Reads raw log data and builds graph representations.

Usage:
    python preprocess.py
    python preprocess.py --force              # Force rebuild all (ignore cache)
    python preprocess.py --versions v1-12896 v2-12893  # Specific versions only
"""

#!/usr/bin/env python3
"""
FALCON Data Preprocessing Script.
Standalone module for building graph representations from raw logs.

Usage:
    python preprocess.py
    python preprocess.py --force
    python preprocess.py --versions v1-12896 v2-12893
"""

import sys
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import config
from graph_builder import GraphBuilder


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FALCON Data Preprocessing")
    parser.add_argument(
        '--data_path', 
        type=str, 
        default=None,
        help='Path to raw data directory (default: from config)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save processed graphs (default: from config)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild all graphs (ignore cache)'
    )
    parser.add_argument(
        '--versions',
        type=str,
        nargs='+',
        default=None,
        help='Specific versions to process (e.g., v1-12896 v2-12893)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Verbose output'
    )
    return parser.parse_args()


def load_ground_truth(data_path: str) -> Dict:
    """
    Load ground truth from JSON file.
    
    Returns:
        Dictionary mapping version -> {file::function: True}
    """
    gt_path = Path(data_path) / config.GROUND_TRUTH_FILE
    
    if not gt_path.exists():
        print(f"Warning: Ground truth file not found: {gt_path}")
        return {}
    
    try:
        with open(gt_path, 'r') as f:
            gt_array = json.load(f)
        
        # Convert array format to dict: version -> {file::function: True}
        ground_truth = {}
        for entry in gt_array:
            version = entry.get('version', '')
            files = entry.get('files', [])
            functions = entry.get('functions', [])
            
            # Create function mappings for this version
            version_gt = {}
            for file in files:
                for func in functions:
                    key = f"{file}::{func}"
                    version_gt[key] = True
                    version_gt[func] = True
            
            ground_truth[version] = version_gt
        
        print(f"✓ Loaded ground truth with {len(ground_truth)} versions")
        return ground_truth
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return {}


def process_version(
    version_name: str,
    version_dir: Path,
    builder: GraphBuilder,
    version_gt: Dict,
    output_path: Path,
    force: bool = False,
    verbose: bool = True
) -> Optional[Dict]:
    """
    Process a single version directory.
    
    Returns:
        Dictionary with processing info, or None if failed
    """
    cache_path = output_path / f"{version_name}.pt"
    
    # Check cache
    if not force and cache_path.exists():
        if verbose:
            print(f"  ✓ Already cached: {cache_path.name}")
        try:
            # PyTorch 2.6+ requires weights_only=False for custom objects
            # Always use map_location='cpu' for preprocessing (portable files)
            data = torch.load(cache_path, map_location='cpu', weights_only=False)
            return {
                'version': version_name,
                'cached': True,
                'num_nodes': data.num_nodes,
                'num_edges': data.num_edges,
                'num_faulty': data.y.sum().item() if hasattr(data, 'y') else 0
            }
        except Exception as e:
            print(f"  Warning: Cache corrupted, rebuilding: {e}")
    
    # Look for fail logs
    fail_dir = version_dir / 'fail'
    if not fail_dir.exists():
        if verbose:
            print(f"  ✗ No fail directory found")
        return None
    
    fail_logs = list(fail_dir.glob("*.log"))
    if not fail_logs:
        if verbose:
            print(f"  ✗ No log files in fail directory")
        return None
    
    # Use first log file
    log_path = fail_logs[0]
    if verbose:
        print(f"  Processing: {log_path.name}")
    
    try:
        # Build graph
        data = builder.build_graph_from_log_file(str(log_path), ground_truth=version_gt)
        
        if data.num_nodes == 0:
            if verbose:
                print(f"  ✗ Empty graph")
            return None
        
        # Store version name
        data.version_name = version_name
        
        # Save to cache
        torch.save(data, cache_path)
        
        num_faulty = data.y.sum().item() if hasattr(data, 'y') else 0
        
        if verbose:
            print(f"  ✓ Saved: {data.num_nodes} nodes, {data.num_edges} edges, {num_faulty} faulty")
        
        return {
            'version': version_name,
            'cached': False,
            'num_nodes': data.num_nodes,
            'num_edges': data.num_edges,
            'num_faulty': num_faulty,
            'log_file': log_path.name
        }
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def main():
    """Main preprocessing pipeline."""
    args = parse_args()
    
    # Use config paths if not specified
    data_path = args.data_path if args.data_path else str(config.get_raw_data_path())
    output_path_str = args.output_path if args.output_path else str(config.get_output_path())
    
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("FALCON DATA PREPROCESSING")
    print("=" * 70)
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Path: {data_path}")
    print(f"Output Path: {output_path_str}")
    print(f"Force Rebuild: {args.force}")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(output_path_str)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load ground truth
    ground_truth = load_ground_truth(data_path)
    
    # Initialize graph builder
    print(f"\nInitializing Graph Builder...")
    builder = GraphBuilder(
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        cache_dir=str(output_path)
    )
    print(f"✓ Graph Builder ready")
    
    # Find version directories
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"\nError: Data directory not found: {data_dir}")
        return 1
    
    version_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
    version_dirs = sorted(version_dirs, key=lambda x: x.name)
    
    # Filter by specific versions if provided
    if args.versions:
        version_dirs = [d for d in version_dirs if d.name in args.versions]
    
    print(f"\nFound {len(version_dirs)} version directories to process")
    
    # Process each version
    results = []
    success_count = 0
    cached_count = 0
    failed_count = 0
    
    print("\n" + "-" * 70)
    
    for i, version_dir in enumerate(version_dirs, 1):
        version_name = version_dir.name
        version_gt = ground_truth.get(version_name, {})
        
        print(f"\n[{i}/{len(version_dirs)}] {version_name}")
        
        if not version_gt:
            print(f"  Warning: No ground truth for {version_name}")
        
        result = process_version(
            version_name=version_name,
            version_dir=version_dir,
            builder=builder,
            version_gt=version_gt,
            output_path=output_path,
            force=args.force,
            verbose=args.verbose
        )
        
        if result:
            results.append(result)
            if result.get('cached'):
                cached_count += 1
            else:
                success_count += 1
        else:
            failed_count += 1
    
    # Save embedding cache
    builder._save_cache()
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total versions: {len(version_dirs)}")
    print(f"  - Newly processed: {success_count}")
    print(f"  - From cache: {cached_count}")
    print(f"  - Failed: {failed_count}")
    print(f"\nDuration: {duration}")
    print(f"Output: {output_path}")
    
    # Save summary
    summary_path = output_path / "preprocessing_summary.json"
    summary = {
        'timestamp': start_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'total_versions': len(version_dirs),
        'success_count': success_count,
        'cached_count': cached_count,
        'failed_count': failed_count,
        'results': results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    
    # Statistics
    if results:
        total_nodes = sum(r['num_nodes'] for r in results)
        total_edges = sum(r['num_edges'] for r in results)
        total_faulty = sum(r['num_faulty'] for r in results)
        
        print(f"\nDataset Statistics:")
        print(f"  Total graphs: {len(results)}")
        print(f"  Total nodes: {total_nodes:,}")
        print(f"  Total edges: {total_edges:,}")
        print(f"  Total faulty nodes: {total_faulty}")
        print(f"  Avg nodes/graph: {total_nodes/len(results):.1f}")
        print(f"  Avg edges/graph: {total_edges/len(results):.1f}")
    
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

