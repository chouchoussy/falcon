#!/usr/bin/env python3
"""
FALCON Main Pipeline Script.
Implements Leave-One-Out Cross Validation (LOOCV) for fault localization.

Usage:
    python main.py
"""

import sys
import json
import torch
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# PyTorch Geometric imports
from torch_geometric.data import Data

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.dataset import GraphBuilder, augment_graph, parse_log_file
from src.models import FALCONModel
from src.training import FALCONTrainer
from src.utils import calculate_metrics, print_metrics, save_results_to_csv, analyze_rank_distribution


def setup_environment():
    """Setup directories and print configuration."""
    print("\n" + "=" * 70)
    print("FALCON: Fault Localization with Contrastive Learning")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create necessary directories
    config.ensure_directories()
    
    # Print configuration
    config.print_config()


def load_ground_truth(data_path: str) -> Dict:
    """
    Load ground truth from JSON file.
    
    Args:
        data_path: Path to data directory
    
    Returns:
        Dictionary mapping version -> {file::function: True/False}
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
                    # Create key: filename::function
                    key = f"{file}::{func}"
                    version_gt[key] = True
                    # Also add just function name for matching
                    version_gt[func] = True
            
            ground_truth[version] = version_gt
        
        print(f"✓ Loaded ground truth with {len(ground_truth)} versions")
        return ground_truth
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        import traceback
        traceback.print_exc()
        return {}


def load_or_build_graph(
    version_name: str,
    log_path: Path,
    builder: GraphBuilder,
    ground_truth: Dict,
    use_cache: bool = True
) -> Optional[Data]:
    """
    Load cached graph or build from log file.
    
    Args:
        version_name: Name of the version (e.g., 'v1-12896')
        log_path: Path to log file
        builder: GraphBuilder instance
        ground_truth: Ground truth dictionary
        use_cache: Whether to use cached graphs
    
    Returns:
        Data object or None if failed
    """
    cache_path = Path(config.PROCESSED_PATH) / f"{version_name}.pt"
    
    # Try to load from cache
    if use_cache and cache_path.exists():
        try:
            data = torch.load(cache_path)
            data.version_name = version_name
            return data
        except Exception as e:
            print(f"  Warning: Failed to load cache for {version_name}: {e}")
            # Fall through to rebuild
    
    # Build graph from log file
    try:
        # Check if log file exists
        if not log_path.exists():
            print(f"  Warning: Log file not found: {log_path}")
            return None
        
        # Parse and build graph
        data = builder.build_graph_from_log_file(
            str(log_path),
            ground_truth=ground_truth
        )
        
        # Check if graph has valid data
        if data.num_nodes == 0:
            print(f"  Warning: Empty graph for {version_name}")
            return None
        
        # Store version name
        data.version_name = version_name
        
        # Save to cache
        if use_cache:
            torch.save(data, cache_path)
        
        return data
        
    except Exception as e:
        print(f"  Error building graph for {version_name}: {e}")
        return None


def load_all_graphs(
    data_path: str,
    builder: GraphBuilder,
    ground_truth: Dict,
    use_cache: bool = True
) -> List[Data]:
    """
    Load all graphs from data directory with caching.
    
    Args:
        data_path: Path to data directory
        builder: GraphBuilder instance
        ground_truth: Ground truth dictionary
        use_cache: Whether to use cached graphs
    
    Returns:
        List of Data objects
    """
    print("\n" + "=" * 70)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 70)
    
    data_dir = Path(data_path)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return []
    
    # Find all version directories (v1-12896, v2-12893, etc.)
    version_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
    version_dirs = sorted(version_dirs, key=lambda x: x.name)
    
    print(f"Found {len(version_dirs)} version directories")
    
    all_graphs = []
    failed_versions = []
    
    for version_dir in version_dirs:
        version_name = version_dir.name
        
        print(f"\nProcessing: {version_name}")
        
        # Look for fail logs in fail/ subdirectory
        fail_dir = version_dir / 'fail'
        if not fail_dir.exists():
            print(f"  Warning: No fail directory found")
            failed_versions.append(version_name)
            continue
        
        # Get all log files in fail directory
        fail_logs = list(fail_dir.glob("*.log"))
        
        if not fail_logs:
            print(f"  Warning: No log files in fail directory")
            failed_versions.append(version_name)
            continue
        
        # Use the first log file (or can concatenate all if needed)
        log_path = fail_logs[0]
        print(f"  Using log: {log_path.name}")
        
        # Get ground truth for this version
        version_gt = ground_truth.get(version_name, {})
        
        if not version_gt:
            print(f"  Warning: No ground truth found for {version_name}")
        
        # Load or build graph
        data = load_or_build_graph(
            version_name=version_name,
            log_path=log_path,
            builder=builder,
            ground_truth=version_gt,
            use_cache=use_cache
        )
        
        if data is not None:
            # Check if graph has faulty nodes
            if hasattr(data, 'y') and data.y.sum() > 0:
                all_graphs.append(data)
                print(f"  ✓ Loaded: {data.num_nodes} nodes, {data.num_edges} edges, "
                      f"{data.y.sum().item()} faulty nodes")
            else:
                print(f"  ⚠ Skipped: No faulty nodes labeled")
                failed_versions.append(version_name)
        else:
            failed_versions.append(version_name)
    
    print(f"\n{'='*70}")
    print(f"Successfully loaded: {len(all_graphs)} graphs")
    if failed_versions:
        print(f"Failed to load: {len(failed_versions)} versions")
        print(f"Failed versions: {', '.join(failed_versions[:5])}" + 
              (f"... and {len(failed_versions)-5} more" if len(failed_versions) > 5 else ""))
    print(f"{'='*70}")
    
    return all_graphs


def find_bug_rank(scores: torch.Tensor, labels: torch.Tensor) -> int:
    """
    Find the rank of the first faulty node.
    
    Args:
        scores: Predicted suspiciousness scores [num_nodes]
        labels: Ground truth labels [num_nodes] (1=faulty, 0=normal)
    
    Returns:
        Rank of first faulty node (1-indexed), or -1 if not found
    """
    # Find faulty nodes
    faulty_indices = (labels == 1).nonzero(as_tuple=True)[0]
    
    if len(faulty_indices) == 0:
        return -1
    
    # Rank all nodes by scores (descending)
    ranked_indices = torch.argsort(scores, descending=True)
    
    # Find positions of faulty nodes
    faulty_ranks = []
    for faulty_idx in faulty_indices:
        # Find position in ranked list
        position = (ranked_indices == faulty_idx).nonzero(as_tuple=True)[0]
        if len(position) > 0:
            rank = position[0].item() + 1  # 1-indexed
            faulty_ranks.append(rank)
    
    # Return the best (smallest) rank
    return min(faulty_ranks) if faulty_ranks else -1


def run_loocv(
    all_graphs: List[Data],
    verbose: bool = True
) -> Tuple[List[int], List[str]]:
    """
    Run Leave-One-Out Cross Validation.
    
    Args:
        all_graphs: List of all graph Data objects
        verbose: Whether to print progress
    
    Returns:
        Tuple of (rank_list, version_names)
    """
    print("\n" + "=" * 70)
    print("STEP 2: LEAVE-ONE-OUT CROSS VALIDATION")
    print("=" * 70)
    print(f"Total versions: {len(all_graphs)}")
    print(f"Training strategy: Leave-One-Out")
    print(f"Device: {config.DEVICE}")
    
    total_ranks = []
    version_names = []
    
    for i, test_graph in enumerate(all_graphs):
        fold_num = i + 1
        test_version = test_graph.version_name if hasattr(test_graph, 'version_name') else f"version_{i}"
        
        print(f"\n{'='*70}")
        print(f"Fold {fold_num}/{len(all_graphs)}: Testing on {test_version}")
        print(f"{'='*70}")
        
        # Split data: test = current graph, train = all others
        train_graphs = all_graphs[:i] + all_graphs[i+1:]
        
        print(f"Train size: {len(train_graphs)}, Test size: 1")
        
        try:
            # Initialize model
            model = FALCONModel(
                input_dim=config.INPUT_DIM,
                hidden_dim=config.HIDDEN_DIM,
                embedding_dim=config.EMBEDDING_DIM,
                proj_hidden_dim=config.PROJECTION_HIDDEN_DIM,
                proj_output_dim=config.PROJECTION_OUTPUT_DIM,
                rank_hidden_dim=config.RANK_HIDDEN_DIM,
                num_gnn_layers=config.NUM_GNN_LAYERS,
                num_gnn_steps=config.NUM_GNN_STEPS,
                dropout=config.DROPOUT
            )
            
            # Initialize trainer
            trainer = FALCONTrainer(
                model=model,
                device=config.DEVICE,
                learning_rate_phase1=config.LEARNING_RATE_PHASE1,
                learning_rate_phase2=config.LEARNING_RATE_PHASE2,
                weight_decay=config.WEIGHT_DECAY,
                node_loss_weight=config.NODE_LOSS_WEIGHT,
                graph_loss_weight=config.GRAPH_LOSS_WEIGHT,
                temperature=config.TEMPERATURE,
                margin=config.MARGIN,
                augment_fn=augment_graph,
                freeze_encoder_phase2=config.FREEZE_ENCODER_PHASE2
            )
            
            # Phase 1: Representation Learning
            print(f"\n[Phase 1: Representation Learning]")
            trainer.train_phase1_representation(
                fail_graphs=train_graphs,
                pass_graphs=train_graphs,  # Use same as both (simplified)
                epochs=config.PHASE1_EPOCHS,
                drop_prob=config.AUGMENTATION_DROP_PROB,
                verbose=verbose
            )
            
            # Phase 2: Fault Localization
            print(f"\n[Phase 2: Fault Localization]")
            trainer.train_phase2_ranking(
                fail_graphs=train_graphs,
                epochs=config.PHASE2_EPOCHS,
                verbose=verbose
            )
            
            # Predict on test graph
            print(f"\n[Evaluation on Test Graph]")
            scores = trainer.predict(test_graph)
            
            # Find rank of bug
            rank = find_bug_rank(scores, test_graph.y)
            
            if rank > 0:
                total_ranks.append(rank)
                version_names.append(test_version)
                print(f"✓ Bug found at Rank: {rank}")
                
                # Show top-5 predictions
                top_indices, top_scores = trainer.predict(test_graph, return_top_k=5)
                print(f"\nTop-5 suspicious nodes:")
                for j, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
                    node_name = test_graph.node_names[idx] if hasattr(test_graph, 'node_names') else f"Node {idx}"
                    is_bug = "🐛" if test_graph.y[idx] == 1 else "  "
                    print(f"  {j}. {is_bug} {node_name}: {score:.4f}")
            else:
                print(f"✗ Bug not found (no faulty nodes or prediction failed)")
            
            # Save checkpoint if configured
            if config.SAVE_CHECKPOINTS:
                checkpoint_path = Path(config.CHECKPOINT_DIR) / f"fold_{fold_num}.pt"
                trainer.save_checkpoint(str(checkpoint_path), epoch=fold_num, phase='complete')
            
        except Exception as e:
            print(f"\n✗ Error in fold {fold_num}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup memory
            del model
            del trainer
            if config.DEVICE == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    
    return total_ranks, version_names


def main():
    """Main pipeline."""
    start_time = datetime.now()
    
    # Step 0: Setup
    setup_environment()
    
    # Step 1: Load ground truth
    ground_truth = load_ground_truth(config.RAW_DATA_PATH)
    
    # Initialize graph builder
    print(f"\nInitializing Graph Builder...")
    builder = GraphBuilder(
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        cache_dir=config.PROCESSED_PATH
    )
    print(f"✓ Graph Builder ready")
    
    # Step 2: Load all graphs with caching
    all_graphs = load_all_graphs(
        data_path=config.RAW_DATA_PATH,
        builder=builder,
        ground_truth=ground_truth,
        use_cache=config.USE_CACHE
    )
    
    if len(all_graphs) == 0:
        print("\nError: No valid graphs loaded. Exiting.")
        return 1
    
    # Step 3: Run LOOCV
    total_ranks, version_names = run_loocv(
        all_graphs=all_graphs,
        verbose=config.VERBOSE
    )
    
    # Step 4: Calculate and report metrics
    if len(total_ranks) == 0:
        print("\nError: No successful predictions. Cannot calculate metrics.")
        return 1
    
    print("\n" + "=" * 70)
    print("STEP 3: FINAL RESULTS")
    print("=" * 70)
    
    metrics = calculate_metrics(total_ranks, top_k_values=config.TOP_K_VALUES)
    print_metrics(metrics, "FALCON Final Results")
    
    # Rank distribution analysis
    analyze_rank_distribution(total_ranks)
    
    # Save results to CSV
    save_results_to_csv(
        rank_list=total_ranks,
        version_names=version_names,
        metrics=metrics,
        output_path=config.RESULT_CSV_PATH
    )
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)
    print(f"Start Time:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:    {duration}")
    print(f"Test Cases:  {len(total_ranks)}/{len(all_graphs)}")
    print(f"Results:     {config.RESULT_CSV_PATH}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

