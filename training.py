#!/usr/bin/env python3
"""
FALCON Training Script.
Trains the model using preprocessed graphs from processed_data/.

Usage:
    python training.py                        # Default: 80/20 train/test split
    python training.py --train_ratio 0.7     # 70% train, 30% test
    python training.py --epochs1 5 --epochs2 5  # Custom epochs
    python training.py --seed 123            # Different random seed
"""

import sys
import json
import torch
import gc
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from torch_geometric.data import Data

from src import config
from src.models import FALCONModel
from src.training import FALCONTrainer
from src.training.augmentation import augment_graph
from src.utils import calculate_metrics, print_metrics, save_results_to_csv, analyze_rank_distribution


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FALCON Training")
    
    # Data paths
    parser.add_argument(
        '--data_path',
        type=str,
        default=config.PROCESSED_PATH,
        help='Path to preprocessed data (.pt files)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./results',
        help='Path to save results'
    )
    
    # Train/Test split
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of data for training (default: 0.8 = 80%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Training hyperparameters
    parser.add_argument('--epochs1', type=int, default=config.PHASE1_EPOCHS, help='Phase 1 epochs')
    parser.add_argument('--epochs2', type=int, default=config.PHASE2_EPOCHS, help='Phase 2 epochs')
    parser.add_argument('--lr1', type=float, default=config.LEARNING_RATE_PHASE1, help='Phase 1 learning rate')
    parser.add_argument('--lr2', type=float, default=config.LEARNING_RATE_PHASE2, help='Phase 2 learning rate')
    parser.add_argument('--drop_prob', type=float, default=config.AUGMENTATION_DROP_PROB, help='Augmentation drop probability')
    
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=config.HIDDEN_DIM, help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=config.EMBEDDING_DIM, help='Embedding dimension')
    
    # Other options
    parser.add_argument('--device', type=str, default=config.DEVICE, help='Device (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    
    return parser.parse_args()


def load_processed_graphs(data_path: str, device: str = 'cpu', verbose: bool = True) -> List[Data]:
    """
    Load preprocessed graphs from .pt files.
    
    Args:
        data_path: Path to directory containing .pt files
        device: Device to load tensors to ('cuda' or 'cpu')
        verbose: Whether to print progress
    
    Returns:
        List of Data objects
    """
    data_dir = Path(data_path)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please run preprocessing first: python preprocess.py")
        return []
    
    # Find all .pt files (version files start with 'v')
    pt_files = sorted(data_dir.glob("v*.pt"))
    
    if verbose:
        print(f"Found {len(pt_files)} preprocessed graphs")
    
    graphs = []
    skipped = 0
    
    for pt_file in pt_files:
        try:
            # PyTorch 2.6+ requires weights_only=False for custom objects
            # map_location allows loading CUDA tensors on CPU or vice versa
            data = torch.load(pt_file, map_location=device, weights_only=False)
            
            # Store version name if not present
            if not hasattr(data, 'version_name'):
                data.version_name = pt_file.stem
            
            # Only include graphs with faulty nodes
            if hasattr(data, 'y') and data.y.sum() > 0:
                graphs.append(data)
            else:
                skipped += 1
                    
        except Exception as e:
            print(f"  Error loading {pt_file.stem}: {e}")
    
    if verbose:
        print(f"Loaded {len(graphs)} valid graphs (skipped {skipped} without faulty nodes)")
    
    return graphs


def find_bug_rank(scores: torch.Tensor, labels: torch.Tensor) -> int:
    """Find the rank of the first faulty node."""
    faulty_indices = (labels == 1).nonzero(as_tuple=True)[0]
    
    if len(faulty_indices) == 0:
        return -1
    
    ranked_indices = torch.argsort(scores, descending=True)
    
    faulty_ranks = []
    for faulty_idx in faulty_indices:
        position = (ranked_indices == faulty_idx).nonzero(as_tuple=True)[0]
        if len(position) > 0:
            rank = position[0].item() + 1
            faulty_ranks.append(rank)
    
    return min(faulty_ranks) if faulty_ranks else -1


def create_model(args) -> FALCONModel:
    """Create FALCON model with given arguments."""
    return FALCONModel(
        input_dim=config.INPUT_DIM,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        proj_hidden_dim=config.PROJECTION_HIDDEN_DIM,
        proj_output_dim=config.PROJECTION_OUTPUT_DIM,
        rank_hidden_dim=config.RANK_HIDDEN_DIM,
        num_gnn_layers=config.NUM_GNN_LAYERS,
        num_gnn_steps=config.NUM_GNN_STEPS,
        dropout=config.DROPOUT
    )


def create_trainer(model, args) -> FALCONTrainer:
    """Create FALCON trainer with given arguments."""
    return FALCONTrainer(
        model=model,
        device=args.device,
        learning_rate_phase1=args.lr1,
        learning_rate_phase2=args.lr2,
        weight_decay=config.WEIGHT_DECAY,
        node_loss_weight=config.NODE_LOSS_WEIGHT,
        graph_loss_weight=config.GRAPH_LOSS_WEIGHT,
        temperature=config.TEMPERATURE,
        margin=config.MARGIN,
        augment_fn=augment_graph,
        freeze_encoder_phase2=config.FREEZE_ENCODER_PHASE2,
        node_loss_chunk_size=config.NODE_LOSS_CHUNK_SIZE,
        node_loss_max_nodes=config.NODE_LOSS_MAX_NODES
    )


def train_and_evaluate(
    train_graphs: List[Data],
    test_graphs: List[Data],
    args
) -> Tuple[List[int], List[str]]:
    """
    Train model and evaluate on test set.
    
    Returns:
        Tuple of (rank_list, version_names)
    """
    # Create model and trainer
    model = create_model(args)
    trainer = create_trainer(model, args)
    
    print(f"\nModel created with {model.get_num_params():,} parameters")
    
    try:
        # Phase 1: Representation Learning
        print(f"\n{'='*70}")
        print(f"PHASE 1: Representation Learning ({args.epochs1} epochs)")
        print(f"{'='*70}")
        
        trainer.train_phase1_representation(
            fail_graphs=train_graphs,
            pass_graphs=train_graphs,
            epochs=args.epochs1,
            drop_prob=args.drop_prob,
            verbose=args.verbose
        )
        
        # Phase 2: Fault Localization
        print(f"\n{'='*70}")
        print(f"PHASE 2: Fault Localization ({args.epochs2} epochs)")
        print(f"{'='*70}")
        
        trainer.train_phase2_ranking(
            fail_graphs=train_graphs,
            epochs=args.epochs2,
            verbose=args.verbose
        )
        
        # Evaluation
        print(f"\n{'='*70}")
        print(f"EVALUATION ({len(test_graphs)} test graphs)")
        print(f"{'='*70}")
        
        total_ranks = []
        version_names = []
        
        for i, test_graph in enumerate(test_graphs):
            test_version = test_graph.version_name if hasattr(test_graph, 'version_name') else f"test_{i}"
            
            # Predict
            scores = trainer.predict(test_graph)
            rank = find_bug_rank(scores, test_graph.y)
            
            if rank > 0:
                total_ranks.append(rank)
                version_names.append(test_version)
                status = f"Rank {rank}" + (" ⭐" if rank <= 5 else "")
            else:
                status = "Not found"
            
            print(f"  [{i+1:2}/{len(test_graphs)}] {test_version}: {status}")
        
        return total_ranks, version_names
        
    finally:
        # Cleanup
        del model
        del trainer
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("FALCON TRAINING")
    print("=" * 70)
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Path: {args.data_path}")
    print(f"Device: {args.device}")
    print(f"Train Ratio: {args.train_ratio*100:.0f}%")
    print(f"Random Seed: {args.seed}")
    print(f"Phase 1 Epochs: {args.epochs1}")
    print(f"Phase 2 Epochs: {args.epochs2}")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load preprocessed graphs (always on CPU to save GPU memory)
    print("\n[Loading Preprocessed Graphs]")
    graphs = load_processed_graphs(args.data_path, device='cpu', verbose=args.verbose)
    
    if len(graphs) == 0:
        print("\nError: No valid graphs loaded.")
        print("Please run preprocessing first: python preprocess.py")
        return 1
    
    # Split into train/test
    print(f"\n[Splitting Data]")
    random.seed(args.seed)
    indices = list(range(len(graphs)))
    random.shuffle(indices)
    
    train_size = int(len(graphs) * args.train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_graphs = [graphs[i] for i in train_indices]
    test_graphs = [graphs[i] for i in test_indices]
    
    print(f"Total graphs: {len(graphs)}")
    print(f"Train set: {len(train_graphs)} ({args.train_ratio*100:.0f}%)")
    print(f"Test set: {len(test_graphs)} ({(1-args.train_ratio)*100:.0f}%)")
    
    # Train and evaluate
    total_ranks, version_names = train_and_evaluate(train_graphs, test_graphs, args)
    
    # Calculate metrics
    if len(total_ranks) == 0:
        print("\nError: No successful predictions.")
        return 1
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    metrics = calculate_metrics(total_ranks, top_k_values=config.TOP_K_VALUES)
    print_metrics(metrics, "FALCON Results")
    
    # Rank distribution
    analyze_rank_distribution(total_ranks)
    
    # Save results
    timestamp_str = start_time.strftime('%Y%m%d_%H%M%S')
    result_csv_path = output_path / f"falcon_results_{timestamp_str}.csv"
    save_results_to_csv(
        rank_list=total_ranks,
        version_names=version_names,
        metrics=metrics,
        output_path=str(result_csv_path)
    )
    
    # Save full results as JSON
    result_json_path = output_path / f"falcon_results_{timestamp_str}.json"
    full_results = {
        'timestamp': start_time.isoformat(),
        'train_ratio': args.train_ratio,
        'seed': args.seed,
        'epochs1': args.epochs1,
        'epochs2': args.epochs2,
        'train_size': len(train_graphs),
        'test_size': len(test_graphs),
        'metrics': metrics,
        'ranks': list(zip(version_names, total_ranks))
    }
    
    with open(result_json_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Duration: {duration}")
    print(f"Results: {result_csv_path}")
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

