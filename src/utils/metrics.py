"""
Metrics module for FALCON evaluation.
Calculates Top-N, MFR, MRR metrics for fault localization.
"""

from typing import List, Dict
import numpy as np


def calculate_metrics(rank_list: List[int], top_k_values: List[int] = [1, 3, 5, 10]) -> Dict:
    """
    Calculate fault localization metrics from rank list.
    
    Args:
        rank_list: List of ranks (1-indexed) for each test case
                  Example: [1, 5, 1, 3, 2] means bug found at rank 1, 5, 1, 3, 2
        top_k_values: List of K values for Top-K metric
    
    Returns:
        Dictionary containing:
            - 'Top-N': Percentage of cases where rank <= N (for each N)
            - 'MFR': Mean First Rank (average rank)
            - 'MRR': Mean Reciprocal Rank (average of 1/rank)
            - 'count': Number of test cases
    """
    if not rank_list:
        return {
            'Top-1': 0.0,
            'Top-3': 0.0,
            'Top-5': 0.0,
            'Top-10': 0.0,
            'MFR': 0.0,
            'MRR': 0.0,
            'count': 0
        }
    
    rank_array = np.array(rank_list)
    num_cases = len(rank_list)
    
    metrics = {}
    
    # Top-K Accuracy: % of cases where rank <= K
    for k in top_k_values:
        top_k_count = np.sum(rank_array <= k)
        top_k_acc = (top_k_count / num_cases) * 100
        metrics[f'Top-{k}'] = top_k_acc
    
    # MFR: Mean First Rank (lower is better)
    mfr = np.mean(rank_array)
    metrics['MFR'] = mfr
    
    # MRR: Mean Reciprocal Rank (higher is better)
    reciprocal_ranks = 1.0 / rank_array
    mrr = np.mean(reciprocal_ranks)
    metrics['MRR'] = mrr
    
    # Additional statistics
    metrics['count'] = num_cases
    metrics['median_rank'] = np.median(rank_array)
    metrics['min_rank'] = np.min(rank_array)
    metrics['max_rank'] = np.max(rank_array)
    
    return metrics


def print_metrics(metrics: Dict, title: str = "Evaluation Results"):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary from calculate_metrics()
        title: Title for the output
    """
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)
    
    # Top-K metrics
    print("\nTop-K Accuracy (%):")
    for key in sorted(metrics.keys()):
        if key.startswith('Top-'):
            print(f"  {key:10} {metrics[key]:6.2f}%")
    
    # Ranking metrics
    print("\nRanking Metrics:")
    if 'MFR' in metrics:
        print(f"  {'MFR':10} {metrics['MFR']:6.2f}  (Mean First Rank)")
    if 'MRR' in metrics:
        print(f"  {'MRR':10} {metrics['MRR']:6.4f}  (Mean Reciprocal Rank)")
    
    # Statistics
    print("\nStatistics:")
    if 'count' in metrics:
        print(f"  {'Count':10} {metrics['count']:6.0f}  (Number of test cases)")
    if 'median_rank' in metrics:
        print(f"  {'Median':10} {metrics['median_rank']:6.1f}")
    if 'min_rank' in metrics:
        print(f"  {'Min Rank':10} {metrics['min_rank']:6.0f}")
    if 'max_rank' in metrics:
        print(f"  {'Max Rank':10} {metrics['max_rank']:6.0f}")
    
    print("=" * 70)


def save_results_to_csv(
    rank_list: List[int],
    version_names: List[str],
    metrics: Dict,
    output_path: str
):
    """
    Save detailed results to CSV file.
    
    Args:
        rank_list: List of ranks
        version_names: List of version/test case names
        metrics: Dictionary from calculate_metrics()
        output_path: Path to save CSV file
    """
    import csv
    from pathlib import Path
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['Version', 'Rank', 'Reciprocal_Rank', 'Top1', 'Top3', 'Top5', 'Top10'])
        
        # Write per-version results
        for version, rank in zip(version_names, rank_list):
            reciprocal_rank = 1.0 / rank
            top1 = 1 if rank <= 1 else 0
            top3 = 1 if rank <= 3 else 0
            top5 = 1 if rank <= 5 else 0
            top10 = 1 if rank <= 10 else 0
            
            writer.writerow([version, rank, reciprocal_rank, top1, top3, top5, top10])
        
        # Write summary metrics
        writer.writerow([])
        writer.writerow(['Summary Metrics'])
        writer.writerow(['Metric', 'Value'])
        
        for key, value in metrics.items():
            if key != 'count':  # Skip count in percentage metrics
                writer.writerow([key, f"{value:.4f}"])
    
    print(f"\n✓ Results saved to: {output_path}")


def compare_methods(
    results_dict: Dict[str, Dict],
    metric_keys: List[str] = ['Top-1', 'Top-5', 'MFR', 'MRR']
):
    """
    Compare multiple methods side by side.
    
    Args:
        results_dict: Dictionary mapping method_name -> metrics_dict
        metric_keys: List of metric keys to compare
    
    Example:
        results = {
            'FALCON': {'Top-1': 65.0, 'Top-5': 90.0, 'MFR': 2.3, 'MRR': 0.73},
            'Tarantula': {'Top-1': 45.0, 'Top-5': 75.0, 'MFR': 4.1, 'MRR': 0.52}
        }
        compare_methods(results)
    """
    print("\n" + "=" * 70)
    print("METHOD COMPARISON")
    print("=" * 70)
    
    # Print header
    methods = list(results_dict.keys())
    header = f"{'Metric':<15}"
    for method in methods:
        header += f"{method:>15}"
    print(header)
    print("-" * 70)
    
    # Print each metric
    for metric_key in metric_keys:
        row = f"{metric_key:<15}"
        for method in methods:
            if metric_key in results_dict[method]:
                value = results_dict[method][metric_key]
                if metric_key.startswith('Top-'):
                    row += f"{value:>14.2f}%"
                else:
                    row += f"{value:>15.4f}"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    print("=" * 70)


def analyze_rank_distribution(rank_list: List[int]):
    """
    Analyze and print rank distribution.
    
    Args:
        rank_list: List of ranks
    """
    import numpy as np
    
    print("\n" + "=" * 70)
    print("RANK DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    rank_array = np.array(rank_list)
    
    # Histogram bins
    bins = [1, 2, 3, 4, 5, 10, 20, 50, 100, float('inf')]
    labels = ['1', '2', '3', '4', '5', '6-10', '11-20', '21-50', '51-100', '>100']
    
    print(f"\n{'Rank Range':<15} {'Count':<10} {'Percentage':<15} {'Bar'}")
    print("-" * 70)
    
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        
        if upper == float('inf'):
            count = np.sum(rank_array > lower)
        else:
            count = np.sum((rank_array >= lower) & (rank_array <= upper))
        
        percentage = (count / len(rank_list)) * 100
        bar = '█' * int(percentage / 2)  # Scale bar to max 50 chars
        
        print(f"{labels[i]:<15} {count:<10} {percentage:>6.2f}%        {bar}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Test the metrics module
    print("Testing FALCON Metrics Module...")
    
    # Example rank list (simulated results)
    print("\n1. Testing calculate_metrics():")
    rank_list = [1, 1, 2, 3, 1, 5, 2, 1, 4, 3, 1, 8, 2, 1, 6, 
                 3, 1, 2, 4, 1, 5, 1, 3, 2, 1]
    
    metrics = calculate_metrics(rank_list)
    print_metrics(metrics, "Test Results")
    
    # Test rank distribution
    print("\n2. Testing rank distribution:")
    analyze_rank_distribution(rank_list)
    
    # Test comparison
    print("\n3. Testing method comparison:")
    results = {
        'FALCON': {'Top-1': 65.0, 'Top-5': 90.0, 'MFR': 2.3, 'MRR': 0.73},
        'Tarantula': {'Top-1': 45.0, 'Top-5': 75.0, 'MFR': 4.1, 'MRR': 0.52},
        'Ochiai': {'Top-1': 50.0, 'Top-5': 80.0, 'MFR': 3.5, 'MRR': 0.61}
    }
    compare_methods(results)
    
    # Test CSV export
    print("\n4. Testing CSV export:")
    version_names = [f"v1-{i}" for i in range(len(rank_list))]
    save_results_to_csv(
        rank_list=rank_list,
        version_names=version_names,
        metrics=metrics,
        output_path="./test_results.csv"
    )
    
    print("\n✓ All metric tests passed!")

