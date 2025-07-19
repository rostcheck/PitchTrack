#!/usr/bin/env python3
"""
Analyze pitch detection results and compare algorithms.
This script processes pitch detection results and generates comparative visualizations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from scipy.stats import pearsonr
import pandas as pd

def load_results(file_path):
    """Load pitch detection results from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data["times"], data["pitches"], data["confidences"]

def calculate_metrics(reference_pitches, detected_pitches, confidence_threshold=0.5, confidences=None):
    """
    Calculate accuracy metrics between reference and detected pitches.
    
    Parameters:
    - reference_pitches: Ground truth pitch values
    - detected_pitches: Detected pitch values
    - confidence_threshold: Threshold for confidence filtering
    - confidences: Confidence values for filtering
    
    Returns:
    - Dictionary of metrics
    """
    # Filter by confidence if provided
    if confidences is not None:
        valid_indices = [i for i, conf in enumerate(confidences) if conf >= confidence_threshold]
        if valid_indices:
            reference_filtered = [reference_pitches[i] for i in valid_indices]
            detected_filtered = [detected_pitches[i] for i in valid_indices]
        else:
            reference_filtered = []
            detected_filtered = []
    else:
        reference_filtered = reference_pitches
        detected_filtered = detected_pitches
    
    # Skip if no valid data points
    if not reference_filtered or not detected_filtered:
        return {
            "mae": float('nan'),
            "rmse": float('nan'),
            "correlation": float('nan'),
            "valid_points": 0,
            "total_points": len(reference_pitches)
        }
    
    # Calculate mean absolute error
    errors = [abs(ref - det) for ref, det in zip(reference_filtered, detected_filtered)]
    mae = sum(errors) / len(errors)
    
    # Calculate root mean square error
    squared_errors = [(ref - det) ** 2 for ref, det in zip(reference_filtered, detected_filtered)]
    rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5
    
    # Calculate correlation coefficient
    try:
        correlation, _ = pearsonr(reference_filtered, detected_filtered)
    except:
        correlation = float('nan')
    
    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
        "valid_points": len(reference_filtered),
        "total_points": len(reference_pitches)
    }

def compare_algorithms(reference_file, algorithm_files, output_dir):
    """
    Compare multiple pitch detection algorithms against a reference.
    
    Parameters:
    - reference_file: Path to reference pitch data
    - algorithm_files: List of paths to algorithm results
    - output_dir: Directory to save comparison results
    """
    # Load reference data
    ref_times, ref_pitches, _ = load_results(reference_file)
    
    # Results dictionary
    results = {}
    
    # Load algorithm data and calculate metrics
    for algo_file in algorithm_files:
        algo_name = os.path.basename(algo_file).split('.')[0]
        algo_times, algo_pitches, algo_confidences = load_results(algo_file)
        
        # Interpolate algorithm results to match reference time points
        interp_pitches = np.interp(ref_times, algo_times, algo_pitches)
        
        # Calculate metrics
        metrics = calculate_metrics(ref_pitches, interp_pitches)
        results[algo_name] = metrics
    
    # Create comparison table
    comparison_df = pd.DataFrame.from_dict(results, orient='index')
    comparison_df = comparison_df.sort_values('rmse')
    
    # Save comparison table
    table_path = os.path.join(output_dir, "algorithm_comparison.csv")
    comparison_df.to_csv(table_path)
    print(f"Comparison table saved to {table_path}")
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot reference pitch
    plt.plot(ref_times, ref_pitches, 'k-', label='Reference', linewidth=2, alpha=0.7)
    
    # Plot algorithm pitches
    colors = plt.cm.tab10.colors
    for i, algo_file in enumerate(algorithm_files):
        algo_name = os.path.basename(algo_file).split('.')[0]
        algo_times, algo_pitches, _ = load_results(algo_file)
        plt.plot(algo_times, algo_pitches, color=colors[i % len(colors)], 
                 label=f"{algo_name} (RMSE: {results[algo_name]['rmse']:.2f}Hz)", 
                 linewidth=1, alpha=0.6)
    
    plt.title("Algorithm Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(True)
    
    # Save comparison plot
    plot_path = os.path.join(output_dir, "algorithm_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_path}")
    plt.close()

def analyze_confidence_impact(result_file, output_dir):
    """
    Analyze the impact of confidence threshold on pitch detection accuracy.
    
    Parameters:
    - result_file: Path to pitch detection results
    - output_dir: Directory to save analysis results
    """
    # Load data
    times, pitches, confidences = load_results(result_file)
    
    # Calculate metrics at different confidence thresholds
    thresholds = np.arange(0, 1.01, 0.1)
    threshold_metrics = []
    
    for threshold in thresholds:
        valid_indices = [i for i, conf in enumerate(confidences) if conf >= threshold]
        valid_count = len(valid_indices)
        coverage = valid_count / len(pitches) if pitches else 0
        
        threshold_metrics.append({
            "threshold": threshold,
            "valid_points": valid_count,
            "coverage": coverage
        })
    
    # Create threshold impact table
    threshold_df = pd.DataFrame(threshold_metrics)
    
    # Save threshold impact table
    table_path = os.path.join(output_dir, "confidence_threshold_impact.csv")
    threshold_df.to_csv(table_path, index=False)
    print(f"Confidence threshold analysis saved to {table_path}")
    
    # Create threshold impact plot
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df["threshold"], threshold_df["coverage"], 'bo-')
    plt.title("Impact of Confidence Threshold on Data Coverage")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Data Coverage (%)")
    plt.grid(True)
    plt.ylim(0, 1.05)
    
    # Add percentage labels
    for i, row in threshold_df.iterrows():
        plt.annotate(f"{row['coverage']*100:.1f}%", 
                     (row['threshold'], row['coverage']),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center')
    
    # Save threshold impact plot
    plot_path = os.path.join(output_dir, "confidence_threshold_impact.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Confidence threshold plot saved to {plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze pitch detection results')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare algorithms against reference')
    compare_parser.add_argument('reference', help='Path to reference pitch data')
    compare_parser.add_argument('algorithms', nargs='+', help='Paths to algorithm results')
    compare_parser.add_argument('--output-dir', default=None, help='Directory to save comparison results')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze confidence impact')
    analyze_parser.add_argument('result_file', help='Path to pitch detection results')
    analyze_parser.add_argument('--output-dir', default=None, help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        # Default to results directory in project
        args.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.command == 'compare':
        compare_algorithms(args.reference, args.algorithms, args.output_dir)
    elif args.command == 'analyze':
        analyze_confidence_impact(args.result_file, args.output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
