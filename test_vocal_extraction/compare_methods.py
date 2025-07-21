#!/usr/bin/env python3
"""
Compare Spleeter and Demucs vocal extraction methods.

This script runs both Spleeter and Demucs on the same audio file and compares
the results in terms of speed and quality.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Import the extraction functions from the test scripts
from test_spleeter import extract_vocals_with_spleeter
from test_demucs import extract_vocals_with_demucs

def main():
    parser = argparse.ArgumentParser(description="Compare Spleeter and Demucs vocal extraction")
    parser.add_argument("input_file", help="Path to the input audio file")
    parser.add_argument("--output", help="Output directory (default: test_results in current directory)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return 1
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    output_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "test_results", 
        base_name
    )
    
    # Create output directories for each method
    spleeter_dir = os.path.join(output_dir, "spleeter")
    demucs_dir = os.path.join(output_dir, "demucs")
    
    # Run Spleeter
    print("\n=== Running Spleeter ===")
    spleeter_result, spleeter_time = extract_vocals_with_spleeter(args.input_file, spleeter_dir)
    
    # Run Demucs
    print("\n=== Running Demucs ===")
    demucs_result, demucs_time = extract_vocals_with_demucs(args.input_file, demucs_dir)
    
    # Print comparison
    print("\n=== Comparison Results ===")
    print(f"Spleeter execution time: {spleeter_time:.2f} seconds")
    print(f"Demucs execution time: {demucs_time:.2f} seconds")
    print(f"Speed ratio: Demucs is {demucs_time/spleeter_time:.1f}x slower than Spleeter")
    
    # Print file locations for manual quality comparison
    print("\n=== Output Files for Quality Comparison ===")
    if spleeter_result:
        print(f"Spleeter vocals: {spleeter_result}")
    if demucs_result:
        print(f"Demucs vocals: {demucs_result}")
    
    print("\nPlease listen to both outputs and compare the quality of vocal isolation.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
