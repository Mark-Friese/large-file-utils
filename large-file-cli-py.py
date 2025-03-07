#!/usr/bin/env python
"""
large_file_cli.py

Command-line interface for large file processing utilities.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import time
import tempfile
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import our modules
try:
    from dependency_installer import DependencyInstaller
    from large_file_handler import LargeFileHandler
    from file_optimizer import FileOptimizer
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.info("Make sure all necessary files are in the same directory.")
    sys.exit(1)

def check_dependencies(args):
    """Check dependencies and optionally install missing ones."""
    print("\nChecking dependencies for large file handling...")
    DependencyInstaller.print_status_report()
    
    if args.install:
        print("\nInstalling missing dependencies...")
        successful, failed = DependencyInstaller.install_missing_dependencies(
            include_optional=args.optional,
            upgrade=args.upgrade
        )
        
        if successful:
            print(f"Successfully installed: {', '.join(successful)}")
        
        if failed:
            print(f"Failed to install: {', '.join(failed)}")
        
        print("\nUpdated dependency status:")
        DependencyInstaller.print_status_report()

def analyze_file(args):
    """Analyze a file to determine its characteristics."""
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return
    
    print(f"\nAnalyzing file: {args.file}")
    
    file_ext = Path(args.file).suffix.lower()
    if file_ext == '.csv':
        analysis = FileOptimizer.analyze_csv(args.file)
        approach = FileOptimizer.determine_optimal_approach(analysis)
        
        # Print analysis
        print("\nFile Analysis:")
        print(f"  File size: {analysis['file_size'] / (1024*1024):.2f} MB")
        print(f"  Size category: {analysis['size_category']}")
        print(f"  Estimated rows: {analysis['estimated_row_count']:,}")
        print(f"  Columns: {analysis['column_count']}")
        
        if 'columns' in analysis and args.verbose:
            print(f"  Column names: {', '.join(analysis['columns'][:10])}...")
        
        print("\nRecommended Approach:")
        print(f"  Approach: {approach['approach']}")
        print(f"  Engine: {approach['engine']}")
        print(f"  Use chunks: {approach.get('use_chunks', False)}")
        print(f"  Convert to Parquet: {approach.get('convert_to_parquet', False)}")
        print(f"  Estimated memory: {approach.get('estimated_memory', 0) / (1024*1024):.2f} MB")
        
        # Pyarrow status
        try:
            import pyarrow
            print(f"  PyArrow available: Yes (version {pyarrow.__version__})")
        except ImportError:
            print("  PyArrow available: No")
        
        # Dask status
        try:
            import dask
            print(f"  Dask available: Yes (version {dask.__version__})")
        except ImportError:
            print("  Dask available: No")
    else:
        print(f"File analysis not supported for {file_ext} files.")

def convert_file(args):
    """Convert a file to another format."""
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    
    input_path = Path(args.input)
    input_ext = input_path.suffix.lower()
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate based on input name and target format
        output_path = input_path.with_suffix(f'.{args.format}')
    
    # Check if output exists and handle force flag
    if output_path.exists() and not args.force:
        logger.error(f"Output file already exists: {output_path}")
        logger.info("Use --force to overwrite existing files.")
        return
    
    print(f"\nConverting {input_path} to {output_path}...")
    
    start_time = time.time()
    
    try:
        if input_ext == '.csv' and args.format == 'parquet':
            # Convert CSV to Parquet
            result_path = FileOptimizer.convert_csv_to_parquet(input_path, output_path)
            print(f"Successfully converted to Parquet: {result_path}")
        else:
            logger.error(f"Conversion from {input_ext} to {args.format} not supported.")
            return
        
        duration = time.time() - start_time
        file_size = os.path.getsize(output_path) / (1024*1024)
        
        print(f"Conversion completed in {duration:.2f}s")
        print(f"Output file size: {file_size:.2f} MB")
        
        # Show compression ratio if applicable
        if input_path.exists():
            input_size = os.path.getsize(input_path) / (1024*1024)
            ratio = input_size / file_size if file_size > 0 else 0
            print(f"Compression ratio: {ratio:.2f}x")
    
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")

def process_file(args):
    """Process a file using optimized approaches."""
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    
    input_path = Path(args.input)
    
    print(f"\nProcessing file: {input_path}")
    
    # Parse filter argument if provided
    filters = {}
    if args.filter:
        for filter_str in args.filter:
            try:
                column, value = filter_str.split('=', 1)
                filters[column.strip()] = value.strip()
            except ValueError:
                logger.error(f"Invalid filter format: {filter_str}")
                logger.info("Filters should be in the format: column=value")
                return
    
    # Parse columns argument if provided
    columns = None
    if args.columns:
        columns = [col.strip() for col in args.columns.split(',')]
    
    try:
        # Load the file with the optimized approach
        print("Loading file...")
        start_time = time.time()
        
        df = FileOptimizer.load_optimized(
            input_path,
            columns=columns,
            filters=filters
        )
        
        load_duration = time.time() - start_time
        print(f"Loaded file in {load_duration:.2f}s")
        
        # Display basic information
        try:
            row_count = len(df)
            col_count = len(df.columns)
            print(f"\nLoaded data: {row_count:,} rows, {col_count} columns")
            
            if args.info:
                # Display memory usage
                memory_usage = df.memory_usage(deep=True).sum() / (1024*1024)
                print(f"Memory usage: {memory_usage:.2f} MB")
                
                # Display column info
                print("\nColumn information:")
                col_info = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    null_count = df[col].isna().sum()
                    null_pct = 100 * null_count / row_count if row_count > 0 else 0
                    col_info.append((col, dtype, null_count, null_pct))
                
                # Sort by null percentage (descending)
                col_info.sort(key=lambda x: x[3], reverse=True)
                
                # Print column info
                for col, dtype, null_count, null_pct in col_info[:10]:  # Show top 10
                    print(f"  {col}: {dtype}, {null_count:,} nulls ({null_pct:.1f}%)")
                
                if len(col_info) > 10:
                    print(f"  ... and {len(col_info) - 10} more columns")
            
            # Save processed file if output specified
            if args.output:
                output_path = Path(args.output)
                
                # Determine output format
                output_format = 'parquet'  # Default
                if output_path.suffix.lower() == '.csv':
                    output_format = 'csv'
                elif output_path.suffix.lower() == '.xlsx':
                    output_format = 'excel'
                elif output_path.suffix.lower() == '.json':
                    output_format = 'json'
                
                print(f"\nSaving processed data to {output_path}...")
                save_start = time.time()
                
                FileOptimizer.write_optimized(
                    df,
                    output_path,
                    format=output_format
                )
                
                save_duration = time.time() - save_start
                print(f"Saved file in {save_duration:.2f}s")
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Large file processing utilities'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Dependencies command
    deps_parser = subparsers.add_parser('deps', help='Check and install dependencies')
    deps_parser.add_argument('--install', action='store_true', help='Install missing dependencies')
    deps_parser.add_argument('--optional', action='store_true', help='Include optional dependencies')
    deps_parser.add_argument('--upgrade', action='store_true', help='Upgrade existing dependencies')
    deps_parser.set_defaults(func=check_dependencies)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a file')
    analyze_parser.add_argument('file', help='File to analyze')
    analyze_parser.add_argument('--verbose', '-v', action='store_true', help='Show more details')
    analyze_parser.set_defaults(func=analyze_file)
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert file format')
    convert_parser.add_argument('input', help='Input file path')
    convert_parser.add_argument('--output', '-o', help='Output file path')
    convert_parser.add_argument('--format', '-f', default='parquet', choices=['parquet', 'csv'], 
                               help='Output format')
    convert_parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    convert_parser.set_defaults(func=convert_file)
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Load and process a file')
    process_parser.add_argument('input', help='Input file path')
    process_parser.add_argument('--output', '-o', help='Output file path for processed data')
    process_parser.add_argument('--filter', '-f', action='append', 
                              help='Filter data (format: column=value), can be used multiple times')
    process_parser.add_argument('--columns', '-c', help='Columns to load (comma-separated)')
    process_parser.add_argument('--info', '-i', action='store_true', help='Show data information')
    process_parser.set_defaults(func=process_file)
    
    # Parse arguments
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()