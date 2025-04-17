import os
import pandas as pd
import re
from pathlib import Path

def process_service_window_file(file_path):
    """
    Process a service_window_mwh.csv file to add the required columns.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the data
        df = pd.read_csv(file_path)
        
        # Skip if already processed
        if 'month_clean' in df.columns and 'day' in df.columns and 'time_window' in df.columns:
            print(f"File already processed: {file_path}")
            return True
        
        # Create month_clean column - extract just the month name
        df['month_clean'] = df['Month'].str.extract(r'(\w+)').fillna('Unknown')
        
        # Extract day and time_window from Window column
        # Window format example: "Weekday 10:00-10:30" or "Monday 10:00-10:30"
        df['day'] = df['Window'].str.extract(r'(\w+)').fillna('Unknown')
        df['time_window'] = df['Window'].str.extract(r'(\d+:\d+-\d+:\d+)').fillna('Unknown')
        
        # Save the processed data back to the same file
        df.to_csv(file_path, index=False)
        return True
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return False

def find_and_process_csv_files(base_dir):
    """
    Recursively find and process all service_window_mwh.csv files.
    
    Args:
        base_dir: The base directory to start searching from
    
    Returns:
        Dictionary with counts of processed and failed files
    """
    results = {
        'processed': 0,
        'failed': 0,
        'total': 0
    }
    
    # Walk through all directories
    for root, dirs, files in os.walk(base_dir):
        # Check if service_window_mwh.csv exists in this directory
        if 'service_window_mwh.csv' in files:
            file_path = os.path.join(root, 'service_window_mwh.csv')
            print(f"Processing: {file_path}")
            
            results['total'] += 1
            
            if process_service_window_file(file_path):
                results['processed'] += 1
            else:
                results['failed'] += 1
    
    return results

def main():
    # Base directory containing the financial year folders
    base_dir = "output/multi_site_results"
    
    print(f"Starting to process service_window_mwh.csv files in {base_dir}")
    
    # Process all files
    results = find_and_process_csv_files(base_dir)
    
    # Print summary
    print("\nProcessing Complete:")
    print(f"Total files found: {results['total']}")
    print(f"Successfully processed: {results['processed']}")
    print(f"Failed: {results['failed']}")

if __name__ == "__main__":
    main()
