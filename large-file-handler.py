"""
large_file_handler.py

Provides utilities for efficiently handling large CSV files.
Implements chunked processing and memory-efficient loading.
"""

import pandas as pd
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
from pathlib import Path
import tempfile
import os
import logging
from typing import Optional, Dict, Union, Callable, List, Tuple

logger = logging.getLogger(__name__)

class LargeFileHandler:
    """Provides utilities for efficiently handling large CSV files."""
    
    # Default chunk size (number of rows)
    DEFAULT_CHUNK_SIZE = 100_000
    
    # Default memory limit (in bytes) - 500MB
    DEFAULT_MEMORY_LIMIT = 500 * 1024 * 1024
    
    @staticmethod
    def estimate_csv_size(file_path: Union[str, Path]) -> int:
        """
        Estimate the size of a CSV file in bytes.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            int: Estimated file size in bytes
        """
        return os.path.getsize(file_path)
    
    @classmethod
    def convert_to_parquet(
        cls,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        chunk_size: Optional[int] = None
    ) -> str:
        """
        Convert a large CSV file to Parquet format for more efficient processing.
        
        Args:
            file_path: Path to the CSV file
            output_path: Optional path to save the Parquet file (default: temp file)
            chunk_size: Optional chunk size for processing (default: DEFAULT_CHUNK_SIZE)
            
        Returns:
            str: Path to the Parquet file
        """
        chunk_size = chunk_size or cls.DEFAULT_CHUNK_SIZE
        
        # Create temporary file if output_path not provided
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = Path(temp_dir) / f"{Path(file_path).stem}.parquet"
        
        # Convert to Path object if string
        if isinstance(output_path, str):
            output_path = Path(output_path)
            
        logger.info(f"Converting CSV to Parquet: {file_path} -> {output_path}")
        
        # Use PyArrow for efficient conversion
        try:
            # Read CSV with PyArrow
            csv_parse_options = csv.ParseOptions(delimiter=',')
            csv_convert_options = csv.ConvertOptions(
                include_columns=None,  # Include all columns
                include_missing_columns=True,
                auto_dict_encode=True  # Automatically dictionary-encode string columns
            )
            
            table = csv.read_csv(
                file_path,
                parse_options=csv_parse_options,
                convert_options=csv_convert_options
            )
            
            # Write to Parquet
            pq.write_table(
                table,
                output_path,
                compression='snappy',  # Good balance of compression and speed
                row_group_size=chunk_size  # Optimize row groups for chunked access
            )
            
            logger.info(f"Successfully converted CSV to Parquet: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error converting CSV to Parquet: {str(e)}")
            
            # Fallback to pandas + pyarrow for conversion
            logger.info("Falling back to pandas for conversion...")
            
            # Process in chunks to handle large files
            for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                if i == 0:
                    # First chunk, create the file
                    chunk.to_parquet(
                        output_path,
                        engine='pyarrow',
                        compression='snappy',
                        index=False
                    )
                else:
                    # Append to existing file
                    chunk.to_parquet(
                        output_path,
                        engine='pyarrow',
                        compression='snappy',
                        index=False,
                        append=True
                    )
            
            logger.info(f"Successfully converted CSV to Parquet using pandas: {output_path}")
            return str(output_path)
    
    @classmethod
    def load_large_csv(
        cls,
        file_path: Union[str, Path],
        use_dask: bool = True,
        memory_limit: Optional[int] = None,
        convert_to_parquet_first: bool = True
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Load a large CSV file efficiently, optionally converting to Parquet first.
        
        Args:
            file_path: Path to the CSV file
            use_dask: Whether to return a Dask DataFrame (True) or Pandas DataFrame (False)
            memory_limit: Optional memory limit in bytes (default: DEFAULT_MEMORY_LIMIT)
            convert_to_parquet_first: Whether to convert to Parquet for efficiency
            
        Returns:
            Union[pd.DataFrame, dd.DataFrame]: Loaded data
        """
        memory_limit = memory_limit or cls.DEFAULT_MEMORY_LIMIT
        
        # For small files, just use pandas directly
        file_size = cls.estimate_csv_size(file_path)
        if file_size < memory_limit and not use_dask and not convert_to_parquet_first:
            return pd.read_csv(file_path)
        
        # For larger files, convert to Parquet first if requested
        if convert_to_parquet_first:
            parquet_path = cls.convert_to_parquet(file_path)
            
            if use_dask:
                return dd.read_parquet(parquet_path)
            else:
                return pd.read_parquet(parquet_path)
        
        # Otherwise, use Dask directly on the CSV
        if use_dask:
            return dd.read_csv(file_path)
        
        # Use chunked processing with pandas for larger CSVs
        logger.info(f"Loading large CSV in chunks: {file_path}")
        chunks = []
        chunk_size = cls.DEFAULT_CHUNK_SIZE
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index=True)
    
    @classmethod
    def filter_dataframe(
        cls,
        df: Union[pd.DataFrame, dd.DataFrame],
        filter_column: str,
        filter_value: any
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Filter a DataFrame (either pandas or dask) by a column value.
        
        Args:
            df: DataFrame to filter
            filter_column: Column name to filter on
            filter_value: Value to filter for
            
        Returns:
            Filtered DataFrame
        """
        if isinstance(df, dd.DataFrame):
            return df[df[filter_column] == filter_value].compute()
        else:
            return df[df[filter_column] == filter_value].copy()
    
    @classmethod
    def process_large_file(
        cls,
        file_path: Union[str, Path],
        processor_func: Callable[[pd.DataFrame], pd.DataFrame],
        output_path: Optional[Union[str, Path]] = None,
        chunk_size: Optional[int] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process a large file in chunks using a custom processor function.
        
        Args:
            file_path: Path to the CSV file
            processor_func: Function to process each chunk
            output_path: Optional path to save the processed file
            chunk_size: Optional chunk size for processing
            **kwargs: Additional arguments to pass to the processor function
            
        Returns:
            Optional[str]: Path to the processed file if output_path is provided
        """
        chunk_size = chunk_size or cls.DEFAULT_CHUNK_SIZE
        
        # Create temporary file if output_path not provided
        if output_path is None:
            return None
            
        if isinstance(output_path, str):
            output_path = Path(output_path)
            
        logger.info(f"Processing large file in chunks: {file_path}")
        
        # Process in chunks
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
            # Process the chunk
            processed_chunk = processor_func(chunk, **kwargs)
            
            if i == 0:
                # First chunk, create the file
                processed_chunk.to_csv(output_path, index=False)
            else:
                # Append to existing file without header
                processed_chunk.to_csv(
                    output_path,
                    mode='a',
                    header=False,
                    index=False
                )
        
        logger.info(f"Successfully processed large file: {output_path}")
        return str(output_path)
