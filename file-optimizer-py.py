"""
file_optimizer.py

Provides automatic detection and optimization for different file types.
Automatically selects the best approach based on file size and content.
"""

import os
import pandas as pd
import logging
import tempfile
from pathlib import Path
from typing import Union, Optional, Dict, Tuple, List, Any
import csv
import json
import time

# Try to import optional dependencies
try:
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import modin.pandas as mpd
    MODIN_AVAILABLE = True
except ImportError:
    MODIN_AVAILABLE = False

logger = logging.getLogger(__name__)

class FileOptimizer:
    """
    Automatically detect and optimize file handling based on file characteristics.
    """
    
    # Size thresholds (in bytes)
    SMALL_FILE_THRESHOLD = 50 * 1024 * 1024    # 50 MB
    MEDIUM_FILE_THRESHOLD = 200 * 1024 * 1024  # 200 MB
    LARGE_FILE_THRESHOLD = 1 * 1024 * 1024 * 1024  # 1 GB
    
    # Row count thresholds
    SMALL_ROW_THRESHOLD = 100_000  # 100k rows
    MEDIUM_ROW_THRESHOLD = 1_000_000  # 1M rows
    LARGE_ROW_THRESHOLD = 10_000_000  # 10M rows
    
    # Default chunk sizes
    DEFAULT_CHUNK_SIZE = 100_000  # 100k rows
    
    @classmethod
    def get_file_size(cls, file_path: Union[str, Path]) -> int:
        """
        Get the size of a file in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            int: File size in bytes
        """
        try:
            return os.path.getsize(file_path)
        except Exception as e:
            logger.error(f"Error getting file size: {str(e)}")
            return 0
    
    @classmethod
    def estimate_row_count(cls, file_path: Union[str, Path], sample_size: int = 10) -> int:
        """
        Estimate the total number of rows in a CSV file.
        
        Args:
            file_path: Path to the CSV file
            sample_size: Number of rows to sample for estimation
            
        Returns:
            int: Estimated row count
        """
        try:
            file_size = cls.get_file_size(file_path)
            
            # If it's a small file, just count the lines
            if file_size < cls.SMALL_FILE_THRESHOLD:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return sum(1 for _ in f)
            
            # For larger files, estimate based on a sample
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read the header
                header = f.readline()
                header_size = len(header)
                
                # Read a sample of rows
                sample_rows = [f.readline() for _ in range(sample_size)]
                if not sample_rows:
                    return 0
                    
                # Calculate average row size
                avg_row_size = sum(len(row) for row in sample_rows) / len(sample_rows)
                
                # Estimate total rows
                data_size = file_size - header_size
                estimated_rows = int(data_size / avg_row_size)
                
                return estimated_rows
                
        except Exception as e:
            logger.error(f"Error estimating row count: {str(e)}")
            return 0
    
    @classmethod
    def analyze_csv(cls, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze a CSV file to determine its characteristics.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dict with file characteristics
        """
        try:
            # Get basic file info
            file_size = cls.get_file_size(file_path)
            size_category = "small"
            if file_size > cls.LARGE_FILE_THRESHOLD:
                size_category = "large"
            elif file_size > cls.MEDIUM_FILE_THRESHOLD:
                size_category = "medium"
            
            # Sample the file to get column info and estimate row count
            sample_df = pd.read_csv(file_path, nrows=5)
            columns = list(sample_df.columns)
            column_count = len(columns)
            
            # Estimate row count for larger files
            row_count = cls.estimate_row_count(file_path)
            
            # Determine row category
            row_category = "small"
            if row_count > cls.LARGE_ROW_THRESHOLD:
                row_category = "large"
            elif row_count > cls.MEDIUM_ROW_THRESHOLD:
                row_category = "medium"
            
            # Check for potentially problematic data types
            has_date_columns = any(
                'date' in col.lower() or 
                'time' in col.lower()
                for col in columns
            )
            
            return {
                'file_path': file_path,
                'file_size': file_size,
                'size_category': size_category,
                'estimated_row_count': row_count,
                'row_category': row_category,
                'column_count': column_count,
                'columns': columns,
                'has_date_columns': has_date_columns,
            }
            
        except Exception as e:
            logger.error(f"Error analyzing CSV file: {str(e)}")
            return {
                'file_path': file_path,
                'error': str(e)
            }
    
    @classmethod
    def determine_optimal_approach(cls, file_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the optimal approach for loading and processing a file.
        
        Args:
            file_analysis: File analysis dictionary from analyze_csv
            
        Returns:
            Dict with recommended approach
        """
        # Check for errors in analysis
        if 'error' in file_analysis:
            return {
                'approach': 'standard',
                'engine': 'pandas',
                'use_chunks': False,
                'convert_to_parquet': False,
                'error': file_analysis['error']
            }
        
        # Default approach for small files
        if file_analysis['size_category'] == 'small':
            return {
                'approach': 'standard',
                'engine': 'pandas',
                'use_chunks': False,
                'convert_to_parquet': False,
                'estimated_memory': file_analysis['file_size'] * 5  # Rough estimate
            }
        
        # For medium files
        if file_analysis['size_category'] == 'medium':
            if PYARROW_AVAILABLE:
                # Use PyArrow for medium files
                return {
                    'approach': 'optimized',
                    'engine': 'pyarrow',
                    'use_chunks': False,
                    'convert_to_parquet': True,
                    'estimated_memory': file_analysis['file_size'] * 3
                }
            else:
                # Fallback to chunked pandas
                return {
                    'approach': 'chunked',
                    'engine': 'pandas',
                    'use_chunks': True,
                    'chunk_size': cls.DEFAULT_CHUNK_SIZE,
                    'convert_to_parquet': False,
                    'estimated_memory': file_analysis['file_size'] * 2
                }
        
        # For large files
        if file_analysis['size_category'] == 'large':
            if DASK_AVAILABLE:
                # Prefer Dask for large files if available
                return {
                    'approach': 'distributed',
                    'engine': 'dask',
                    'convert_to_parquet': True,
                    'use_chunks': False,  # Dask handles chunking internally
                    'estimated_memory': file_analysis['file_size'] * 1.5
                }
            elif PYARROW_AVAILABLE:
                # Use PyArrow with chunking for large files
                return {
                    'approach': 'optimized',
                    'engine': 'pyarrow',
                    'use_chunks': True,
                    'chunk_size': cls.DEFAULT_CHUNK_SIZE,
                    'convert_to_parquet': True,
                    'estimated_memory': file_analysis['file_size'] * 2
                }
            else:
                # Fallback to highly chunked pandas
                return {
                    'approach': 'chunked',
                    'engine': 'pandas',
                    'use_chunks': True,
                    'chunk_size': cls.DEFAULT_CHUNK_SIZE // 2,  # Smaller chunks
                    'convert_to_parquet': False,
                    'estimated_memory': file_analysis['file_size'] * 3
                }
        
        # Default fallback
        return {
            'approach': 'standard',
            'engine': 'pandas',
            'use_chunks': False,
            'convert_to_parquet': False,
            'estimated_memory': file_analysis['file_size'] * 5
        }
    
    @classmethod
    def convert_csv_to_parquet(
        cls,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Convert CSV to Parquet format for better performance.
        
        Args:
            file_path: Path to the CSV file
            output_path: Optional output path for Parquet file
            
        Returns:
            str: Path to the created Parquet file
        """
        if not PYARROW_AVAILABLE:
            raise ImportError("PyArrow is required for CSV to Parquet conversion")
            
        # Create temp file if output path not provided
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = Path(temp_dir) / f"{Path(file_path).stem}.parquet"
        
        # Convert using PyArrow for best performance
        try:
            logger.info(f"Converting CSV to Parquet: {file_path} → {output_path}")
            
            read_options = pa_csv.ReadOptions(use_threads=True)
            parse_options = pa_csv.ParseOptions(delimiter=',')
            
            # Try to auto-detect types
            table = pa_csv.read_csv(
                file_path,
                read_options=read_options,
                parse_options=parse_options
            )
            
            # Write to parquet
            pq.write_table(
                table,
                output_path,
                compression='snappy',
                flavor='spark'  # For better compatibility
            )
            
            logger.info(f"Successfully converted CSV to Parquet: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error converting CSV to Parquet with PyArrow: {str(e)}")
            
            # Fallback to pandas
            try:
                logger.info("Falling back to pandas for CSV to Parquet conversion")
                
                # Analyze file to determine chunking needs
                file_analysis = cls.analyze_csv(file_path)
                approach = cls.determine_optimal_approach(file_analysis)
                
                if approach['use_chunks']:
                    # Process in chunks
                    chunk_size = approach.get('chunk_size', cls.DEFAULT_CHUNK_SIZE)
                    
                    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                        if i == 0:
                            # Write with new file
                            chunk.to_parquet(
                                output_path,
                                engine='pyarrow',
                                compression='snappy',
                                index=False
                            )
                        else:
                            # Append to file
                            chunk.to_parquet(
                                output_path,
                                engine='pyarrow',
                                compression='snappy',
                                index=False,
                                append=True
                            )
                else:
                    # Process all at once
                    df = pd.read_csv(file_path)
                    df.to_parquet(
                        output_path,
                        engine='pyarrow',
                        compression='snappy',
                        index=False
                    )
                
                logger.info(f"Successfully converted CSV to Parquet using pandas: {output_path}")
                return str(output_path)
                
            except Exception as e2:
                logger.error(f"Error in pandas fallback for Parquet conversion: {str(e2)}")
                raise RuntimeError(
                    f"Failed to convert CSV to Parquet: {str(e)} | Pandas fallback: {str(e2)}"
                )
    
    @classmethod
    def load_optimized(
        cls,
        file_path: Union[str, Path],
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[pd.DataFrame, Any]:
        """
        Load a file using the optimal approach based on file analysis.
        
        Args:
            file_path: Path to the file
            columns: Optional list of columns to load
            filters: Optional filters to apply during loading
            **kwargs: Additional arguments for loading
            
        Returns:
            Loaded data (pandas DataFrame, Dask DataFrame, etc.)
        """
        start_time = time.time()
        
        # Determine file type
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            # Analyze CSV file
            analysis = cls.analyze_csv(file_path)
            approach = cls.determine_optimal_approach(analysis)
            
            logger.info(
                f"File analysis: {file_path} - "
                f"Size: {analysis['size_category']} "
                f"({analysis['file_size'] / (1024*1024):.2f} MB), "
                f"Rows: ~{analysis['estimated_row_count']:,}"
            )
            
            logger.info(
                f"Optimal approach: {approach['approach']} - "
                f"Engine: {approach['engine']}, "
                f"Chunks: {approach.get('use_chunks', False)}"
            )
            
            engine = approach['engine']
            
            # Handle different engines
            if engine == 'dask':
                if not DASK_AVAILABLE:
                    logger.warning("Dask not available, falling back to pandas")
                    engine = 'pandas'
                    approach['use_chunks'] = True
                else:
                    if approach['convert_to_parquet']:
                        # Convert to parquet first for better performance
                        parquet_path = cls.convert_csv_to_parquet(file_path)
                        result = dd.read_parquet(parquet_path, columns=columns)
                    else:
                        # Read CSV directly with Dask
                        result = dd.read_csv(file_path, assume_missing=True)
                        
                        # Apply filters if provided
                        if filters:
                            for col, val in filters.items():
                                result = result[result[col] == val]
                    
                    duration = time.time() - start_time
                    logger.info(f"Loaded file with Dask in {duration:.2f}s")
                    return result
            
            elif engine == 'pyarrow':
                if not PYARROW_AVAILABLE:
                    logger.warning("PyArrow not available, falling back to pandas")
                    engine = 'pandas'
                    approach['use_chunks'] = True
                else:
                    if approach['convert_to_parquet']:
                        # Convert to parquet first
                        parquet_path = cls.convert_csv_to_parquet(file_path)
                        
                        # Read parquet with PyArrow
                        parquet_file = pq.ParquetFile(parquet_path)
                        
                        if approach['use_chunks']:
                            # Read in batches
                            batches = []
                            for batch in parquet_file.iter_batches(
                                columns=columns,
                                use_threads=True
                            ):
                                df_batch = batch.to_pandas()
                                
                                # Apply filters if provided
                                if filters:
                                    for col, val in filters.items():
                                        df_batch = df_batch[df_batch[col] == val]
                                
                                batches.append(df_batch)
                            
                            result = pd.concat(batches, ignore_index=True)
                        else:
                            # Read all at once
                            table = pq.read_table(parquet_path, columns=columns)
                            result = table.to_pandas()
                            
                            # Apply filters if provided
                            if filters:
                                for col, val in filters.items():
                                    result = result[result[col] == val]
                    else:
                        # Read CSV directly with PyArrow
                        table = pa_csv.read_csv(file_path)
                        result = table.to_pandas()
                        
                        # Apply filters if provided
                        if filters:
                            for col, val in filters.items():
                                result = result[result[col] == val]
                    
                    duration = time.time() - start_time
                    logger.info(f"Loaded file with PyArrow in {duration:.2f}s")
                    return result
            
            # Default pandas engine
            if approach['use_chunks']:
                chunk_size = approach.get('chunk_size', cls.DEFAULT_CHUNK_SIZE)
                logger.info(f"Reading file in chunks (size: {chunk_size:,})")
                
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=chunk_size, **kwargs):
                    # Apply column filters
                    if columns:
                        chunk = chunk[columns]
                    
                    # Apply filters if provided
                    if filters:
                        for col, val in filters.items():
                            chunk = chunk[chunk[col] == val]
                    
                    chunks.append(chunk)
                
                if chunks:
                    result = pd.concat(chunks, ignore_index=True)
                else:
                    result = pd.DataFrame()
            else:
                # Read all at once
                result = pd.read_csv(file_path, **kwargs)
                
                # Apply column filters
                if columns:
                    result = result[columns]
                
                # Apply filters if provided
                if filters:
                    for col, val in filters.items():
                        result = result[result[col] == val]
            
            duration = time.time() - start_time
            logger.info(f"Loaded file with pandas in {duration:.2f}s")
            return result
        
        elif file_ext == '.parquet':
            # For parquet files, use appropriate engine
            if DASK_AVAILABLE and cls.get_file_size(file_path) > cls.MEDIUM_FILE_THRESHOLD:
                # Use Dask for large parquet files
                result = dd.read_parquet(file_path, columns=columns)
                
                # Apply filters if provided
                if filters:
                    for col, val in filters.items():
                        result = result[result[col] == val]
                
                duration = time.time() - start_time
                logger.info(f"Loaded parquet file with Dask in {duration:.2f}s")
                return result
            elif PYARROW_AVAILABLE:
                # Use PyArrow for regular parquet files
                table = pq.read_table(file_path, columns=columns)
                result = table.to_pandas()
                
                # Apply filters if provided
                if filters:
                    for col, val in filters.items():
                        result = result[result[col] == val]
                
                duration = time.time() - start_time
                logger.info(f"Loaded parquet file with PyArrow in {duration:.2f}s")
                return result
            else:
                # Fallback to pandas
                result = pd.read_parquet(file_path, columns=columns)
                
                # Apply filters if provided
                if filters:
                    for col, val in filters.items():
                        result = result[result[col] == val]
                
                duration = time.time() - start_time
                logger.info(f"Loaded parquet file with pandas in {duration:.2f}s")
                return result
        
        else:
            # For other file types, just use pandas
            if file_ext == '.xlsx' or file_ext == '.xls':
                result = pd.read_excel(file_path, **kwargs)
            elif file_ext == '.json':
                result = pd.read_json(file_path, **kwargs)
            elif file_ext == '.hdf' or file_ext == '.h5':
                result = pd.read_hdf(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Apply column filters
            if columns:
                result = result[columns]
            
            # Apply filters if provided
            if filters:
                for col, val in filters.items():
                    result = result[result[col] == val]
            
            duration = time.time() - start_time
            logger.info(f"Loaded file with pandas in {duration:.2f}s")
            return result
    
    @classmethod
    def filter_optimized(
        cls,
        df: Union[pd.DataFrame, Any],
        filters: Dict[str, Any]
    ) -> Union[pd.DataFrame, Any]:
        """
        Apply filters to a dataframe in an optimized way based on its type.
        
        Args:
            df: Dataframe to filter (pandas, Dask, etc.)
            filters: Dictionary of column:value filters
            
        Returns:
            Filtered dataframe
        """
        if DASK_AVAILABLE and isinstance(df, dd.DataFrame):
            # Dask filtering
            result = df
            for col, val in filters.items():
                result = result[result[col] == val]
            return result
        else:
            # Pandas filtering
            result = df
            for col, val in filters.items():
                result = result[result[col] == val]
            return result
    
    @classmethod
    def write_optimized(
        cls,
        df: Union[pd.DataFrame, Any],
        output_path: Union[str, Path],
        format: str = 'parquet',
        **kwargs
    ) -> str:
        """
        Write data to disk in an optimized way based on data size and format.
        
        Args:
            df: Data to write (pandas DataFrame, Dask DataFrame, etc.)
            output_path: Path to write to
            format: Output format ('parquet', 'csv', etc.)
            **kwargs: Additional arguments for writing
            
        Returns:
            str: Path where data was written
        """
        start_time = time.time()
        output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle different data types
        if DASK_AVAILABLE and isinstance(df, dd.DataFrame):
            if format == 'parquet':
                df.to_parquet(output_path, **kwargs)
            elif format == 'csv':
                df.to_csv(output_path, **kwargs)
            else:
                # Compute and then use pandas
                df = df.compute()
                if format == 'excel':
                    df.to_excel(output_path, **kwargs)
                elif format == 'json':
                    df.to_json(output_path, **kwargs)
                else:
                    raise ValueError(f"Unsupported format for Dask: {format}")
        else:
            # Regular pandas DataFrame
            if format == 'parquet' and PYARROW_AVAILABLE:
                # Use PyArrow for parquet if available
                table = pa.Table.from_pandas(df)
                pq.write_table(table, output_path, **kwargs)
            elif format == 'parquet':
                # Fallback to pandas parquet
                df.to_parquet(output_path, **kwargs)
            elif format == 'csv':
                # For large DataFrames, write in chunks
                if len(df) > cls.MEDIUM_ROW_THRESHOLD:
                    chunk_size = cls.DEFAULT_CHUNK_SIZE
                    logger.info(f"Writing large DataFrame in chunks (size: {chunk_size:,})")
                    
                    # Write first chunk with header
                    df.iloc[:chunk_size].to_csv(
                        output_path, 
                        index=kwargs.get('index', False),
                        mode='w'
                    )
                    
                    # Write remaining chunks without header
                    for i in range(chunk_size, len(df), chunk_size):
                        end_idx = min(i + chunk_size, len(df))
                        df.iloc[i:end_idx].to_csv(
                            output_path,
                            header=False,
                            index=kwargs.get('index', False),
                            mode='a'
                        )
                else:
                    # Write all at once
                    df.to_csv(output_path, index=kwargs.get('index', False))
            elif format == 'excel':
                df.to_excel(output_path, **kwargs)
            elif format == 'json':
                df.to_json(output_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        duration = time.time() - start_time
        logger.info(f"Wrote data to {output_path} in {duration:.2f}s")
        return str(output_path)