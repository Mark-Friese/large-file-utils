"""
batch_processor.py - Updated for database integration with fiscal year support
and large file handling capabilities
"""

from utils import DemandHandler
from competition_config import ConfigMode
from scenario_forecasting import generate_scenario_forecast
from competition_builder import create_monthly_competitions
from data_access import FlexibilityDataAccess
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import logging
from typing import Dict, List, Optional, Set, Union, Any
from pathlib import Path
import json
import os
import tempfile
from sqlalchemy.orm import Session

# Check if dask is available for distributed computing
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

logger = logging.getLogger(__name__)

class SubstationBatchProcessor:
    def __init__(
        self,
        demand_data: Union[pd.DataFrame, Dict, str, Path, Any],
        peaks_data: Union[pd.DataFrame, Dict, str, Path],
        firm_capacity_data: Union[pd.DataFrame, Dict, str, Path],
        schema_path: str,
        financial_year: Optional[str] = None,
        db_session: Optional[Session] = None,
        config_mode: ConfigMode = ConfigMode.STANDARD,
        custom_fields: Optional[Set[str]] = None
    ):
        """
        Initialize with data and configuration.
        
        Args:
            demand_data: Demand data as DataFrame, path, or Dask DataFrame
            peaks_data: Peak forecasts as DataFrame, path, or Dask DataFrame
            firm_capacity_data: Firm capacity data as DataFrame, path, or Dask DataFrame
            schema_path: Path to competition schema
            financial_year: Optional fiscal year in format 'YYYY/YY' (e.g., '2024/25')
            db_session: Optional SQLAlchemy session (if using database)
            config_mode: Configuration mode for competition generation
            custom_fields: Optional set of custom fields to include
        """
        self.demand_data = demand_data
        self.peaks_data = peaks_data
        self.firm_capacity_data = firm_capacity_data
        self.schema_path = schema_path
        self.financial_year = financial_year
        self.config_mode = config_mode
        self.custom_fields = custom_fields
        self.results = {}
        
        # Set up database access if provided
        self.data_access = None
        if db_session is not None:
            self.data_access = FlexibilityDataAccess(db_session)
            
            # If fiscal year provided, validate and get calendar year
            self.forecast_year = None
            if financial_year:
                self.forecast_year = self.data_access.fiscal_to_calendar_year(financial_year)
                
        logger.info(f"Initialized batch processor for fiscal year {financial_year}")
        
        # Check if input data is using Dask
        self.using_dask = DASK_AVAILABLE and (
            isinstance(demand_data, dd.DataFrame) or
            isinstance(peaks_data, dd.DataFrame) or
            isinstance(firm_capacity_data, dd.DataFrame)
        )
        
        if self.using_dask:
            logger.info("Using Dask for distributed computing")
    
    def _get_network_group_data(
        self, 
        network_group: str,
        chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get data for a specific network group, either from database or from provided data.
        
        Args:
            network_group: Name of the network group
            chunk_size: Optional chunk size for processing large files
            
        Returns:
            Dict with demand data, peaks, and firm capacity
        """
        # If using database, get data from there
        if self.data_access is not None:
            df = self.data_access.get_demand_data(network_group)
            peaks = self.data_access.get_peak_forecasts(
                network_group, 
                forecast_year=self.forecast_year
            )
            firm_capacity = self.data_access.get_firm_capacity(network_group)
            
            return {
                'demand': df,
                'peaks': peaks,
                'firm_capacity': firm_capacity
            }
        
        # Otherwise, get data from provided dataframes
        try:
            # Handle Dask DataFrames
            if self.using_dask:
                # Filter demand data for this network group
                if isinstance(self.demand_data, dd.DataFrame):
                    demand_df = self.demand_data[
                        self.demand_data['Network Group Name'] == network_group
                    ].compute()
                else:
                    demand_df = self.demand_data[
                        self.demand_data['Network Group Name'] == network_group
                    ].copy()
                
                # Get peaks for this network group
                if isinstance(self.peaks_data, dd.DataFrame):
                    peaks_row = self.peaks_data[
                        self.peaks_data['substation_name'] == network_group
                    ].compute()
                else:
                    peaks_row = self.peaks_data[
                        self.peaks_data['substation_name'] == network_group
                    ]
                
                # Get firm capacity for this network group
                if isinstance(self.firm_capacity_data, dd.DataFrame):
                    firm_capacity_row = self.firm_capacity_data[
                        self.firm_capacity_data['Network Group Name'] == network_group
                    ].compute()
                else:
                    firm_capacity_row = self.firm_capacity_data[
                        self.firm_capacity_data['Network Group Name'] == network_group
                    ]
            else:
                # Process regular pandas DataFrames
                # For large files, this might need chunk processing
                if chunk_size and len(self.demand_data) > chunk_size:
                    # Process in chunks to avoid memory issues
                    chunks = []
                    for chunk_start in range(0, len(self.demand_data), chunk_size):
                        chunk_end = min(chunk_start + chunk_size, len(self.demand_data))
                        chunk = self.demand_data.iloc[chunk_start:chunk_end]
                        filtered_chunk = chunk[chunk['Network Group Name'] == network_group]
                        if not filtered_chunk.empty:
                            chunks.append(filtered_chunk)
                    
                    if chunks:
                        demand_df = pd.concat(chunks, ignore_index=True)
                    else:
                        raise ValueError(f"No data found for network group {network_group}")
                else:
                    # Regular filtering for smaller datasets
                    demand_df = self.demand_data[
                        self.demand_data['Network Group Name'] == network_group
                    ].copy()
                
                # Get peaks and firm capacity
                peaks_row = self.peaks_data[
                    self.peaks_data['substation_name'] == network_group
                ]
                
                firm_capacity_row = self.firm_capacity_data[
                    self.firm_capacity_data['Network Group Name'] == network_group
                ]
            
            # Check if we found data
            if demand_df.empty:
                raise ValueError(f"No demand data found for network group {network_group}")
            
            if peaks_row.empty:
                raise ValueError(f"No peak forecasts found for network group {network_group}")
                
            if firm_capacity_row.empty:
                raise ValueError(f"No firm capacity found for network group {network_group}")
            
            # Extract peaks and firm capacity values
            peaks = {
                'base_peak': peaks_row.iloc[0]['base_peak'],
                'high_peak': peaks_row.iloc[0]['high_peak'],
                'low_peak': peaks_row.iloc[0]['low_peak'],
            }
            
            firm_capacity = firm_capacity_row.iloc[0]['firm_capacity']
            
            return {
                'demand': demand_df,
                'peaks': peaks,
                'firm_capacity': firm_capacity
            }
                
        except Exception as e:
            logger.error(f"Error getting data for network group {network_group}: {str(e)}")
            raise
    
    def _process_single_substation(
        self, 
        network_group: str,
        chunk_size: Optional[int] = None
    ) -> Dict:
        """
        Process a single substation's data.
        
        Args:
            network_group: Name of the network group
            chunk_size: Optional chunk size for processing large files
            
        Returns:
            Dict containing processing results and metadata
        """
        try:
            # Get data for this network group
            data = self._get_network_group_data(network_group, chunk_size)
            
            # Generate forecast
            forecast_results, metadata = generate_scenario_forecast(
                data['demand'],
                base_peak=data['peaks']['base_peak'],
                high_peak=data['peaks']['high_peak'],
                low_peak=data['peaks']['low_peak'],
                financial_year=self.financial_year
            )
            
            # Add additional metadata if available
            if self.data_access:
                substation_metadata = self.data_access.get_metadata(network_group)
                metadata.update(substation_metadata)
            
            # Add firm capacity to metadata
            metadata['firm_capacity'] = data['firm_capacity']
            
            # Generate competitions
            competitions = create_monthly_competitions(
                data['demand'],
                forecast_results,
                data['firm_capacity'],
                schema_path=str(self.schema_path),
                config_mode=self.config_mode,
                custom_fields=self.custom_fields,
                risk_threshold=0.05
            )
            
            return {
                'status': 'success',
                'forecast_results': forecast_results,
                'forecast_metadata': metadata,
                'competitions': competitions
            }
                
        except Exception as e:
            logger.error(f"Error processing network group {network_group}: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e)
            }
    
    def process_substations(
        self,
        network_group_list: Optional[List[str]] = None,
        max_workers: int = 4,
        chunk_size: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Process multiple substations in parallel.
        
        Args:
            network_group_list: Optional list of network groups to process
            max_workers: Maximum number of parallel workers
            chunk_size: Optional chunk size for processing large files
            
        Returns:
            Dict mapping network group names to their processing results
        """
        # Get list of network groups from database if available and not provided
        if network_group_list is None:
            if self.data_access:
                network_group_list = self.data_access.get_network_groups()
            else:
                # Otherwise, get from peaks_data
                if isinstance(self.peaks_data, dd.DataFrame):
                    network_group_list = self.peaks_data['substation_name'].unique().compute().tolist()
                else:
                    network_group_list = self.peaks_data['substation_name'].unique().tolist()
        
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_group = {
                executor.submit(
                    self._process_single_substation,
                    network_group,
                    chunk_size
                ): network_group
                for network_group in network_group_list
            }
            
            for future in as_completed(future_to_group):
                network_group = future_to_group[future]
                try:
                    results[network_group] = future.result()
                except Exception as e:
                    logger.error(f"Error processing {network_group}: {str(e)}")
                    results[network_group] = {
                        'status': 'error',
                        'error_message': str(e)
                    }
        
        self.results = results
        return results
    
    def save_results(self, output_dir: str):
        """
        Save processing results to files.
        
        Args:
            output_dir: Directory to save results in
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_data = []
        for network_group, result in self.results.items():
            if result['status'] == 'success':
                metadata = result['forecast_metadata']
                summary_data.append({
                    'Network Group Name': network_group,
                    'Status': 'success',
                    'Fiscal Year': self.financial_year,
                    'Original Peak (MW)': metadata['original_peak'],
                    'Base Scenario Peak (MW)': metadata['base_scenario_peak'],
                    'High Scenario Peak (MW)': metadata['high_scenario_peak'],
                    'Competitions Generated': len(result['competitions'])
                })
            else:
                summary_data.append({
                    'Network Group Name': network_group,
                    'Status': 'error',
                    'Error': result['error_message']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / 'processing_summary.csv', index=False)
        
        # Save individual results - use chunking for large files
        for network_group, result in self.results.items():
            if result['status'] == 'success':
                group_dir = output_path / network_group.replace('/', '_')
                group_dir.mkdir(exist_ok=True)
                
                # Save forecast results - handle potentially large dataframes
                forecast_df = result['forecast_results']
                
                # Check if the dataframe is very large
                if len(forecast_df) > 1_000_000:  # 1 million rows threshold
                    # Use chunks to save large files
                    chunk_size = 500_000  # 500k rows per chunk
                    
                    # Get number of chunks needed
                    num_chunks = (len(forecast_df) + chunk_size - 1) // chunk_size
                    
                    for i in range(num_chunks):
                        start_idx = i * chunk_size
                        end_idx = min(start_idx + chunk_size, len(forecast_df))
                        chunk = forecast_df.iloc[start_idx:end_idx]
                        
                        # For first chunk, create file
                        if i == 0:
                            chunk.to_csv(group_dir / 'forecast_results.csv', index=True)
                        else:
                            # For subsequent chunks, append without header
                            chunk.to_csv(
                                group_dir / 'forecast_results.csv',
                                mode='a',
                                header=False,
                                index=True
                            )
                else:
                    # For smaller dataframes, save all at once
                    forecast_df.to_csv(group_dir / 'forecast_results.csv', index=True)
                
                # Save competitions
                if result['competitions']:
                    with open(group_dir / 'competitions.json', 'w') as f:
                        json.dump(result['competitions'], f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
