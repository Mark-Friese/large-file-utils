"""
Updated imports for batch_ui.py to support enhanced visualizations and large file handling
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import os
import tempfile
from typing import Dict, List, Optional, Set, Union, Tuple
from batch_processor import SubstationBatchProcessor
from viz_controls import create_visualization_controls, create_enhanced_demand_plot
from large_file_handler import LargeFileHandler  # Import our new large file handler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def display_batch_upload_section():
    """Display file upload section for batch processing."""
    st.subheader("Batch Processing Data Upload")
    
    col1, col2 = st.columns(2)
    
    uploaded_data = {}
    temp_files = {}
    
    with col1:
        # Add a setting for large file handling
        large_file_mode = st.checkbox(
            "Enable Large File Handling",
            value=True,
            help="Use optimized processing for large files (>200MB)"
        )
        
        # Memory settings (only shown when large file mode is enabled)
        memory_limit_mb = None
        if large_file_mode:
            memory_limit_mb = st.slider(
                "Memory Limit (MB)",
                min_value=100,
                max_value=2000,
                value=500,
                step=100,
                help="Memory limit for file processing"
            )
        
        demand_file = st.file_uploader(
            "Upload Demand Data (CSV)",
            type=['csv'],
            help="CSV file containing demand data for all substations"
        )
        if demand_file:
            try:
                # Save uploaded file to a temporary file
                temp_file = Path(tempfile.gettempdir()) / demand_file.name
                with open(temp_file, "wb") as f:
                    f.write(demand_file.getbuffer())
                
                # Handle file based on size and settings
                file_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
                st.write(f"Demand file size: {file_size_mb:.2f} MB")
                
                if large_file_mode and file_size_mb > 200:
                    # For large files, use our optimized handler
                    st.info(f"Processing large file ({file_size_mb:.2f} MB) using optimized handler")
                    
                    # Convert to parquet for faster processing
                    temp_files['demand'] = temp_file
                    memory_limit = memory_limit_mb * 1024 * 1024 if memory_limit_mb else None
                    
                    # Create a preview of the data
                    preview = pd.read_csv(temp_file, nrows=5)
                    st.write("Data preview (first 5 rows):")
                    st.dataframe(preview)
                    
                    # Mark for later processing
                    uploaded_data['demand'] = {
                        'path': str(temp_file),
                        'large_file': True,
                        'memory_limit': memory_limit
                    }
                else:
                    # For smaller files, read directly with pandas
                    uploaded_data['demand'] = pd.read_csv(temp_file)
            except Exception as e:
                st.error(f"Error loading demand data: {str(e)}")
                logger.exception("Error loading demand data")
            
        peaks_file = st.file_uploader(
            "Upload Peak Forecasts (CSV)",
            type=['csv'],
            help="CSV with base, high, and low peaks for each substation"
        )
        if peaks_file:
            try:
                # Save uploaded file to a temporary file
                temp_file = Path(tempfile.gettempdir()) / peaks_file.name
                with open(temp_file, "wb") as f:
                    f.write(peaks_file.getbuffer())
                
                # For peak files, we usually don't have size issues
                uploaded_data['peaks'] = pd.read_csv(temp_file)
                temp_files['peaks'] = temp_file
            except Exception as e:
                st.error(f"Error loading peaks data: {str(e)}")
                logger.exception("Error loading peaks data")
    
    with col2:
        firm_capacity_file = st.file_uploader(
            "Upload Firm Capacities (CSV)",
            type=['csv'],
            help="CSV with firm capacity values for each substation"
        )
        if firm_capacity_file:
            try:
                # Save uploaded file to a temporary file
                temp_file = Path(tempfile.gettempdir()) / firm_capacity_file.name
                with open(temp_file, "wb") as f:
                    f.write(firm_capacity_file.getbuffer())
                    
                # For firm capacity files, we usually don't have size issues
                uploaded_data['firm_capacity'] = pd.read_csv(temp_file)
                temp_files['firm_capacity'] = temp_file
            except Exception as e:
                st.error(f"Error loading firm capacity data: {str(e)}")
                logger.exception("Error loading firm capacity data")
        
        use_financial_year = st.checkbox(
            "Adjust to Financial Year",
            value=False,
            key='batch_use_fy'
        )
        
        financial_year = None
        if use_financial_year:
            current_year = pd.Timestamp.now().year
            financial_year = st.selectbox(
                "Select Financial Year",
                [f"{year}/{str(year+1)[2:]}" for year in range(current_year-1, current_year+3)],
                index=1
            )
    
    return {
        'data': uploaded_data if len(uploaded_data) == 3 else None,
        'financial_year': financial_year if use_financial_year else None,
        'temp_files': temp_files,
        'large_file_mode': large_file_mode,
        'memory_limit_mb': memory_limit_mb
    }

def display_substation_selector(peaks_data: pd.DataFrame) -> List[str]:
    """Display substation selection interface."""
    st.subheader("Select Substations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selection_mode = st.radio(
            "Selection Mode:",
            ["All Substations", "Select Specific Substations"]
        )
    
    selected_substations = None
    if selection_mode == "Select Specific Substations":
        with col2:
            selected_substations = st.multiselect(
                "Choose Substations:",
                options=peaks_data['substation_name'].tolist()
            )
    
    return selected_substations

def display_batch_processing_parameters():
    """Display batch processing parameters interface."""
    st.subheader("Processing Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_workers = st.slider(
            "Maximum Parallel Processes",
            min_value=1,
            max_value=8,
            value=4,
            help="Number of substations to process simultaneously"
        )
    
    with col2:
        chunk_size = st.slider(
            "Processing Chunk Size",
            min_value=10000,
            max_value=500000,
            value=100000,
            step=10000,
            help="Number of rows to process at once (for large files)"
        )
    
    with col3:
        output_dir = st.text_input(
            "Output Directory",
            value="output/batch_results",
            help="Directory where results will be saved"
        )
    
    return max_workers, chunk_size, output_dir

def display_batch_results(results: Dict[str, Dict]):
    """Display batch processing results with enhanced visualizations."""
    st.header("Batch Processing Results")
    
    # Summary metrics
    total = len(results)
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    failed = total - successful
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Network Groups", total)
    with col2:
        st.metric("Successfully Processed", successful)
    with col3:
        st.metric("Failed", failed)
    
    # Process successful results for visualization
    successful_results = {
        name: result for name, result in results.items() 
        if result['status'] == 'success'
    }
    
    if successful_results:
        # Network Group Selection for Visualization
        selected_group = st.selectbox(
            "Select Network Group to View",
            options=list(successful_results.keys())
        )
        
        if selected_group:
            group_results = successful_results[selected_group]
            
            # Create tabs for different visualizations
            viz_tab1, viz_tab2 = st.tabs(["Forecast Results", "Competition Analysis"])
            
            with viz_tab1:
                # Get forecast results and metadata
                forecast_results = group_results['forecast_results']
                metadata = group_results['forecast_metadata']
                
                # Display financial year if used
                if metadata.get('financial_year'):
                    st.info(f"Results adjusted to Financial Year: {metadata['financial_year']}")
                
                # Forecast metrics
                fcol1, fcol2, fcol3 = st.columns(3)
                with fcol1:
                    st.metric(
                        "Base Scenario Peak",
                        f"{metadata['base_scenario_peak']:.1f} MW",
                        delta=f"{(metadata['base_scenario_peak'] - metadata['original_peak']):.1f} MW"
                    )
                with fcol2:
                    st.metric(
                        "High Scenario Peak",
                        f"{metadata['high_scenario_peak']:.1f} MW"
                    )
                with fcol3:
                    st.metric(
                        "Low Scenario Peak",
                        f"{metadata['low_scenario_peak']:.1f} MW"
                    )
                
                # Default colors for visualization
                default_colors = {
                    'main_line': '#2c4b7c',
                    'secondary_line': '#ff9e16',
                    'firm_capacity': '#e41a1c',
                    'fill': '#2c4b7c'
                }
                
                # Visualization controls
                viz_settings = create_visualization_controls(
                    st, 
                    default_colors, 
                    key_prefix=f"batch_{selected_group}"
                )
                
                # Create forecast plot
                fig = create_enhanced_demand_plot(
                    forecast_results,
                    "Scenario-based",
                    metadata.get('firm_capacity', 0),
                    viz_settings
                )
                
                # Use unique key for forecast plot
                st.plotly_chart(
                    fig, 
                    use_container_width=True,
                    key=f"forecast_plot_{selected_group}"
                )
            
            with viz_tab2:
                if group_results.get('competitions'):
                    st.subheader("Competition Analysis")
                    
                    # Extract service window data
                    window_data = []
                    for comp in group_results['competitions']:
                        for period in comp['service_periods']:
                            for window in period['service_windows']:
                                window_data.append({
                                    'Month': period['name'],
                                    'Type': 'Weekend' if 'Saturday' in window['service_days'] else 'Weekday',
                                    'Time': f"{window['start']}-{window['end']}",
                                    'Required Reduction': float(window['capacity_required'])
                                })
                    
                    if window_data:
                        df_windows = pd.DataFrame(window_data)
                        
                        # Create service window visualization
                        fig = go.Figure()
                        
                        for window_type in ['Weekday', 'Weekend']:
                            df_subset = df_windows[df_windows['Type'] == window_type]
                            
                            fig.add_trace(go.Bar(
                                name=window_type,
                                x=df_subset['Month'],
                                y=df_subset['Required Reduction'],
                                text=df_subset['Time'],
                                textposition='auto',
                            ))
                        
                        fig.update_layout(
                            title="Required Reduction by Service Window",
                            xaxis_title="Month",
                            yaxis_title="Required Reduction (MW)",
                            barmode='group',
                            height=500
                        )
                        
                        # Use unique key for service window plot
                        st.plotly_chart(
                            fig, 
                            use_container_width=True,
                            key=f"service_window_plot_{selected_group}"
                        )
                        
                        # Service window summary metrics
                        wcol1, wcol2, wcol3 = st.columns(3)
                        with wcol1:
                            st.metric("Total Service Windows", len(df_windows))
                        with wcol2:
                            st.metric(
                                "Average Required Reduction",
                                f"{df_windows['Required Reduction'].mean():.2f} MW"
                            )
                        with wcol3:
                            st.metric(
                                "Total Reduction Required",
                                f"{df_windows['Required Reduction'].sum():.2f} MW"
                            )
                else:
                    st.info("No competitions generated for this network group.")
    
    # Create detailed results table
    st.subheader("Detailed Results")
    results_data = []
    for name, result in results.items():
        if result['status'] == 'success':
            metadata = result['forecast_metadata']
            results_data.append({
                'Network Group': name,
                'Status': '✅ Success',
                'Original Peak (MW)': round(metadata['original_peak'], 2),
                'Base Peak (MW)': round(metadata['base_scenario_peak'], 2),
                'High Peak (MW)': round(metadata['high_scenario_peak'], 2),
                'Competitions Generated': len(result.get('competitions', [])),
                'Financial Year': metadata.get('financial_year', 'N/A')
            })
        else:
            results_data.append({
                'Network Group': name,
                'Status': '❌ Failed',
                'Error': result['error_message']
            })
    
    results_df = pd.DataFrame(results_data)
    
    # Display results table with styling
    try:
        st.dataframe(
            results_df.style.apply(
                lambda x: ['background-color: #EFF8F6' if '✅' in str(x['Status']) 
                          else 'background-color: #FFF1F1' for _ in x],
                axis=1
            ),
            key="results_table"
        )
    except:
        # Fallback if styling fails
        st.dataframe(results_df, key="results_table_fallback")
    
    # Add download button for results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results Summary (CSV)",
        data=csv,
        file_name="batch_processing_results.csv",
        mime="text/csv",
        key="download_results"
    )

def display_batch_errors(results: Dict[str, Dict]):
    """Display any errors that occurred during batch processing."""
    errors = [(sub, res['error_message']) 
              for sub, res in results.items() 
              if res['status'] == 'error']
    
    if errors:
        st.subheader("Processing Errors")
        for substation, error in errors:
            with st.expander(f"Error in {substation}"):
                st.error(error)

def integrate_batch_processing():
    """Main function to integrate batch processing into the Streamlit app."""
    st.title("Network Flow Demand Forecasting")
    
    # Mode selection
    mode = st.radio(
        "Select Processing Mode:",
        ["Individual Substation", "Batch Processing"],
        help="Choose to process a single substation or multiple substations in batch"
    )
    
    if mode == "Batch Processing":
        st.markdown("### Batch Processing Mode")
        st.info("""
        Upload files containing data for multiple substations to process them in batch. 
        Required files:
        - Demand Data CSV (with columns: Timestamp, Network Group Name, Demand (MW))
        - Peak Forecasts CSV (with columns: Network Group Name, base_peak, high_peak, low_peak)
        - Firm Capacities CSV (with columns: Network Group Name, firm_capacity)
        
        For large files (>200MB), enable Large File Handling option.
        """)
        
        # File upload section
        upload_results = display_batch_upload_section()
        
        # Check if all required files are uploaded
        if upload_results['data'] is not None:
            try:
                # Load peaks data for network group selection
                peaks_df = upload_results['data']['peaks']
                
                # Network group selection
                selected_substations = display_substation_selector(peaks_df)
                
                # Processing parameters
                max_workers, chunk_size, output_dir = display_batch_processing_parameters()
                
                # Process button
                if st.button("Start Batch Processing"):
                    with st.spinner("Processing network groups..."):
                        try:
                            # Initialize large file handler if needed
                            large_file_mode = upload_results['large_file_mode']
                            memory_limit = (
                                upload_results['memory_limit_mb'] * 1024 * 1024 
                                if upload_results['memory_limit_mb'] else None
                            )
                            
                            # Check if demand data is a large file that needs special handling
                            demand_data = upload_results['data']['demand']
                            if isinstance(demand_data, dict) and demand_data.get('large_file'):
                                # Load using large file handler
                                st.info("Loading large demand file using optimized handler...")
                                handler = LargeFileHandler()
                                
                                # Decide if we should use Dask or pandas
                                use_dask = False  # Conservative default
                                
                                # Load the file
                                demand_data = handler.load_large_csv(
                                    demand_data['path'],
                                    use_dask=use_dask,
                                    memory_limit=memory_limit,
                                    convert_to_parquet_first=True
                                )
                                
                                # If we're using Dask, we'll need to convert peaks and firm_capacity
                                # to Dask dataframes as well for compatibility
                                if use_dask:
                                    import dask.dataframe as dd
                                    peaks_data = dd.from_pandas(
                                        upload_results['data']['peaks'], npartitions=1
                                    )
                                    firm_capacity_data = dd.from_pandas(
                                        upload_results['data']['firm_capacity'], npartitions=1
                                    )
                                else:
                                    peaks_data = upload_results['data']['peaks']
                                    firm_capacity_data = upload_results['data']['firm_capacity']
                            else:
                                # Regular file handling
                                demand_data = upload_results['data']['demand']
                                peaks_data = upload_results['data']['peaks']
                                firm_capacity_data = upload_results['data']['firm_capacity']
                            
                            # Initialize batch processor
                            processor = SubstationBatchProcessor(
                                demand_data=demand_data,
                                peaks_data=peaks_data,
                                firm_capacity_data=firm_capacity_data,
                                schema_path="data/schemas/flexibility_competition_schema.json",
                                financial_year=upload_results['financial_year']
                            )
                            
                            # Process network groups with chunking for large files
                            results = processor.process_substations(
                                network_group_list=selected_substations,
                                max_workers=max_workers,
                                chunk_size=chunk_size
                            )
                            
                            # Save results
                            processor.save_results(output_dir)
                            
                            # Display results
                            display_batch_results(results)
                            display_batch_errors(results)
                            
                            st.success(f"Results saved to {output_dir}")
                        
                        except Exception as e:
                            st.error(f"Error during batch processing: {str(e)}")
                            logger.exception("Batch processing error")
            
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                logger.exception("Data loading error")
        
        return True
    
    return False

if __name__ == "__main__":
    st.set_page_config(
        page_title="Network Flow Demand Forecasting",
        page_icon="⚡",
        layout="wide"
    )
    integrate_batch_processing()
