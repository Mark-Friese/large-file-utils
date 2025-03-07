"""
dependency_installer.py

Provides functionality to check and install necessary dependencies
for handling large files efficiently.
"""

import subprocess
import sys
import importlib
import logging
from typing import List, Dict, Tuple, Set, Optional

logger = logging.getLogger(__name__)

class DependencyInstaller:
    """
    Utility for installing and checking dependencies for large file handling.
    """
    
    # Core dependencies for large file handling
    LARGE_FILE_DEPENDENCIES = {
        'pyarrow': 'pyarrow>=13.0.0',    # Fast CSV and Parquet handling
        'dask': 'dask[dataframe]>=2023.4.0',  # Parallel computing
        'fastparquet': 'fastparquet>=2023.4.0',  # Better Parquet support
    }
    
    # Optional enhancements
    OPTIONAL_DEPENDENCIES = {
        'modin': 'modin[dask]>=0.20.0',  # Drop-in replacement for pandas
        'snappy': 'python-snappy>=0.6.1',  # Compression for Parquet
        'zstandard': 'zstandard>=0.21.0',  # Better compression
        'lz4': 'lz4>=4.3.2',  # Fast compression
    }
    
    @classmethod
    def check_dependency(cls, package_name: str) -> bool:
        """
        Check if a Python package is installed.
        
        Args:
            package_name: Name of the package to check
            
        Returns:
            bool: True if installed, False otherwise
        """
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False
    
    @classmethod
    def get_missing_dependencies(cls, include_optional: bool = False) -> List[str]:
        """
        Get a list of missing dependencies.
        
        Args:
            include_optional: Whether to include optional dependencies
            
        Returns:
            List[str]: Names of missing dependencies
        """
        missing = []
        
        # Check core dependencies
        for package in cls.LARGE_FILE_DEPENDENCIES:
            if not cls.check_dependency(package):
                missing.append(package)
        
        # Check optional dependencies if requested
        if include_optional:
            for package in cls.OPTIONAL_DEPENDENCIES:
                if not cls.check_dependency(package):
                    missing.append(package)
        
        return missing
    
    @classmethod
    def install_dependency(cls, package_spec: str) -> bool:
        """
        Install a Python package using pip.
        
        Args:
            package_spec: Package specification (e.g., 'pyarrow>=13.0.0')
            
        Returns:
            bool: True if installation succeeded, False otherwise
        """
        try:
            logger.info(f"Installing {package_spec}...")
            
            # Run pip install
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_spec],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package_spec}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error installing {package_spec}: {e}")
            return False
    
    @classmethod
    def install_missing_dependencies(
        cls, 
        include_optional: bool = False,
        upgrade: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        Install all missing dependencies.
        
        Args:
            include_optional: Whether to include optional dependencies
            upgrade: Whether to upgrade existing dependencies
            
        Returns:
            Tuple[List[str], List[str]]: (successfully installed, failed)
        """
        missing = cls.get_missing_dependencies(include_optional)
        
        if not missing:
            logger.info("All required dependencies are already installed.")
            return [], []
        
        logger.info(f"Installing {len(missing)} missing dependencies...")
        
        successful = []
        failed = []
        
        # Install core dependencies
        for package in missing:
            if package in cls.LARGE_FILE_DEPENDENCIES:
                spec = cls.LARGE_FILE_DEPENDENCIES[package]
                if upgrade:
                    spec = f"{spec} --upgrade"
                
                if cls.install_dependency(spec):
                    successful.append(package)
                else:
                    failed.append(package)
        
        # Install optional dependencies if requested
        if include_optional:
            for package in missing:
                if package in cls.OPTIONAL_DEPENDENCIES:
                    spec = cls.OPTIONAL_DEPENDENCIES[package]
                    if upgrade:
                        spec = f"{spec} --upgrade"
                    
                    if cls.install_dependency(spec):
                        successful.append(package)
                    else:
                        failed.append(package)
        
        return successful, failed
    
    @classmethod
    def get_dependency_status(cls) -> Dict[str, bool]:
        """
        Get the installation status of all dependencies.
        
        Returns:
            Dict[str, bool]: Dictionary mapping package names to installation status
        """
        status = {}
        
        # Check core dependencies
        for package in cls.LARGE_FILE_DEPENDENCIES:
            status[package] = cls.check_dependency(package)
        
        # Check optional dependencies
        for package in cls.OPTIONAL_DEPENDENCIES:
            status[package] = cls.check_dependency(package)
        
        return status
    
    @classmethod
    def print_status_report(cls) -> None:
        """Print a report of installed and missing dependencies."""
        status = cls.get_dependency_status()
        
        print("\n=== Large File Handling Dependencies ===")
        print("\nCore Dependencies:")
        for package, installed in [(p, s) for p, s in status.items() 
                                 if p in cls.LARGE_FILE_DEPENDENCIES]:
            print(f"  {package}: {'✓ Installed' if installed else '✗ Missing'}")
        
        print("\nOptional Dependencies:")
        for package, installed in [(p, s) for p, s in status.items() 
                                 if p in cls.OPTIONAL_DEPENDENCIES]:
            print(f"  {package}: {'✓ Installed' if installed else '○ Not installed'}")
        
        # Recommendations
        print("\nRecommendations:")
        missing_core = [p for p, s in status.items() 
                      if p in cls.LARGE_FILE_DEPENDENCIES and not s]
        
        if missing_core:
            print(f"  Install missing core dependencies: {', '.join(missing_core)}")
            print(f"  Run: pip install {' '.join([cls.LARGE_FILE_DEPENDENCIES[p] for p in missing_core])}")
        else:
            print("  All core dependencies are installed.")
            
            # If all core are installed, recommend optionals
            missing_opt = [p for p, s in status.items() 
                        if p in cls.OPTIONAL_DEPENDENCIES and not s]
            if missing_opt:
                print(f"  For better performance, consider installing: {', '.join(missing_opt)}")
                print(f"  Run: pip install {' '.join([cls.OPTIONAL_DEPENDENCIES[p] for p in missing_opt])}")


if __name__ == "__main__":
    # If run directly, show status report and install dependencies
    logging.basicConfig(level=logging.INFO)
    
    print("\nChecking dependencies for large file handling...")
    DependencyInstaller.print_status_report()
    
    # Ask to install missing dependencies
    missing = DependencyInstaller.get_missing_dependencies(include_optional=False)
    if missing:
        print(f"\nFound {len(missing)} missing core dependencies: {', '.join(missing)}")
        choice = input("Do you want to install them now? (y/n): ").strip().lower()
        
        if choice == 'y':
            successful, failed = DependencyInstaller.install_missing_dependencies()
            
            if successful:
                print(f"Successfully installed: {', '.join(successful)}")
            
            if failed:
                print(f"Failed to install: {', '.join(failed)}")
        
        print("\nUpdated dependency status:")
        DependencyInstaller.print_status_report()
    
    # Ask about optional dependencies
    missing_opt = [p for p in DependencyInstaller.OPTIONAL_DEPENDENCIES
                  if not DependencyInstaller.check_dependency(p)]
    
    if missing_opt:
        print(f"\nFound {len(missing_opt)} missing optional dependencies: {', '.join(missing_opt)}")
        choice = input("Do you want to install optional dependencies for better performance? (y/n): ").strip().lower()
        
        if choice == 'y':
            successful, failed = DependencyInstaller.install_missing_dependencies(include_optional=True)
            
            if successful:
                print(f"Successfully installed: {', '.join(successful)}")
            
            if failed:
                print(f"Failed to install: {', '.join(failed)}")
            
            print("\nFinal dependency status:")
            DependencyInstaller.print_status_report()
