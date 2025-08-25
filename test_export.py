#!/usr/bin/env python
"""
Test script for the new run storage and export system
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'my_backend.settings')
django.setup()

from api.models import OptimizationRun, DesignPoint
from api.services.run_storage import RunStorageService
from api.services.result_export import ResultExportService

def test_timestamped_filename():
    """Test timestamped filename generation"""
    filename = ResultExportService.generate_timestamped_filename("testrun")
    print(f"Generated filename: {filename}")
    assert "testrun_" in filename
    assert "_" in filename
    print("✓ Timestamped filename generation works")

def test_run_creation():
    """Test creating a test run"""
    # Create a test run
    run = RunStorageService.create_optimization_run(
        name="Test Run",
        description="Test run for export functionality",
        algorithm="GA",
        population_size=50,
        generations=100,
        trace_name="test-trace"
    )
    print(f"Created test run: {run.run_id}")
    
    # Add some test design points
    test_points = [
        {
            'execution_time_ms': 2000.0,
            'energy_mj': 150000.0,
            'chiplets': {'GPU': 2, 'Attention': 3, 'Sparse': 4, 'Convolution': 3}
        },
        {
            'execution_time_ms': 1800.0,
            'energy_mj': 160000.0,
            'chiplets': {'GPU': 1, 'Attention': 4, 'Sparse': 5, 'Convolution': 2}
        },
        {
            'execution_time_ms': 2200.0,
            'energy_mj': 140000.0,
            'chiplets': {'GPU': 3, 'Attention': 2, 'Sparse': 3, 'Convolution': 4}
        }
    ]
    
    RunStorageService.store_design_points(run, test_points)
    RunStorageService.complete_run(run)
    
    print(f"Added {len(test_points)} design points")
    print("✓ Run creation and design point storage works")
    
    return run

def test_export_functionality(run):
    """Test exporting the run to zip"""
    try:
        zip_path = ResultExportService.export_run_to_zip(run, "test_export")
        print(f"Exported to: {zip_path}")
        
        # Check if file exists
        if os.path.exists(zip_path):
            print(f"✓ Zip file created successfully ({os.path.getsize(zip_path)} bytes)")
            
            # Get download URL
            download_url = ResultExportService.get_export_url(zip_path)
            print(f"Download URL: {download_url}")
        else:
            print("✗ Zip file not created")
            
    except Exception as e:
        print(f"✗ Export failed: {e}")

def test_migration():
    """Test migrating existing data"""
    try:
        stats = RunStorageService.migrate_existing_data()
        print(f"Migration stats: {stats}")
        print("✓ Migration functionality works")
    except Exception as e:
        print(f"✗ Migration failed: {e}")

def main():
    """Run all tests"""
    print("Testing Run Storage and Export System")
    print("=" * 40)
    
    # Test timestamped filename generation
    test_timestamped_filename()
    
    # Test run creation and storage
    run = test_run_creation()
    
    # Test export functionality
    test_export_functionality(run)
    
    # Test migration
    test_migration()
    
    print("\n" + "=" * 40)
    print("All tests completed!")
    
    # Clean up test run
    try:
        run.delete()
        print("Cleaned up test run")
    except:
        pass

if __name__ == "__main__":
    main() 