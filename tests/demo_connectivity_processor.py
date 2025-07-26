#!/usr/bin/env python3
"""
Demo script for ConnectivityProcessor functionality
"""

import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from elikopy.processing.connectivity import (
    ConnectivityProcessor, ConnectivityConfig, AtlasData, ConnectivityMatrix
)
from dipy.tracking.streamline import Streamlines


def create_demo_data():
    """Create demo data for connectivity analysis"""
    
    # Create mock streamlines
    streamlines = []
    for i in range(50):
        # Create streamlines connecting different regions
        if i < 20:
            # Streamlines in region 1-2 connection
            streamline = np.array([
                [10 + np.random.rand(), 10 + np.random.rand(), 10 + np.random.rand()],
                [15 + np.random.rand(), 15 + np.random.rand(), 15 + np.random.rand()],
                [30 + np.random.rand(), 30 + np.random.rand(), 30 + np.random.rand()],
                [35 + np.random.rand(), 35 + np.random.rand(), 35 + np.random.rand()]
            ])
        elif i < 40:
            # Streamlines in region 2-3 connection
            streamline = np.array([
                [30 + np.random.rand(), 30 + np.random.rand(), 30 + np.random.rand()],
                [35 + np.random.rand(), 35 + np.random.rand(), 35 + np.random.rand()],
                [50 + np.random.rand(), 50 + np.random.rand(), 50 + np.random.rand()],
                [55 + np.random.rand(), 55 + np.random.rand(), 55 + np.random.rand()]
            ])
        else:
            # Streamlines in region 1-3 connection
            streamline = np.array([
                [10 + np.random.rand(), 10 + np.random.rand(), 10 + np.random.rand()],
                [15 + np.random.rand(), 15 + np.random.rand(), 15 + np.random.rand()],
                [50 + np.random.rand(), 50 + np.random.rand(), 50 + np.random.rand()],
                [55 + np.random.rand(), 55 + np.random.rand(), 55 + np.random.rand()]
            ])
        streamlines.append(streamline)
    
    # Create atlas data
    atlas_image = np.zeros((64, 64, 64), dtype=np.int32)
    atlas_image[8:22, 8:22, 8:22] = 1  # Region 1
    atlas_image[28:42, 28:42, 28:42] = 2  # Region 2
    atlas_image[48:62, 48:62, 48:62] = 3  # Region 3
    
    atlas_data = AtlasData(
        atlas_image=atlas_image,
        labels=["Left Frontal", "Right Frontal", "Occipital"],
        label_indices=[1, 2, 3],
        affine=np.eye(4),
        atlas_name="demo_atlas"
    )
    
    return Streamlines(streamlines), atlas_data


def demo_basic_connectivity():
    """Demonstrate basic connectivity matrix computation"""
    print("=== Basic Connectivity Matrix Demo ===")
    
    # Create demo data
    streamlines, atlas_data = create_demo_data()
    
    # Create processor with default configuration
    config = ConnectivityConfig(
        atlas_name="demo_atlas",
        weighting="count",
        output_formats=["csv", "json"]
    )
    processor = ConnectivityProcessor(config)
    
    print(f"Created {len(streamlines)} demo streamlines")
    print(f"Atlas has {len(atlas_data.labels)} regions: {atlas_data.labels}")
    
    # Prepare atlas result (simulating registration)
    atlas_result = {
        "atlas_labels": atlas_data.atlas_image,
        "unique_labels": np.array(atlas_data.label_indices),
        "label_names": atlas_data.labels,
        "atlas_name": atlas_data.atlas_name
    }
    
    # Mock DIPY connectivity_matrix function for demo
    with patch('elikopy.processing.connectivity.utils.connectivity_matrix') as mock_conn:
        # Create a realistic connectivity matrix
        demo_matrix = np.array([
            [0, 20, 10],  # Region 1 connections
            [20, 0, 20],  # Region 2 connections  
            [10, 20, 0]   # Region 3 connections
        ])
        mock_conn.return_value = (demo_matrix, None)
        
        # Compute connectivity matrix
        connectivity_matrix = processor.compute_connectivity_matrix(
            streamlines, atlas_result, np.eye(4)
        )
        
        print(f"\nConnectivity Matrix ({connectivity_matrix.weighting} weighting):")
        print(f"Shape: {connectivity_matrix.matrix.shape}")
        print(f"Matrix:\n{connectivity_matrix.matrix}")
        print(f"Regions: {connectivity_matrix.atlas_labels}")
        print(f"Streamlines used: {connectivity_matrix.streamline_count}")


def demo_weighted_connectivity():
    """Demonstrate different weighting methods"""
    print("\n=== Weighted Connectivity Demo ===")
    
    streamlines, atlas_data = create_demo_data()
    
    # Test different weighting methods
    weighting_methods = ["count", "length", "fa"]
    
    for weighting in weighting_methods:
        print(f"\n--- {weighting.upper()} Weighting ---")
        
        config = ConnectivityConfig(
            atlas_name="demo_atlas",
            weighting=weighting,
            symmetric=True,
            normalize=False
        )
        processor = ConnectivityProcessor(config)
        
        atlas_result = {
            "atlas_labels": atlas_data.atlas_image,
            "unique_labels": np.array(atlas_data.label_indices),
            "label_names": atlas_data.labels,
            "atlas_name": atlas_data.atlas_name
        }
        
        # Mock different matrices for different weightings
        if weighting == "count":
            demo_matrix = np.array([[0, 20, 10], [20, 0, 20], [10, 20, 0]])
        elif weighting == "length":
            demo_matrix = np.array([[0, 400, 300], [400, 0, 500], [300, 500, 0]])
        else:  # fa
            demo_matrix = np.array([[0, 0.6, 0.5], [0.6, 0, 0.7], [0.5, 0.7, 0]])
        
        with patch('elikopy.processing.connectivity.utils.connectivity_matrix') as mock_conn:
            mock_conn.return_value = (demo_matrix, None)
            
            connectivity_matrix = processor.compute_connectivity_matrix(
                streamlines, atlas_result, np.eye(4)
            )
            
            print(f"Matrix:\n{connectivity_matrix.matrix}")
            print(f"Symmetric: {processor.config.symmetric}")


def demo_export_formats():
    """Demonstrate different export formats"""
    print("\n=== Export Formats Demo ===")
    
    # Create a sample connectivity matrix
    matrix = np.array([[0, 15, 8], [15, 0, 12], [8, 12, 0]])
    conn_matrix = ConnectivityMatrix(
        matrix=matrix,
        atlas_labels=["Region1", "Region2", "Region3"],
        label_indices=[1, 2, 3],
        subject_id="demo-01",
        session_id="ses-01",
        atlas_name="demo_atlas",
        weighting="count",
        streamline_count=100,
        processing_info={"symmetric": True, "normalized": False}
    )
    
    processor = ConnectivityProcessor()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test CSV export
        csv_file = temp_path / "connectivity.csv"
        processor._export_csv(conn_matrix, csv_file)
        print(f"CSV exported to: {csv_file}")
        
        # Test JSON export
        json_file = temp_path / "connectivity.json"
        processor._export_json(conn_matrix, json_file)
        print(f"JSON exported to: {json_file}")
        
        # Test NumPy export
        npy_file = temp_path / "connectivity.npy"
        processor._export_npy(conn_matrix, npy_file)
        print(f"NumPy exported to: {npy_file}")
        
        # Read back and verify
        import pandas as pd
        import json
        
        # Verify CSV
        df = pd.read_csv(csv_file, index_col=0)
        print(f"\nCSV content shape: {df.shape}")
        print(f"CSV columns: {list(df.columns)}")
        
        # Verify JSON
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        print(f"JSON keys: {list(json_data.keys())}")
        print(f"Atlas name from JSON: {json_data['atlas_name']}")
        
        # Verify NumPy
        loaded_matrix = np.load(npy_file)
        print(f"NumPy matrix shape: {loaded_matrix.shape}")
        print(f"Matrices equal: {np.array_equal(matrix, loaded_matrix)}")


def demo_configuration():
    """Demonstrate configuration options"""
    print("\n=== Configuration Demo ===")
    
    # Default configuration
    default_config = ConnectivityConfig()
    print("Default Configuration:")
    print(f"  Atlas: {default_config.atlas_name}")
    print(f"  Weighting: {default_config.weighting}")
    print(f"  Symmetric: {default_config.symmetric}")
    print(f"  Normalize: {default_config.normalize}")
    print(f"  Dilation radius: {default_config.dilation_radius}")
    print(f"  Output formats: {default_config.output_formats}")
    
    # Custom configuration
    custom_config = ConnectivityConfig(
        atlas_name="custom_atlas",
        weighting="length",
        symmetric=False,
        normalize=True,
        dilation_radius=2,
        min_streamline_length=30.0,
        max_streamline_length=150.0,
        output_formats=["csv", "mat"]
    )
    
    print("\nCustom Configuration:")
    print(f"  Atlas: {custom_config.atlas_name}")
    print(f"  Weighting: {custom_config.weighting}")
    print(f"  Symmetric: {custom_config.symmetric}")
    print(f"  Normalize: {custom_config.normalize}")
    print(f"  Dilation radius: {custom_config.dilation_radius}")
    print(f"  Streamline length range: {custom_config.min_streamline_length}-{custom_config.max_streamline_length}")
    print(f"  Output formats: {custom_config.output_formats}")
    
    # Test configuration validation
    processor = ConnectivityProcessor(custom_config)
    
    valid_config = {
        "weighting": "count",
        "output_formats": ["csv", "json"],
        "dilation_radius": 1
    }
    
    invalid_config = {
        "weighting": "invalid_method",
        "dilation_radius": -1
    }
    
    print(f"\nValid config validation: {processor.validate_config(valid_config)}")
    print(f"Invalid config validation: {processor.validate_config(invalid_config)}")


def main():
    """Run all demos"""
    print("ConnectivityProcessor Demo")
    print("=" * 50)
    
    try:
        demo_basic_connectivity()
        demo_weighted_connectivity()
        demo_export_formats()
        demo_configuration()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()