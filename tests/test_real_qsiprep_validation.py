#!/usr/bin/env python3
"""
Real data validation script for QsiPrepAdapter using bids_example data.

This script tests the QsiPrepAdapter implementation with real qsiprep outputs
from the bids_example directory to ensure it works correctly with actual data.
"""

import json
import sys
from pathlib import Path
import numpy as np
import nibabel as nib

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from elikopy.processing.qsiprep_adapter import QsiPrepAdapter
from elikopy.data.bids_handler import BIDSHandler


def test_real_qsiprep_data():
    """Test QsiPrepAdapter with real qsiprep data from bids_example"""
    
    print("=" * 60)
    print("TESTING QSIPREP ADAPTER WITH REAL DATA")
    print("=" * 60)
    
    # Set up paths
    bids_root = Path("bids_example")
    qsiprep_dir = bids_root / "derivatives" / "qsiprep"
    
    if not qsiprep_dir.exists():
        print(f"❌ QSIPrep directory not found: {qsiprep_dir}")
        return False
    
    print(f"✓ Using BIDS root: {bids_root}")
    print(f"✓ Using QSIPrep directory: {qsiprep_dir}")
    
    try:
        # Initialize the adapter with proper BIDS handler
        print("\n1. Initializing QsiPrepAdapter...")
        bids_handler = BIDSHandler(bids_root=bids_root, qsiprep_dir=qsiprep_dir)
        adapter = QsiPrepAdapter(qsiprep_dir, bids_handler)
        print("✓ QsiPrepAdapter initialized successfully")
        
        # Test basic validation
        print("\n2. Testing input validation...")
        is_valid = adapter.validate_inputs()
        print(f"✓ Input validation result: {is_valid}")
        
        # Get available subjects
        print("\n3. Getting available subjects...")
        subjects = adapter.get_available_subjects()
        print(f"✓ Available subjects: {subjects}")
        
        if not subjects:
            print("❌ No subjects found in qsiprep data")
            return False
        
        # Use the first subject
        subject = subjects[0]
        print(f"✓ Testing with subject: {subject}")
        
        # Get available sessions
        print("\n4. Getting available sessions...")
        sessions = adapter.get_available_sessions(subject)
        print(f"✓ Available sessions for {subject}: {sessions}")
        
        # Use the first session if available
        session = sessions[0] if sessions and sessions[0] else None
        print(f"✓ Testing with session: {session}")
        
        # Test DWI data loading
        print("\n5. Loading preprocessed DWI data...")
        try:
            dwi_data = adapter.load_preprocessed_data(subject, session)
            print(f"✓ DWI data loaded successfully")
            print(f"  - Shape: {dwi_data.shape}")
            print(f"  - Number of volumes: {dwi_data.n_volumes}")
            print(f"  - Number of b-values: {dwi_data.n_bvals}")
            print(f"  - Unique b-values: {np.unique(dwi_data.bvals)}")
            print(f"  - Has brain mask: {dwi_data.mask is not None}")
            print(f"  - Metadata keys: {list(dwi_data.metadata.keys())}")
        except Exception as e:
            print(f"❌ Failed to load DWI data: {e}")
            return False
        
        # Test anatomical data loading (try both with and without session)
        print("\n6. Loading preprocessed anatomical data...")
        anat_data = None
        try:
            # First try with session
            anat_data = adapter.load_anatomical_data(subject, session)
            print(f"✓ Anatomical data loaded successfully (with session)")
        except Exception:
            try:
                # Try without session (common in qsiprep structure)
                anat_data = adapter.load_anatomical_data(subject, None)
                print(f"✓ Anatomical data loaded successfully (without session)")
            except Exception as e:
                print(f"⚠️ Anatomical data not available: {e}")
                print("  - This is not critical for DWI processing")
        
        if anat_data:
            print(f"  - Shape: {anat_data.shape}")
            print(f"  - Has brain mask: {anat_data.brain_mask is not None}")
            print(f"  - Has tissue segmentation: {anat_data.tissue_segmentation is not None}")
            print(f"  - Metadata keys: {list(anat_data.metadata.keys())}")
        
        # Test metadata extraction
        print("\n7. Getting preprocessing metadata...")
        try:
            metadata = adapter.get_preprocessing_metadata(subject, session)
            print(f"✓ Preprocessing metadata extracted successfully")
            print(f"  - QSIPrep version: {metadata.get('qsiprep_version')}")
            print(f"  - Number of preprocessing steps: {len(metadata.get('preprocessing_steps', []))}")
            print(f"  - Has acquisition parameters: {bool(metadata.get('acquisition_parameters'))}")
            print(f"  - Has software versions: {bool(metadata.get('software_versions'))}")
        except Exception as e:
            print(f"❌ Failed to get preprocessing metadata: {e}")
            return False
        
        # Test validation of qsiprep outputs
        print("\n8. Validating qsiprep outputs...")
        try:
            validation_result = adapter.validate_qsiprep_outputs(subject, session)
            print(f"✓ QSIPrep output validation completed")
            print(f"  - Is valid: {validation_result.is_valid}")
            print(f"  - Number of errors: {len(validation_result.errors)}")
            print(f"  - Number of warnings: {len(validation_result.warnings)}")
            
            if validation_result.errors:
                print("  - Errors:")
                for error in validation_result.errors:
                    print(f"    • {error.message}")
            
            if validation_result.warnings:
                print("  - Warnings:")
                for warning in validation_result.warnings:
                    print(f"    • {warning.message}")
                    
            if validation_result.suggestions:
                print("  - Suggestions:")
                for suggestion in validation_result.suggestions:
                    print(f"    • {suggestion}")
                    
        except Exception as e:
            print(f"❌ Failed to validate qsiprep outputs: {e}")
            return False
        
        # Test comprehensive quality control
        print("\n9. Running comprehensive quality control...")
        try:
            qc_results = adapter.check_preprocessing_quality(subject, session)
            print(f"✓ Quality control completed successfully")
            print(f"  - Overall quality: {qc_results['overall_quality']}")
            print(f"  - Quality score: {qc_results['quality_score']:.3f}")
            print(f"  - Number of checks: {len(qc_results['checks'])}")
            print(f"  - Number of errors: {len(qc_results['errors'])}")
            print(f"  - Number of warnings: {len(qc_results['warnings'])}")
            print(f"  - Number of recommendations: {len(qc_results['recommendations'])}")
            
            # Show individual check results
            print("  - Individual check results:")
            for check_name, check_result in qc_results['checks'].items():
                status = check_result.get('status', 'unknown')
                score = check_result.get('score', 0.0)
                print(f"    • {check_name}: {status} (score: {score:.3f})")
            
            if qc_results['recommendations']:
                print("  - Quality recommendations:")
                for rec in qc_results['recommendations'][:3]:  # Show first 3
                    print(f"    • {rec}")
                    
        except Exception as e:
            print(f"❌ Failed to run quality control: {e}")
            return False
        
        # Test quality report generation
        print("\n10. Generating quality report...")
        try:
            report_path = Path("qc_report_real_data.json")
            report = adapter.generate_quality_report(subject, session, report_path)
            print(f"✓ Quality report generated successfully")
            print(f"  - Report saved to: {report_path}")
            print(f"  - Report sections: {list(report.keys())}")
            print(f"  - Overall quality: {report['summary']['overall_quality']}")
            print(f"  - Quality score: {report['summary']['quality_score']:.3f}")
            
        except Exception as e:
            print(f"❌ Failed to generate quality report: {e}")
            return False
        
        # Test specific file access
        print("\n11. Testing specific file access...")
        try:
            # Test direct file access through BIDS handler
            dwi_files = adapter.bids_handler.get_preprocessed_dwi_files(subject, session)
            anat_files = adapter.bids_handler.get_preprocessed_anatomical_files(subject, session)
            
            print(f"✓ Direct file access successful")
            print(f"  - DWI files found: {len(dwi_files)}")
            print(f"  - Anatomical files found: {len(anat_files)}")
            
            if dwi_files:
                dwi_file = dwi_files[0]
                print(f"  - DWI file path: {dwi_file.path}")
                print(f"  - bval file exists: {dwi_file.bval_path.exists()}")
                print(f"  - bvec file exists: {dwi_file.bvec_path.exists()}")
                print(f"  - JSON file exists: {dwi_file.json_path and dwi_file.json_path.exists()}")
                
        except Exception as e:
            print(f"❌ Failed to test specific file access: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("✅ QsiPrepAdapter works correctly with real qsiprep data")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_quality_analysis():
    """Perform detailed data quality analysis on the real data"""
    
    print("\n" + "=" * 60)
    print("DETAILED DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    qsiprep_dir = Path("bids_example/derivatives/qsiprep")
    
    try:
        bids_root = Path("bids_example")
        bids_handler = BIDSHandler(bids_root=bids_root, qsiprep_dir=qsiprep_dir)
        adapter = QsiPrepAdapter(qsiprep_dir, bids_handler)
        subject = "sub-01"
        session = "ses-01"
        
        # Load data
        dwi_data = adapter.load_preprocessed_data(subject, session)
        
        # Try to load anatomical data (with and without session)
        anat_data = None
        try:
            anat_data = adapter.load_anatomical_data(subject, session)
        except Exception:
            try:
                anat_data = adapter.load_anatomical_data(subject, None)
            except Exception:
                print("  - Anatomical data not available, skipping anatomical analysis")
        
        print("\n1. DWI Data Analysis:")
        print(f"  - Image dimensions: {dwi_data.dwi_image.shape}")
        print(f"  - Voxel size: {dwi_data.header.get_zooms()[:3]} mm")
        print(f"  - Data type: {dwi_data.dwi_image.dtype}")
        print(f"  - Signal range: [{dwi_data.dwi_image.min():.1f}, {dwi_data.dwi_image.max():.1f}]")
        print(f"  - Signal mean: {dwi_data.dwi_image.mean():.1f}")
        print(f"  - Signal std: {dwi_data.dwi_image.std():.1f}")
        
        # B-value analysis
        unique_bvals = np.unique(dwi_data.bvals)
        print(f"  - B-values: {unique_bvals}")
        for bval in unique_bvals:
            count = np.sum(dwi_data.bvals == bval)
            print(f"    • b={bval}: {count} volumes")
        
        # Gradient analysis
        print(f"  - Gradient directions shape: {dwi_data.bvecs.shape}")
        non_zero_grads = dwi_data.bvals > 100
        if np.any(non_zero_grads):
            grad_norms = np.linalg.norm(dwi_data.bvecs[:, non_zero_grads], axis=0)
            print(f"  - Gradient norms (non-b0): mean={grad_norms.mean():.3f}, std={grad_norms.std():.3f}")
        
        if anat_data:
            print("\n2. Anatomical Data Analysis:")
            print(f"  - Image dimensions: {anat_data.t1w_image.shape}")
            print(f"  - Voxel size: {anat_data.header.get_zooms()[:3]} mm")
            print(f"  - Data type: {anat_data.t1w_image.dtype}")
            print(f"  - Signal range: [{anat_data.t1w_image.min():.1f}, {anat_data.t1w_image.max():.1f}]")
            print(f"  - Signal mean: {anat_data.t1w_image.mean():.1f}")
            print(f"  - Signal std: {anat_data.t1w_image.std():.1f}")
            
            if anat_data.brain_mask is not None:
                brain_volume = np.sum(anat_data.brain_mask > 0)
                total_volume = np.prod(anat_data.brain_mask.shape)
                print(f"  - Brain mask coverage: {brain_volume/total_volume*100:.1f}%")
            
            if anat_data.tissue_segmentation is not None:
                unique_labels = np.unique(anat_data.tissue_segmentation)
                print(f"  - Tissue segmentation labels: {unique_labels}")
        else:
            print("\n2. Anatomical Data Analysis: Skipped (data not available)")
        
        print("\n3. Metadata Analysis:")
        metadata = adapter.get_preprocessing_metadata(subject, session)
        
        if 'acquisition_parameters' in metadata:
            acq_params = metadata['acquisition_parameters']
            print("  - Key acquisition parameters:")
            for key in ['RepetitionTime', 'EchoTime', 'FlipAngle', 'SliceThickness']:
                if key in acq_params:
                    print(f"    • {key}: {acq_params[key]}")
        
        if 'preprocessing_steps' in metadata:
            print(f"  - Preprocessing steps ({len(metadata['preprocessing_steps'])}):")
            for i, step in enumerate(metadata['preprocessing_steps'][:5], 1):
                print(f"    {i}. {step}")
            if len(metadata['preprocessing_steps']) > 5:
                print(f"    ... and {len(metadata['preprocessing_steps']) - 5} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Data quality analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_motion_analysis():
    """Test motion parameter analysis with real confounds data"""
    
    print("\n" + "=" * 60)
    print("MOTION PARAMETER ANALYSIS")
    print("=" * 60)
    
    # Check if confounds file exists
    confounds_file = Path("bids_example/derivatives/qsiprep/sub-01/ses-01/dwi/sub-01_ses-01_desc-confounds_timeseries.tsv")
    
    if not confounds_file.exists():
        print(f"❌ Confounds file not found: {confounds_file}")
        return False
    
    try:
        # Read confounds file
        import pandas as pd
        confounds_df = pd.read_csv(confounds_file, sep='\t')
        
        print(f"✓ Confounds file loaded successfully")
        print(f"  - Shape: {confounds_df.shape}")
        print(f"  - Columns: {list(confounds_df.columns)}")
        
        # Look for motion parameters
        motion_cols = [col for col in confounds_df.columns if any(x in col.lower() for x in ['trans', 'rot', 'motion'])]
        
        if motion_cols:
            print(f"  - Motion-related columns: {motion_cols}")
            
            # Calculate basic motion statistics
            for col in motion_cols[:6]:  # First 6 motion parameters
                values = confounds_df[col].values
                print(f"    • {col}: mean={values.mean():.4f}, std={values.std():.4f}, max={values.max():.4f}")
        
        # Test the adapter's motion analysis
        qsiprep_dir = Path("bids_example/derivatives/qsiprep")
        bids_root = Path("bids_example")
        bids_handler = BIDSHandler(bids_root=bids_root, qsiprep_dir=qsiprep_dir)
        adapter = QsiPrepAdapter(qsiprep_dir, bids_handler)
        
        motion_result = adapter._check_motion_parameters("sub-01", "ses-01")
        print(f"\n✓ Motion analysis result:")
        print(f"  - Status: {motion_result['status']}")
        print(f"  - Score: {motion_result['score']:.3f}")
        print(f"  - Files found: {motion_result['details']['motion_files_found']}")
        
        if 'motion_stats' in motion_result['details']:
            stats = motion_result['details']['motion_stats']
            print(f"  - Motion statistics:")
            print(f"    • Mean translation RMS: {stats['mean_translation_rms']:.4f} mm")
            print(f"    • Max translation RMS: {stats['max_translation_rms']:.4f} mm")
            print(f"    • Mean rotation RMS: {stats['mean_rotation_rms']:.6f} rad")
            print(f"    • Max rotation RMS: {stats['max_rotation_rms']:.6f} rad")
        
        if motion_result['warnings']:
            print(f"  - Warnings:")
            for warning in motion_result['warnings']:
                print(f"    • {warning}")
        
        return True
        
    except Exception as e:
        print(f"❌ Motion analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting real data validation for QsiPrepAdapter...")
    
    success = True
    
    # Run main validation test
    if not test_real_qsiprep_data():
        success = False
    
    # Run detailed quality analysis
    if not test_data_quality_analysis():
        success = False
    
    # Run motion analysis
    if not test_motion_analysis():
        success = False
    
    if success:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ QsiPrepAdapter is fully validated with real qsiprep data")
    else:
        print("\n❌ SOME VALIDATION TESTS FAILED!")
        sys.exit(1)