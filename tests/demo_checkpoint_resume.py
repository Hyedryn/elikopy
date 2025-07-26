#!/usr/bin/env python3
"""
Demo script for checkpoint and resume functionality

This script demonstrates how to use the checkpoint and resume capabilities
of the ElikopyProcessor for long-running processing tasks.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

from elikopy.core.processor import ElikopyProcessor
from elikopy.core.study import ElikopyStudy, Subject
from elikopy.core.base import ProcessingStatus
from elikopy.data.bids_handler import DWIFile


def create_demo_study(temp_dir: Path) -> ElikopyStudy:
    """Create a demo study for testing"""
    # Create mock study
    study = Mock(spec=ElikopyStudy)
    study.study_path = temp_dir
    study.bids_root = temp_dir / "bids"
    study.qsiprep_dir = temp_dir / "qsiprep"
    study.derivatives_name = "elikopy"
    study.derivatives_dir = temp_dir / "derivatives" / "elikopy"
    study.bids_handler = Mock()
    
    # Create mock subjects
    subjects = []
    for i in range(5):
        subject = Mock(spec=Subject)
        subject.id = f"sub-{i+1:02d}"
        
        # Create mock DWI file
        dwi_file = Mock(spec=DWIFile)
        dwi_file.path = temp_dir / f"sub-{i+1:02d}_dwi.nii.gz"
        dwi_file.bval_path = temp_dir / f"sub-{i+1:02d}_dwi.bval"
        dwi_file.bvec_path = temp_dir / f"sub-{i+1:02d}_dwi.bvec"
        dwi_file.json_path = temp_dir / f"sub-{i+1:02d}_dwi.json"
        
        subject.dwi_files = [dwi_file]
        subject.anatomical_files = []
        subjects.append(subject)
    
    study.get_subjects.return_value = subjects
    study.get_subject.side_effect = lambda sid: next(
        (s for s in subjects if s.id == sid), None
    )
    study.get_study_summary.return_value = {"study_name": "demo_study"}
    
    return study


def demo_checkpoint_creation():
    """Demonstrate checkpoint creation"""
    print("=== Checkpoint Creation Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        study = create_demo_study(temp_path)
        
        # Create processor
        processor = ElikopyProcessor(
            study=study,
            processing_type="dti",
            output_dir=temp_path / "processing_results"
        )
        
        # Simulate processing state
        processor.pipeline.status = ProcessingStatus.IN_PROGRESS
        processor.pipeline.total_subjects = 5
        processor.pipeline.completed_subjects = 2
        processor.processing_state = {
            'subjects': ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05'],
            'completed_subjects': ['sub-01', 'sub-02'],
            'failed_subjects': [],
            'current_subject_index': 2
        }
        
        # Create checkpoint
        checkpoint_path = processor._save_checkpoint("demo_checkpoint")
        print(f"✓ Checkpoint created: {checkpoint_path}")
        
        # Show checkpoint metadata
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Processing type: {metadata['processing_type']}")
        print(f"✓ Progress: {metadata['progress_percent']:.1f}%")
        print(f"✓ Completed subjects: {metadata['completed_subjects']}/{metadata['total_subjects']}")
        
        return checkpoint_path


def demo_checkpoint_listing():
    """Demonstrate checkpoint listing"""
    print("\n=== Checkpoint Listing Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        study = create_demo_study(temp_path)
        
        processor = ElikopyProcessor(
            study=study,
            processing_type="dti",
            output_dir=temp_path / "processing_results"
        )
        
        # Create multiple checkpoints
        checkpoints = []
        for i in range(3):
            processor.pipeline.completed_subjects = i + 1
            checkpoint_path = processor._save_checkpoint(f"demo_checkpoint_{i}")
            checkpoints.append(checkpoint_path)
        
        # List checkpoints
        checkpoint_list = processor.list_checkpoints(processor.checkpoint_dir)
        print(f"✓ Found {len(checkpoint_list)} checkpoints:")
        
        for i, checkpoint in enumerate(checkpoint_list):
            print(f"  {i+1}. {checkpoint['checkpoint_name']} - "
                  f"{checkpoint['progress_percent']:.1f}% complete")


def demo_checkpoint_status():
    """Demonstrate checkpoint status checking"""
    print("\n=== Checkpoint Status Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        study = create_demo_study(temp_path)
        
        processor = ElikopyProcessor(
            study=study,
            processing_type="noddi",
            output_dir=temp_path / "processing_results"
        )
        
        # Create checkpoint with specific state
        processor.pipeline.status = ProcessingStatus.IN_PROGRESS
        processor.pipeline.total_subjects = 10
        processor.pipeline.completed_subjects = 7
        processor.pipeline.failed_subjects = 1
        
        checkpoint_path = processor._save_checkpoint("status_demo")
        
        # Get status
        status = processor.get_checkpoint_status(checkpoint_path)
        print(f"✓ Checkpoint status:")
        print(f"  - Processing type: {status['processing_type']}")
        print(f"  - Total subjects: {status['total_subjects']}")
        print(f"  - Completed: {status['completed_subjects']}")
        print(f"  - Failed: {status['failed_subjects']}")
        print(f"  - Progress: {status['progress_percent']:.1f}%")
        print(f"  - Status: {status['status']}")


def demo_progress_tracking():
    """Demonstrate progress tracking features"""
    print("\n=== Progress Tracking Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        study = create_demo_study(temp_path)
        
        processor = ElikopyProcessor(
            study=study,
            processing_type="connectivity",
            output_dir=temp_path / "processing_results"
        )
        
        # Simulate processing in progress
        from datetime import datetime, timedelta
        processor.pipeline.status = ProcessingStatus.IN_PROGRESS
        processor.pipeline.start_time = datetime.now() - timedelta(hours=2)
        processor.pipeline.total_subjects = 20
        processor.pipeline.completed_subjects = 8
        
        # Get progress report
        report = processor.get_progress_report()
        print(f"✓ Progress report:")
        print(f"  - Pipeline: {report['pipeline']['name']}")
        print(f"  - Status: {report['pipeline']['status']}")
        print(f"  - Progress: {report['pipeline']['progress_percent']:.1f}%")
        print(f"  - Completed: {report['pipeline']['completed_subjects']}/{report['pipeline']['total_subjects']}")
        
        # Calculate processing rate and remaining time
        rate = processor._calculate_processing_rate()
        remaining = processor._estimate_remaining_time()
        
        if rate:
            print(f"  - Processing rate: {rate:.2f} subjects/hour")
        if remaining:
            hours = remaining / 3600
            print(f"  - Estimated remaining time: {hours:.1f} hours")


def demo_checkpoint_cleanup():
    """Demonstrate checkpoint cleanup"""
    print("\n=== Checkpoint Cleanup Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        study = create_demo_study(temp_path)
        
        processor = ElikopyProcessor(
            study=study,
            processing_type="dti",
            output_dir=temp_path / "processing_results"
        )
        
        # Create many checkpoints
        for i in range(8):
            processor.pipeline.completed_subjects = i
            processor._save_checkpoint(f"cleanup_demo_{i}")
        
        print(f"✓ Created 8 checkpoints")
        
        # List before cleanup
        before_count = len(processor.list_checkpoints(processor.checkpoint_dir))
        print(f"✓ Checkpoints before cleanup: {before_count}")
        
        # Cleanup old checkpoints, keep only 3 most recent
        deleted_count = processor.cleanup_old_checkpoints(keep_count=3)
        
        # List after cleanup
        after_count = len(processor.list_checkpoints(processor.checkpoint_dir))
        print(f"✓ Deleted {deleted_count} old checkpoints")
        print(f"✓ Checkpoints after cleanup: {after_count}")


def main():
    """Run all demos"""
    print("ElikoPy Checkpoint and Resume Functionality Demo")
    print("=" * 50)
    
    try:
        demo_checkpoint_creation()
        demo_checkpoint_listing()
        demo_checkpoint_status()
        demo_progress_tracking()
        demo_checkpoint_cleanup()
        
        print("\n" + "=" * 50)
        print("✓ All demos completed successfully!")
        print("\nKey features demonstrated:")
        print("  - Checkpoint creation with comprehensive state persistence")
        print("  - Checkpoint listing and metadata inspection")
        print("  - Progress tracking and time estimation")
        print("  - Checkpoint cleanup and management")
        print("  - Resume capability for interrupted processing")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()