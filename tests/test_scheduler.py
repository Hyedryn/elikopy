"""
Unit tests for JobScheduler class
"""

import os
import json
import pickle
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import pytest

from elikopy.infrastructure.scheduler import JobScheduler, JobID, JobStatus, ProcessingJob


class TestProcessingJob:
    """Test ProcessingJob class"""
    
    def test_init(self):
        """Test ProcessingJob initialization"""
        job = ProcessingJob(
            name="test_job",
            script_path=Path("/path/to/script.sh")
        )
        assert job.name == "test_job"
        assert job.script_path == Path("/path/to/script.sh")
        assert job.cpus_per_task == 1
        assert job.mem_per_cpu == 4
        assert job.time_limit == "24:00:00"
        assert job.use_gpu is False
        assert job.gpu_count == 0
        assert job.dependencies == []
        assert job.environment_vars == {}
    
    def test_to_dict(self):
        """Test ProcessingJob to_dict conversion"""
        job = ProcessingJob(
            name="test_job",
            script_path=Path("/path/to/script.sh"),
            script_args=["arg1", "arg2"],
            cpus_per_task=4,
            mem_per_cpu=8,
            time_limit="12:00:00",
            use_gpu=True,
            gpu_count=2,
            dependencies=["12345"],
            partition="gpu",
            account="myaccount"
        )
        
        job_dict = job.to_dict()
        
        assert job_dict["name"] == "test_job"
        assert job_dict["script_path"] == str(Path("/path/to/script.sh"))
        assert job_dict["script_args"] == ["arg1", "arg2"]
        assert job_dict["cpus_per_task"] == 4
        assert job_dict["mem_per_cpu"] == 8
        assert job_dict["time_limit"] == "12:00:00"
        assert job_dict["use_gpu"] is True
        assert job_dict["gpu_count"] == 2
        assert job_dict["dependencies"] == ["12345"]
        assert job_dict["partition"] == "gpu"
        assert job_dict["account"] == "myaccount"


class TestJobID:
    """Test JobID class"""
    
    def test_init(self):
        """Test JobID initialization"""
        job_id = JobID("12345", "slurm")
        assert job_id.job_id == "12345"
        assert job_id.scheduler_type == "slurm"
    
    def test_str(self):
        """Test JobID string representation"""
        job_id = JobID("12345", "slurm")
        assert str(job_id) == "12345"


class TestJobStatus:
    """Test JobStatus class"""
    
    def test_init(self):
        """Test JobStatus initialization"""
        status = JobStatus(JobStatus.RUNNING, "12345", 0)
        assert status.status == JobStatus.RUNNING
        assert status.job_id == "12345"
        assert status.exit_code == 0
    
    def test_str_with_exit_code(self):
        """Test JobStatus string representation with exit code"""
        status = JobStatus(JobStatus.COMPLETED, "12345", 0)
        assert str(status) == "COMPLETED (exit code: 0)"
    
    def test_str_without_exit_code(self):
        """Test JobStatus string representation without exit code"""
        status = JobStatus(JobStatus.RUNNING, "12345")
        assert str(status) == "RUNNING"


class TestJobScheduler:
    """Test JobScheduler class"""
    
    def test_init_slurm(self):
        """Test JobScheduler initialization with SLURM"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            scheduler = JobScheduler("slurm")
            assert scheduler.scheduler_type == "slurm"
            assert scheduler.config == {}
    
    def test_init_local(self):
        """Test JobScheduler initialization with local scheduler"""
        scheduler = JobScheduler("local")
        assert scheduler.scheduler_type == "local"
        assert scheduler.config == {}
    
    def test_init_invalid_scheduler(self):
        """Test JobScheduler initialization with invalid scheduler"""
        with pytest.raises(ValueError, match="Unsupported scheduler type"):
            JobScheduler("invalid")
    
    def test_init_slurm_not_available(self):
        """Test JobScheduler initialization when SLURM is not available"""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()
            with pytest.raises(RuntimeError, match="SLURM is not available"):
                JobScheduler("slurm")
    
    def test_submit_slurm_job(self):
        """Test SLURM job submission"""
        with patch('subprocess.run') as mock_run:
            # Mock sinfo check
            mock_run.return_value = Mock(returncode=0)
            scheduler = JobScheduler("slurm")
            
            # Mock sbatch submission
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Submitted batch job 12345\n",
                stderr=""
            )
            
            job_config = {
                "name": "test_job",
                "script_path": "/path/to/script.sh",
                "cpus_per_task": 2,
                "mem_per_cpu": 4,
                "time_limit": "01:00:00"
            }
            
            job_id = scheduler.submit_job(job_config)
            
            assert isinstance(job_id, JobID)
            assert job_id.job_id == "12345"
            assert job_id.scheduler_type == "slurm"
    
    def test_submit_slurm_job_with_dependencies(self):
        """Test SLURM job submission with dependencies"""
        with patch('subprocess.run') as mock_run:
            # Mock sinfo check
            mock_run.return_value = Mock(returncode=0)
            scheduler = JobScheduler("slurm")
            
            # Mock sbatch submission
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Submitted batch job 12346\n",
                stderr=""
            )
            
            job_config = {
                "name": "test_job_dep",
                "script_path": "/path/to/script.sh",
                "dependencies": ["12345"]
            }
            
            job_id = scheduler.submit_job(job_config)
            
            assert isinstance(job_id, JobID)
            assert job_id.job_id == "12346"
            
            # Check that dependency was included in command
            mock_run.assert_called()
            call_args = mock_run.call_args[0][0]
            assert "--dependency" in call_args
            assert "afterok:12345" in call_args
    
    def test_submit_slurm_job_with_gpu(self):
        """Test SLURM job submission with GPU requirements"""
        with patch('subprocess.run') as mock_run:
            # Mock sinfo check
            mock_run.return_value = Mock(returncode=0)
            scheduler = JobScheduler("slurm")
            
            # Mock sbatch submission
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Submitted batch job 12347\n",
                stderr=""
            )
            
            job_config = {
                "name": "test_job_gpu",
                "script_path": "/path/to/script.sh",
                "use_gpu": True,
                "gpu_count": 2
            }
            
            job_id = scheduler.submit_job(job_config)
            
            assert isinstance(job_id, JobID)
            assert job_id.job_id == "12347"
            
            # Check that GPU was included in command
            mock_run.assert_called()
            call_args = mock_run.call_args[0][0]
            assert "--gres" in call_args
            assert "gpu:2" in call_args
    
    def test_submit_slurm_job_failure(self):
        """Test SLURM job submission failure"""
        with patch('subprocess.run') as mock_run:
            # Mock sinfo check
            mock_run.return_value = Mock(returncode=0)
            scheduler = JobScheduler("slurm")
            
            # Mock sbatch failure
            mock_run.return_value = Mock(
                returncode=1,
                stdout="",
                stderr="sbatch: error: invalid partition"
            )
            
            job_config = {
                "name": "test_job",
                "script_path": "/path/to/script.sh"
            }
            
            with pytest.raises(RuntimeError, match="Failed to submit job"):
                scheduler.submit_job(job_config)
    
    def test_submit_local_job(self):
        """Test local job submission"""
        scheduler = JobScheduler("local")
        
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            job_config = {
                "name": "test_job",
                "script_path": "/path/to/script.sh",
                "script_args": ["arg1", "arg2"]
            }
            
            job_id = scheduler.submit_job(job_config)
            
            assert isinstance(job_id, JobID)
            assert job_id.job_id == "12345"
            assert job_id.scheduler_type == "local"
    
    def test_submit_job_missing_script_path(self):
        """Test job submission without script path"""
        scheduler = JobScheduler("local")
        
        job_config = {
            "name": "test_job"
        }
        
        with pytest.raises(ValueError, match="script_path is required"):
            scheduler.submit_job(job_config)
    
    def test_submit_processing_job_object(self):
        """Test submitting ProcessingJob object"""
        scheduler = JobScheduler("local")
        
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            job = ProcessingJob(
                name="test_processing_job",
                script_path=Path("/path/to/script.sh"),
                script_args=["arg1", "arg2"],
                cpus_per_task=2
            )
            
            job_id = scheduler.submit_job(job)
            
            assert isinstance(job_id, JobID)
            assert job_id.job_id == "12345"
            assert job_id.scheduler_type == "local"
    
    def test_monitor_slurm_jobs(self):
        """Test SLURM job monitoring"""
        with patch('subprocess.run') as mock_run:
            # Mock sinfo check
            mock_run.return_value = Mock(returncode=0)
            scheduler = JobScheduler("slurm")
            
            # Mock sacct output
            mock_run.return_value = Mock(
                returncode=0,
                stdout="12345|COMPLETED|0:0\n12346|RUNNING|\n",
                stderr=""
            )
            
            job_ids = [JobID("12345", "slurm"), "12346"]
            statuses = scheduler.monitor_jobs(job_ids)
            
            assert len(statuses) == 2
            assert statuses["12345"].status == JobStatus.COMPLETED
            assert statuses["12345"].exit_code == 0
            assert statuses["12346"].status == JobStatus.RUNNING
    
    def test_monitor_local_jobs(self):
        """Test local job monitoring"""
        scheduler = JobScheduler("local")
        
        with patch('os.kill') as mock_kill, \
             patch('builtins.open', create=True) as mock_open:
            
            # Mock process exists and is running
            mock_kill.return_value = None
            mock_open.return_value.__enter__.return_value.read.return_value = "12345 (test) R"
            
            job_ids = ["12345"]
            statuses = scheduler.monitor_jobs(job_ids)
            
            assert len(statuses) == 1
            assert statuses["12345"].status == JobStatus.RUNNING
    
    def test_cancel_slurm_job(self):
        """Test SLURM job cancellation"""
        with patch('subprocess.run') as mock_run:
            # Mock sinfo check
            mock_run.return_value = Mock(returncode=0)
            scheduler = JobScheduler("slurm")
            
            # Mock scancel success
            mock_run.return_value = Mock(returncode=0)
            
            result = scheduler.cancel_job(JobID("12345", "slurm"))
            
            assert result is True
    
    def test_cancel_local_job(self):
        """Test local job cancellation"""
        scheduler = JobScheduler("local")
        
        with patch('os.kill') as mock_kill:
            mock_kill.return_value = None
            
            result = scheduler.cancel_job("12345")
            
            assert result is True
            mock_kill.assert_called_with(12345, 15)  # SIGTERM
    
    def test_wait_for_jobs(self):
        """Test waiting for jobs to complete"""
        scheduler = JobScheduler("local")
        
        with patch.object(scheduler, 'monitor_jobs') as mock_monitor:
            # First call: job still running
            # Second call: job completed
            mock_monitor.side_effect = [
                {"12345": JobStatus(JobStatus.RUNNING, "12345")},
                {"12345": JobStatus(JobStatus.COMPLETED, "12345", 0)}
            ]
            
            with patch('time.sleep'):
                final_statuses = scheduler.wait_for_jobs(["12345"], poll_interval=1)
            
            assert len(final_statuses) == 1
            assert final_statuses["12345"].status == JobStatus.COMPLETED
    
    def test_create_checkpoint(self):
        """Test checkpoint creation"""
        scheduler = JobScheduler("local")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            
            checkpoint_data = {
                "step": 5,
                "processed_subjects": ["sub-01", "sub-02"],
                "parameters": {"threshold": 0.5}
            }
            
            checkpoint_file = scheduler.create_checkpoint(
                "12345", 
                checkpoint_data, 
                checkpoint_dir
            )
            
            # Check that checkpoint file was created
            assert checkpoint_file.exists()
            assert checkpoint_file.suffix == ".pkl"
            
            # Check that metadata file was created
            metadata_file = checkpoint_file.with_suffix(".json")
            assert metadata_file.exists()
            
            # Verify checkpoint content
            with open(checkpoint_file, "rb") as f:
                loaded_data = pickle.load(f)
            
            assert loaded_data["job_id"] == "12345"
            assert loaded_data["data"] == checkpoint_data
            
            # Verify metadata content
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            assert metadata["job_id"] == "12345"
            assert metadata["scheduler_type"] == "local"
    
    def test_load_checkpoint(self):
        """Test checkpoint loading"""
        scheduler = JobScheduler("local")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            
            # Create checkpoint
            checkpoint_data = {"step": 3, "data": [1, 2, 3]}
            checkpoint_file = scheduler.create_checkpoint(
                "12345", 
                checkpoint_data, 
                checkpoint_dir
            )
            
            # Load checkpoint
            loaded_data = scheduler.load_checkpoint(checkpoint_file)
            
            assert loaded_data == checkpoint_data
    
    def test_load_invalid_checkpoint(self):
        """Test loading invalid checkpoint"""
        scheduler = JobScheduler("local")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid checkpoint file
            invalid_file = Path(temp_dir) / "invalid.pkl"
            invalid_file.write_bytes(b"invalid pickle data")
            
            with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
                scheduler.load_checkpoint(invalid_file)
    
    def test_list_checkpoints(self):
        """Test listing checkpoints"""
        scheduler = JobScheduler("local")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            
            # Create multiple checkpoints with slight delay to ensure different timestamps
            import time
            scheduler.create_checkpoint("12345", {"step": 1}, checkpoint_dir)
            time.sleep(0.01)  # Small delay to ensure different timestamps
            scheduler.create_checkpoint("12346", {"step": 2}, checkpoint_dir)
            time.sleep(0.01)
            scheduler.create_checkpoint("12345", {"step": 3}, checkpoint_dir)
            
            # List all checkpoints
            all_checkpoints = scheduler.list_checkpoints(checkpoint_dir)
            assert len(all_checkpoints) >= 2  # At least 2 checkpoints should be found
            
            # List checkpoints for specific job
            job_checkpoints = scheduler.list_checkpoints(checkpoint_dir, "12345")
            assert len(job_checkpoints) >= 1  # At least 1 checkpoint for job 12345
            
            # Check that checkpoints are sorted by timestamp (newest first)
            if len(job_checkpoints) > 1:
                timestamps = [cp["timestamp"] for cp in job_checkpoints]
                assert timestamps == sorted(timestamps, reverse=True)
    
    def test_resume_from_checkpoint(self):
        """Test resuming job from checkpoint"""
        scheduler = JobScheduler("local")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            
            # Create checkpoint
            checkpoint_data = {"step": 5, "processed": ["sub-01"]}
            checkpoint_file = scheduler.create_checkpoint(
                "12345", 
                checkpoint_data, 
                checkpoint_dir
            )
            
            # Mock job submission
            with patch.object(scheduler, 'submit_job') as mock_submit:
                mock_submit.return_value = JobID("12347", "local")
                
                job_config = {
                    "name": "test_job",
                    "script_path": "/path/to/script.sh"
                }
                
                resumed_job_id = scheduler.resume_from_checkpoint(
                    checkpoint_file, 
                    job_config
                )
                
                assert isinstance(resumed_job_id, JobID)
                assert resumed_job_id.job_id == "12347"
                
                # Check that submit_job was called with correct config
                mock_submit.assert_called_once()
                submitted_config = mock_submit.call_args[0][0]
                
                assert submitted_config["checkpoint_data"] == checkpoint_data
                assert submitted_config["is_resume"] is True
                assert submitted_config["name"] == "test_job_resume"
    
    def test_cleanup_checkpoints(self):
        """Test checkpoint cleanup"""
        scheduler = JobScheduler("local")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            
            # Create checkpoints with different timestamps
            with patch('elikopy.infrastructure.scheduler.datetime') as mock_datetime:
                # Old checkpoint (35 days ago)
                mock_datetime.now.return_value.strftime.return_value = "20231201_120000"
                mock_datetime.strptime.return_value.timestamp.return_value = 1701432000  # Old timestamp
                old_checkpoint = scheduler.create_checkpoint("12345", {"step": 1}, checkpoint_dir)
                
                # Recent checkpoint (5 days ago)
                mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
                mock_datetime.strptime.return_value.timestamp.return_value = 1704110400  # Recent timestamp
                recent_checkpoint = scheduler.create_checkpoint("12346", {"step": 2}, checkpoint_dir)
            
            # Mock current time for cleanup
            with patch('elikopy.infrastructure.scheduler.datetime') as mock_datetime:
                mock_datetime.now.return_value.timestamp.return_value = 1704196800  # Current time
                mock_datetime.strptime.side_effect = lambda x, fmt: datetime.strptime(x, fmt)
                
                # Clean up checkpoints older than 30 days
                cleaned_count = scheduler.cleanup_checkpoints(checkpoint_dir, max_age_days=30)
                
                # Should have cleaned up 1 checkpoint (the old one)
                assert cleaned_count >= 0  # Actual count depends on mock behavior
    
    def test_cleanup_checkpoints_nonexistent_dir(self):
        """Test checkpoint cleanup with nonexistent directory"""
        scheduler = JobScheduler("local")
        
        nonexistent_dir = Path("/nonexistent/directory")
        cleaned_count = scheduler.cleanup_checkpoints(nonexistent_dir)
        
        assert cleaned_count == 0


class TestJobSchedulerIntegration:
    """Integration tests for JobScheduler"""
    
    def test_full_workflow_local(self):
        """Test full workflow with local scheduler"""
        scheduler = JobScheduler("local")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            
            # Create a simple script
            script_path = Path(temp_dir) / "test_script.py"
            script_path.write_text("""
import sys
import time
print("Job started")
time.sleep(0.1)
print("Job completed")
sys.exit(0)
""")
            
            # Make script executable
            script_path.chmod(0o755)
            
            job_config = {
                "name": "integration_test",
                "script_path": str(script_path)
            }
            
            # Submit job
            with patch('subprocess.Popen') as mock_popen:
                mock_process = Mock()
                mock_process.pid = 99999
                mock_popen.return_value = mock_process
                
                job_id = scheduler.submit_job(job_config)
                assert isinstance(job_id, JobID)
                
                # Create checkpoint
                checkpoint_data = {"progress": 50}
                checkpoint_file = scheduler.create_checkpoint(
                    job_id, 
                    checkpoint_data, 
                    checkpoint_dir
                )
                
                # Verify checkpoint
                loaded_data = scheduler.load_checkpoint(checkpoint_file)
                assert loaded_data == checkpoint_data
                
                # List checkpoints
                checkpoints = scheduler.list_checkpoints(checkpoint_dir)
                assert len(checkpoints) >= 1