"""
JobScheduler class - HPC/SLURM job management
"""


@dataclass
class ProcessingJob:
    """Class representing a processing job configuration"""
    
    name: str
    script_path: Path
    script_args: List[str] = field(default_factory=list)
    cpus_per_task: int = 1
    mem_per_cpu: int = 4  # GB
    time_limit: str = "24:00:00"
    use_gpu: bool = False
    gpu_count: int = 0
    dependencies: List[str] = field(default_factory=list)
    partition: Optional[str] = None
    account: Optional[str] = None
    output_file: Optional[str] = None
    error_file: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ProcessingJob to dictionary format
        
        Returns:
            Dictionary representation of the job
        """
        return {
            "name": self.name,
            "script_path": str(self.script_path),
            "script_args": self.script_args,
            "cpus_per_task": self.cpus_per_task,
            "mem_per_cpu": self.mem_per_cpu,
            "time_limit": self.time_limit,
            "use_gpu": self.use_gpu,
            "gpu_count": self.gpu_count,
            "dependencies": self.dependencies,
            "partition": self.partition,
            "account": self.account,
            "output_file": self.output_file,
            "error_file": self.error_file,
            "environment_vars": self.environment_vars
        }

import os
import subprocess
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass, field
from dataclasses import dataclass, field


@dataclass
class ProcessingJob:
    """Class representing a processing job"""
    
    name: str
    script_path: Path
    script_args: List[str] = field(default_factory=list)
    working_dir: Optional[Path] = None
    output_file: Optional[str] = None
    error_file: Optional[str] = None
    cpus_per_task: int = 1
    mem_per_cpu: int = 4  # GB
    time_limit: str = "24:00:00"
    partition: Optional[str] = None
    account: Optional[str] = None
    use_gpu: bool = False
    gpu_count: int = 0
    dependencies: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for scheduler submission"""
        job_dict = {
            "name": self.name,
            "script_path": str(self.script_path),
            "script_args": self.script_args,
            "cpus_per_task": self.cpus_per_task,
            "mem_per_cpu": self.mem_per_cpu,
            "time_limit": self.time_limit,
            "use_gpu": self.use_gpu,
            "gpu_count": self.gpu_count,
            "dependencies": self.dependencies
        }
        
        if self.working_dir:
            job_dict["working_dir"] = str(self.working_dir)
        if self.output_file:
            job_dict["output_file"] = self.output_file
        if self.error_file:
            job_dict["error_file"] = self.error_file
        if self.partition:
            job_dict["partition"] = self.partition
        if self.account:
            job_dict["account"] = self.account
        if self.environment_vars:
            job_dict["environment_vars"] = self.environment_vars
            
        return job_dict


class JobID:
    """Class representing a job ID"""
    
    def __init__(self, job_id: str, scheduler_type: str = "slurm"):
        """Initialize a job ID
        
        Args:
            job_id: Job ID string
            scheduler_type: Scheduler type (slurm, local)
        """
        self.job_id = job_id
        self.scheduler_type = scheduler_type
    
    def __str__(self) -> str:
        """String representation"""
        return self.job_id


class JobStatus:
    """Class representing a job status"""
    
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"
    
    def __init__(self, status: str, job_id: str, exit_code: Optional[int] = None):
        """Initialize a job status
        
        Args:
            status: Status string
            job_id: Job ID
            exit_code: Exit code (optional)
        """
        self.status = status
        self.job_id = job_id
        self.exit_code = exit_code
    
    def __str__(self) -> str:
        """String representation"""
        if self.exit_code is not None:
            return f"{self.status} (exit code: {self.exit_code})"
        return self.status


class JobScheduler:
    """Class for HPC job scheduling"""
    
    def __init__(self, scheduler_type: str = "slurm", config: Optional[Dict[str, Any]] = None):
        """Initialize job scheduler
        
        Args:
            scheduler_type: Scheduler type (slurm, local)
            config: Scheduler configuration
        """
        self.scheduler_type = scheduler_type
        self.config = config or {}
        
        # Validate scheduler type
        if scheduler_type not in ["slurm", "local"]:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        # Check if scheduler is available
        if scheduler_type == "slurm":
            self._check_slurm_available()
    
    def _check_slurm_available(self) -> None:
        """Check if SLURM is available"""
        try:
            subprocess.run(["sinfo", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError("SLURM is not available on this system")
    
    def submit_job(self, job: Union[ProcessingJob, Dict[str, Any]]) -> JobID:
        """Submit processing job
        
        Args:
            job: ProcessingJob object or job configuration dictionary
            
        Returns:
            JobID object
        """
        # Convert ProcessingJob to dictionary if needed
        if isinstance(job, ProcessingJob):
            job_config = job.to_dict()
        else:
            job_config = job
            
        if self.scheduler_type == "slurm":
            return self._submit_slurm_job(job_config)
        else:
            return self._submit_local_job(job_config)
    
    def _submit_slurm_job(self, job_config: Dict[str, Any]) -> JobID:
        """Submit job to SLURM
        
        Args:
            job_config: Job configuration
            
        Returns:
            JobID object
        """
        # Extract job parameters
        job_name = job_config.get("name", "elikopy_job")
        script_path = job_config.get("script_path")
        
        if not script_path:
            raise ValueError("script_path is required for SLURM jobs")
        
        # Build sbatch command
        cmd = ["sbatch"]
        
        # Add job name
        cmd.extend(["--job-name", job_name])
        
        # Add output file
        output_file = job_config.get("output_file", f"{job_name}.out")
        cmd.extend(["--output", output_file])
        
        # Add error file
        error_file = job_config.get("error_file", f"{job_name}.err")
        cmd.extend(["--error", error_file])
        
        # Add CPU and memory requirements
        cpus_per_task = job_config.get("cpus_per_task", self.config.get("cpus_per_task", 1))
        cmd.extend(["--cpus-per-task", str(cpus_per_task)])
        
        mem_per_cpu = job_config.get("mem_per_cpu", self.config.get("mem_per_cpu", 4))
        cmd.extend(["--mem-per-cpu", f"{mem_per_cpu}G"])
        
        # Add time limit
        time_limit = job_config.get("time_limit", self.config.get("time_limit", "24:00:00"))
        cmd.extend(["--time", time_limit])
        
        # Add partition if specified
        partition = job_config.get("partition", self.config.get("partition"))
        if partition:
            cmd.extend(["--partition", partition])
        
        # Add account if specified
        account = job_config.get("account", self.config.get("account"))
        if account:
            cmd.extend(["--account", account])
        
        # Add GPU requirements if specified
        use_gpu = job_config.get("use_gpu", self.config.get("use_gpu", False))
        if use_gpu:
            gpu_count = job_config.get("gpu_count", self.config.get("gpu_count", 1))
            cmd.extend(["--gres", f"gpu:{gpu_count}"])
        
        # Add dependencies if specified
        dependencies = job_config.get("dependencies", [])
        if dependencies:
            dep_str = "afterok:" + ":".join(dependencies)
            cmd.extend(["--dependency", dep_str])
        
        # Add script path
        cmd.append(str(script_path))
        
        # Add script arguments
        script_args = job_config.get("script_args", [])
        cmd.extend(script_args)
        
        # Submit job
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to submit job: {result.stderr}")
            
            # Extract job ID from output
            # Output format: "Submitted batch job 123456"
            job_id = result.stdout.strip().split()[-1]
            
            return JobID(job_id, "slurm")
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Failed to submit job: {e}")
    
    def _submit_local_job(self, job_config: Dict[str, Any]) -> JobID:
        """Submit job locally
        
        Args:
            job_config: Job configuration
            
        Returns:
            JobID object
        """
        # Extract job parameters
        script_path = job_config.get("script_path")
        
        if not script_path:
            raise ValueError("script_path is required for local jobs")
        
        # Build command
        cmd = [str(script_path)]
        
        # Add script arguments
        script_args = job_config.get("script_args", [])
        cmd.extend(script_args)
        
        # Run job in background
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Use process ID as job ID
            job_id = str(process.pid)
            
            return JobID(job_id, "local")
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Failed to submit job: {e}")
    
    def monitor_jobs(self, job_ids: List[Union[JobID, str]]) -> Dict[str, JobStatus]:
        """Monitor job status
        
        Args:
            job_ids: List of JobID objects or job ID strings
            
        Returns:
            Dictionary mapping job IDs to JobStatus objects
        """
        # Normalize job IDs
        normalized_job_ids = []
        for job_id in job_ids:
            if isinstance(job_id, JobID):
                normalized_job_ids.append(job_id.job_id)
            else:
                normalized_job_ids.append(job_id)
        
        if self.scheduler_type == "slurm":
            return self._monitor_slurm_jobs(normalized_job_ids)
        else:
            return self._monitor_local_jobs(normalized_job_ids)
    
    def _monitor_slurm_jobs(self, job_ids: List[str]) -> Dict[str, JobStatus]:
        """Monitor SLURM jobs
        
        Args:
            job_ids: List of job ID strings
            
        Returns:
            Dictionary mapping job IDs to JobStatus objects
        """
        # Build sacct command
        cmd = [
            "sacct",
            "-j", ",".join(job_ids),
            "--format=JobID,State,ExitCode",
            "--parsable2",
            "--noheader"
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to monitor jobs: {result.stderr}")
            
            # Parse output
            job_statuses = {}
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                
                parts = line.split("|")
                if len(parts) < 3:
                    continue
                
                job_id = parts[0].split(".")[0]  # Remove .batch suffix
                state = parts[1]
                exit_code = parts[2].split(":")[0]  # Format: "0:0"
                
                # Map SLURM state to JobStatus
                status = JobStatus.UNKNOWN
                if state in ["PENDING", "CONFIGURING", "SUSPENDED"]:
                    status = JobStatus.PENDING
                elif state in ["RUNNING", "COMPLETING"]:
                    status = JobStatus.RUNNING
                elif state in ["COMPLETED"]:
                    status = JobStatus.COMPLETED
                elif state in ["FAILED", "TIMEOUT", "OUT_OF_MEMORY"]:
                    status = JobStatus.FAILED
                elif state in ["CANCELLED", "PREEMPTED"]:
                    status = JobStatus.CANCELLED
                
                job_statuses[job_id] = JobStatus(status, job_id, int(exit_code) if exit_code.isdigit() else None)
            
            return job_statuses
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Failed to monitor jobs: {e}")
    
    def _monitor_local_jobs(self, job_ids: List[str]) -> Dict[str, JobStatus]:
        """Monitor local jobs
        
        Args:
            job_ids: List of job ID strings (process IDs)
            
        Returns:
            Dictionary mapping job IDs to JobStatus objects
        """
        job_statuses = {}
        
        for job_id in job_ids:
            try:
                # Check if process exists
                os.kill(int(job_id), 0)
                
                # Process exists, check if it's a zombie
                with open(f"/proc/{job_id}/stat", "r") as f:
                    stat = f.read().split()
                    state = stat[2]
                    
                    if state == "Z":
                        job_statuses[job_id] = JobStatus(JobStatus.COMPLETED, job_id)
                    else:
                        job_statuses[job_id] = JobStatus(JobStatus.RUNNING, job_id)
            except ProcessLookupError:
                # Process doesn't exist, check exit code
                try:
                    with open(f"{job_id}.exitcode", "r") as f:
                        exit_code = int(f.read().strip())
                        
                        if exit_code == 0:
                            job_statuses[job_id] = JobStatus(JobStatus.COMPLETED, job_id, exit_code)
                        else:
                            job_statuses[job_id] = JobStatus(JobStatus.FAILED, job_id, exit_code)
                except (FileNotFoundError, ValueError):
                    # Can't determine exit code
                    job_statuses[job_id] = JobStatus(JobStatus.UNKNOWN, job_id)
            except (FileNotFoundError, ValueError):
                # Can't determine status
                job_statuses[job_id] = JobStatus(JobStatus.UNKNOWN, job_id)
        
        return job_statuses
    
    def cancel_job(self, job_id: Union[JobID, str]) -> bool:
        """Cancel running job
        
        Args:
            job_id: JobID object or job ID string
            
        Returns:
            True if job was cancelled
        """
        # Normalize job ID
        if isinstance(job_id, JobID):
            job_id_str = job_id.job_id
        else:
            job_id_str = job_id
        
        if self.scheduler_type == "slurm":
            return self._cancel_slurm_job(job_id_str)
        else:
            return self._cancel_local_job(job_id_str)
    
    def _cancel_slurm_job(self, job_id: str) -> bool:
        """Cancel SLURM job
        
        Args:
            job_id: Job ID string
            
        Returns:
            True if job was cancelled
        """
        try:
            result = subprocess.run(
                ["scancel", job_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            return result.returncode == 0
        except subprocess.SubprocessError:
            return False
    
    def _cancel_local_job(self, job_id: str) -> bool:
        """Cancel local job
        
        Args:
            job_id: Job ID string (process ID)
            
        Returns:
            True if job was cancelled
        """
        try:
            os.kill(int(job_id), 15)  # SIGTERM
            return True
        except (ProcessLookupError, ValueError):
            return False
    
    def wait_for_jobs(self, job_ids: List[Union[JobID, str]], 
                     poll_interval: int = 10) -> Dict[str, JobStatus]:
        """Wait for jobs to complete
        
        Args:
            job_ids: List of JobID objects or job ID strings
            poll_interval: Polling interval in seconds
            
        Returns:
            Dictionary mapping job IDs to final JobStatus objects
        """
        # Normalize job IDs
        normalized_job_ids = []
        for job_id in job_ids:
            if isinstance(job_id, JobID):
                normalized_job_ids.append(job_id.job_id)
            else:
                normalized_job_ids.append(job_id)
        
        # Monitor jobs until all complete
        pending_jobs = set(normalized_job_ids)
        final_statuses = {}
        
        while pending_jobs:
            # Get current status
            statuses = self.monitor_jobs(list(pending_jobs))
            
            # Update final statuses and remove completed jobs
            for job_id, status in statuses.items():
                if status.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    final_statuses[job_id] = status
                    pending_jobs.remove(job_id)
            
            # Wait before polling again
            if pending_jobs:
                time.sleep(poll_interval)
        
        return final_statuses
    
    def create_checkpoint(self, job_id: Union[JobID, str], 
                         checkpoint_data: Dict[str, Any],
                         checkpoint_dir: Optional[Path] = None) -> Path:
        """Create checkpoint for job
        
        Args:
            job_id: JobID object or job ID string
            checkpoint_data: Data to checkpoint
            checkpoint_dir: Directory to store checkpoint (optional)
            
        Returns:
            Path to checkpoint file
        """
        # Normalize job ID
        if isinstance(job_id, JobID):
            job_id_str = job_id.job_id
        else:
            job_id_str = job_id
        
        # Set default checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = Path.cwd() / "checkpoints"
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = checkpoint_dir / f"checkpoint_{job_id_str}_{timestamp}.pkl"
        
        # Add metadata to checkpoint data
        checkpoint_data_with_meta = {
            "job_id": job_id_str,
            "timestamp": timestamp,
            "scheduler_type": self.scheduler_type,
            "data": checkpoint_data
        }
        
        # Save checkpoint
        try:
            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint_data_with_meta, f)
            
            # Also save a JSON metadata file for easier inspection
            metadata_file = checkpoint_file.with_suffix(".json")
            metadata = {
                "job_id": job_id_str,
                "timestamp": timestamp,
                "scheduler_type": self.scheduler_type,
                "checkpoint_file": str(checkpoint_file),
                "data_keys": list(checkpoint_data.keys()) if isinstance(checkpoint_data, dict) else []
            }
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return checkpoint_file
        except Exception as e:
            raise RuntimeError(f"Failed to create checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_file: Path) -> Dict[str, Any]:
        """Load checkpoint data
        
        Args:
            checkpoint_file: Path to checkpoint file
            
        Returns:
            Checkpoint data
        """
        try:
            with open(checkpoint_file, "rb") as f:
                checkpoint_data = pickle.load(f)
            
            # Validate checkpoint format
            if not isinstance(checkpoint_data, dict) or "data" not in checkpoint_data:
                raise ValueError("Invalid checkpoint format")
            
            return checkpoint_data["data"]
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    def list_checkpoints(self, checkpoint_dir: Optional[Path] = None,
                        job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available checkpoints
        
        Args:
            checkpoint_dir: Directory to search for checkpoints (optional)
            job_id: Filter by job ID (optional)
            
        Returns:
            List of checkpoint metadata
        """
        # Set default checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = Path.cwd() / "checkpoints"
        
        if not checkpoint_dir.exists():
            return []
        
        checkpoints = []
        
        # Search for checkpoint metadata files
        pattern = f"checkpoint_{job_id}_*.json" if job_id else "checkpoint_*.json"
        
        for metadata_file in checkpoint_dir.glob(pattern):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Check if corresponding pickle file exists
                checkpoint_file = Path(metadata["checkpoint_file"])
                if checkpoint_file.exists():
                    checkpoints.append(metadata)
            except Exception:
                # Skip invalid metadata files
                continue
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return checkpoints
    
    def resume_from_checkpoint(self, checkpoint_file: Path,
                              job_config: Dict[str, Any]) -> JobID:
        """Resume job from checkpoint
        
        Args:
            checkpoint_file: Path to checkpoint file
            job_config: Job configuration for resumed job
            
        Returns:
            JobID of resumed job
        """
        # Load checkpoint data
        checkpoint_data = self.load_checkpoint(checkpoint_file)
        
        # Update job config with checkpoint data
        resumed_job_config = job_config.copy()
        resumed_job_config["checkpoint_data"] = checkpoint_data
        resumed_job_config["is_resume"] = True
        
        # Add resume flag to job name
        original_name = resumed_job_config.get("name", "elikopy_job")
        resumed_job_config["name"] = f"{original_name}_resume"
        
        # Submit resumed job
        return self.submit_job(resumed_job_config)
    
    def cleanup_checkpoints(self, checkpoint_dir: Optional[Path] = None,
                           max_age_days: int = 30,
                           job_id: Optional[str] = None) -> int:
        """Clean up old checkpoints
        
        Args:
            checkpoint_dir: Directory to clean up (optional)
            max_age_days: Maximum age of checkpoints to keep in days
            job_id: Clean up checkpoints for specific job ID (optional)
            
        Returns:
            Number of checkpoints cleaned up
        """
        # Set default checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = Path.cwd() / "checkpoints"
        
        if not checkpoint_dir.exists():
            return 0
        
        cleaned_count = 0
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        
        # Get list of checkpoints
        checkpoints = self.list_checkpoints(checkpoint_dir, job_id)
        
        for checkpoint_meta in checkpoints:
            try:
                # Parse timestamp
                timestamp_str = checkpoint_meta["timestamp"]
                checkpoint_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").timestamp()
                
                # Check if checkpoint is old enough to clean up
                if checkpoint_time < cutoff_time:
                    # Remove checkpoint files
                    checkpoint_file = Path(checkpoint_meta["checkpoint_file"])
                    metadata_file = checkpoint_file.with_suffix(".json")
                    
                    if checkpoint_file.exists():
                        checkpoint_file.unlink()
                    
                    if metadata_file.exists():
                        metadata_file.unlink()
                    
                    cleaned_count += 1
            except Exception:
                # Skip problematic checkpoints
                continue
        
        return cleaned_count