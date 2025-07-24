"""
JobScheduler class - HPC/SLURM job management
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any


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
    
    def submit_job(self, job_config: Dict[str, Any]) -> JobID:
        """Submit processing job
        
        Args:
            job_config: Job configuration
            
        Returns:
            JobID object
        """
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