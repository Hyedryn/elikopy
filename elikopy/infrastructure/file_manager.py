"""
FileManager class - File operations and management
"""

import os
import shutil
import hashlib
import tempfile
import stat
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from .exceptions import ElikopyError, FileOperationError, ErrorContext


class FileManager:
    """Class for file operations and management"""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize file manager
        
        Args:
            base_path: Base path for file operations (optional)
        """
        self.base_path = Path(base_path) if base_path else None
        self.temp_dirs: List[Path] = []
    
    def create_directory_structure(self, structure: Dict[str, Any], 
                                 base_path: Optional[Path] = None) -> Dict[str, Path]:
        """Create directory structure safely with proper permissions
        
        Args:
            structure: Directory structure specification
            base_path: Base path for directory creation (optional)
            
        Returns:
            Dictionary mapping structure keys to created paths
            
        Raises:
            FileOperationError: If directory creation fails
        """
        # Use provided base path or instance base path
        base = Path(base_path) if base_path else self.base_path
        if not base:
            raise ValueError("Base path not specified")
        
        created_paths = {}
        
        try:
            # Create directories
            for key, value in structure.items():
                if isinstance(value, str):
                    # Create directory
                    dir_path = base / value
                    self._create_directory_with_permissions(dir_path)
                    created_paths[key] = dir_path
                elif isinstance(value, dict):
                    # Create parent directory
                    parent_path = base / key
                    self._create_directory_with_permissions(parent_path)
                    
                    # Create subdirectories
                    sub_paths = self.create_directory_structure(value, parent_path)
                    
                    # Add parent path
                    created_paths[key] = parent_path
                    
                    # Add subdirectories with qualified keys
                    for sub_key, sub_path in sub_paths.items():
                        created_paths[f"{key}/{sub_key}"] = sub_path
        
        except Exception as e:
            context = ErrorContext(operation="create_directory_structure", file_path=base)
            raise FileOperationError(
                f"Failed to create directory structure: {e}",
                context=context,
                cause=e
            )
        
        return created_paths
    
    def _create_directory_with_permissions(self, dir_path: Path) -> None:
        """Create directory with proper permissions
        
        Args:
            dir_path: Path to directory to create
        """
        try:
            dir_path.mkdir(exist_ok=True, parents=True)
            # Set appropriate permissions (readable/writable by owner, readable by group)
            dir_path.chmod(0o755)
        except PermissionError as e:
            context = ErrorContext(operation="create_directory", file_path=dir_path)
            raise FileOperationError(
                f"Permission denied creating directory {dir_path}: {e}",
                file_path=dir_path,
                operation="create_directory",
                context=context,
                cause=e
            )
        except OSError as e:
            context = ErrorContext(operation="create_directory", file_path=dir_path)
            raise FileOperationError(
                f"OS error creating directory {dir_path}: {e}",
                file_path=dir_path,
                operation="create_directory",
                context=context,
                cause=e
            )
    
    def copy_with_validation(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """Copy files with integrity validation
        
        Args:
            src: Source file path
            dst: Destination file path
            
        Returns:
            True if copy was successful and validated
            
        Raises:
            FileOperationError: If copy operation fails
        """
        src_path = Path(src)
        dst_path = Path(dst)
        
        # Check if source exists
        if not src_path.exists():
            context = ErrorContext(operation="copy_file", file_path=src_path)
            raise FileOperationError(
                f"Source file not found: {src_path}",
                file_path=src_path,
                operation="copy_file",
                context=context
            )
        
        # Check if source is readable
        if not os.access(src_path, os.R_OK):
            context = ErrorContext(operation="copy_file", file_path=src_path)
            raise FileOperationError(
                f"Source file not readable: {src_path}",
                file_path=src_path,
                operation="copy_file",
                context=context
            )
        
        try:
            # Create destination directory if it doesn't exist
            self._create_directory_with_permissions(dst_path.parent)
            
            # Calculate source file hash
            src_hash = self._calculate_file_hash(src_path)
            
            # Copy file
            shutil.copy2(src_path, dst_path)
            
            # Calculate destination file hash
            dst_hash = self._calculate_file_hash(dst_path)
            
            # Validate hashes
            if src_hash != dst_hash:
                # Clean up corrupted destination file
                self.safe_delete(dst_path)
                context = ErrorContext(operation="copy_file_validation", file_path=dst_path)
                raise FileOperationError(
                    f"File integrity validation failed for {dst_path}",
                    file_path=dst_path,
                    operation="copy_file_validation",
                    context=context
                )
            
            return True
            
        except shutil.Error as e:
            context = ErrorContext(operation="copy_file", file_path=dst_path)
            raise FileOperationError(
                f"Copy operation failed: {e}",
                file_path=dst_path,
                operation="copy_file",
                context=context,
                cause=e
            )
        except Exception as e:
            context = ErrorContext(operation="copy_file", file_path=dst_path)
            raise FileOperationError(
                f"Unexpected error during file copy: {e}",
                file_path=dst_path,
                operation="copy_file",
                context=context,
                cause=e
            )
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file hash
        
        Args:
            file_path: Path to file
            
        Returns:
            File hash as hexadecimal string
            
        Raises:
            FileOperationError: If hash calculation fails
        """
        try:
            hash_md5 = hashlib.md5()
            
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            return hash_md5.hexdigest()
            
        except IOError as e:
            context = ErrorContext(operation="calculate_hash", file_path=file_path)
            raise FileOperationError(
                f"Failed to calculate hash for {file_path}: {e}",
                file_path=file_path,
                operation="calculate_hash",
                context=context,
                cause=e
            )
    
    def create_temp_directory(self, prefix: str = "elikopy_") -> Path:
        """Create temporary directory
        
        Args:
            prefix: Directory name prefix
            
        Returns:
            Path to temporary directory
        """
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup_temporary_files(self, temp_dir: Optional[Path] = None) -> None:
        """Clean up temporary files
        
        Args:
            temp_dir: Specific temporary directory to clean up (optional)
            
        Raises:
            FileOperationError: If cleanup fails
        """
        cleanup_errors = []
        
        if temp_dir:
            # Clean up specific directory
            try:
                if temp_dir.exists():
                    self._remove_readonly_and_delete(temp_dir)
                
                # Remove from tracked directories
                if temp_dir in self.temp_dirs:
                    self.temp_dirs.remove(temp_dir)
            except Exception as e:
                cleanup_errors.append(f"Failed to cleanup {temp_dir}: {e}")
        else:
            # Clean up all tracked directories
            for dir_path in list(self.temp_dirs):  # Create copy to avoid modification during iteration
                try:
                    if dir_path.exists():
                        self._remove_readonly_and_delete(dir_path)
                    self.temp_dirs.remove(dir_path)
                except Exception as e:
                    cleanup_errors.append(f"Failed to cleanup {dir_path}: {e}")
        
        if cleanup_errors:
            context = ErrorContext(operation="cleanup_temporary_files")
            raise FileOperationError(
                f"Cleanup failed for some temporary files: {'; '.join(cleanup_errors)}",
                operation="cleanup_temporary_files",
                context=context
            )
    
    def _remove_readonly_and_delete(self, path: Path) -> None:
        """Remove read-only permissions and delete path
        
        Args:
            path: Path to delete
        """
        def handle_remove_readonly(func, path, exc):
            """Error handler for removing read-only files"""
            if os.path.exists(path):
                os.chmod(path, stat.S_IWRITE)
                func(path)
        
        if path.is_dir():
            shutil.rmtree(path, onerror=handle_remove_readonly)
        else:
            try:
                path.unlink()
            except PermissionError:
                path.chmod(stat.S_IWRITE)
                path.unlink()
    
    def safe_delete(self, path: Union[str, Path]) -> bool:
        """Safely delete file or directory
        
        Args:
            path: Path to file or directory
            
        Returns:
            True if deletion was successful
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            return True
        
        try:
            self._remove_readonly_and_delete(path_obj)
            return True
        except Exception:
            return False
    
    def get_file_size(self, path: Union[str, Path]) -> int:
        """Get file size in bytes
        
        Args:
            path: Path to file
            
        Returns:
            File size in bytes
            
        Raises:
            FileOperationError: If file size cannot be determined
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            context = ErrorContext(operation="get_file_size", file_path=path_obj)
            raise FileOperationError(
                f"File not found: {path_obj}",
                file_path=path_obj,
                operation="get_file_size",
                context=context
            )
        
        try:
            return path_obj.stat().st_size
        except OSError as e:
            context = ErrorContext(operation="get_file_size", file_path=path_obj)
            raise FileOperationError(
                f"Failed to get file size for {path_obj}: {e}",
                file_path=path_obj,
                operation="get_file_size",
                context=context,
                cause=e
            )
    
    def get_directory_size(self, path: Union[str, Path]) -> int:
        """Get directory size in bytes
        
        Args:
            path: Path to directory
            
        Returns:
            Directory size in bytes
            
        Raises:
            FileOperationError: If directory size cannot be calculated
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            context = ErrorContext(operation="get_directory_size", file_path=path_obj)
            raise FileOperationError(
                f"Directory not found: {path_obj}",
                file_path=path_obj,
                operation="get_directory_size",
                context=context
            )
        
        if not path_obj.is_dir():
            context = ErrorContext(operation="get_directory_size", file_path=path_obj)
            raise FileOperationError(
                f"Not a directory: {path_obj}",
                file_path=path_obj,
                operation="get_directory_size",
                context=context
            )
        
        total_size = 0
        access_errors = []
        
        try:
            for dirpath, dirnames, filenames in os.walk(path_obj):
                for filename in filenames:
                    try:
                        file_path = Path(dirpath) / filename
                        total_size += file_path.stat().st_size
                    except (OSError, PermissionError) as e:
                        access_errors.append(f"{file_path}: {e}")
        except Exception as e:
            context = ErrorContext(operation="get_directory_size", file_path=path_obj)
            raise FileOperationError(
                f"Failed to calculate directory size for {path_obj}: {e}",
                file_path=path_obj,
                operation="get_directory_size",
                context=context,
                cause=e
            )
        
        if access_errors:
            # Log warnings but don't fail completely
            # In a real implementation, this would use the logging system
            pass
        
        return total_size
    
    def __del__(self):
        """Destructor to clean up temporary files"""
        self.cleanup_temporary_files()