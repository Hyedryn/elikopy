"""
FileManager class - File operations and management
"""

import os
import shutil
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any


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
        """Create directory structure safely
        
        Args:
            structure: Directory structure specification
            base_path: Base path for directory creation (optional)
            
        Returns:
            Dictionary mapping structure keys to created paths
        """
        # Use provided base path or instance base path
        base = Path(base_path) if base_path else self.base_path
        if not base:
            raise ValueError("Base path not specified")
        
        created_paths = {}
        
        # Create directories
        for key, value in structure.items():
            if isinstance(value, str):
                # Create directory
                dir_path = base / value
                dir_path.mkdir(exist_ok=True, parents=True)
                created_paths[key] = dir_path
            elif isinstance(value, dict):
                # Create parent directory
                parent_path = base / key
                parent_path.mkdir(exist_ok=True, parents=True)
                
                # Create subdirectories
                sub_paths = self.create_directory_structure(value, parent_path)
                
                # Add parent path
                created_paths[key] = parent_path
                
                # Add subdirectories with qualified keys
                for sub_key, sub_path in sub_paths.items():
                    created_paths[f"{key}/{sub_key}"] = sub_path
        
        return created_paths
    
    def copy_with_validation(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """Copy files with integrity validation
        
        Args:
            src: Source file path
            dst: Destination file path
            
        Returns:
            True if copy was successful and validated
        """
        src_path = Path(src)
        dst_path = Path(dst)
        
        # Check if source exists
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        
        # Create destination directory if it doesn't exist
        dst_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Calculate source file hash
        src_hash = self._calculate_file_hash(src_path)
        
        # Copy file
        shutil.copy2(src_path, dst_path)
        
        # Calculate destination file hash
        dst_hash = self._calculate_file_hash(dst_path)
        
        # Validate hashes
        return src_hash == dst_hash
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file hash
        
        Args:
            file_path: Path to file
            
        Returns:
            File hash as hexadecimal string
        """
        hash_md5 = hashlib.md5()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
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
        """
        if temp_dir:
            # Clean up specific directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            # Remove from tracked directories
            if temp_dir in self.temp_dirs:
                self.temp_dirs.remove(temp_dir)
        else:
            # Clean up all tracked directories
            for dir_path in self.temp_dirs:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
            
            self.temp_dirs = []
    
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
            if path_obj.is_dir():
                shutil.rmtree(path_obj)
            else:
                path_obj.unlink()
            return True
        except Exception:
            return False
    
    def get_file_size(self, path: Union[str, Path]) -> int:
        """Get file size in bytes
        
        Args:
            path: Path to file
            
        Returns:
            File size in bytes
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path_obj}")
        
        return path_obj.stat().st_size
    
    def get_directory_size(self, path: Union[str, Path]) -> int:
        """Get directory size in bytes
        
        Args:
            path: Path to directory
            
        Returns:
            Directory size in bytes
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Directory not found: {path_obj}")
        
        if not path_obj.is_dir():
            raise ValueError(f"Not a directory: {path_obj}")
        
        total_size = 0
        
        for dirpath, dirnames, filenames in os.walk(path_obj):
            for filename in filenames:
                file_path = Path(dirpath) / filename
                total_size += file_path.stat().st_size
        
        return total_size
    
    def __del__(self):
        """Destructor to clean up temporary files"""
        self.cleanup_temporary_files()