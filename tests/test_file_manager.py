"""
Unit tests for FileManager class
"""

import pytest
import tempfile
import shutil
import os
import stat
from pathlib import Path
from unittest.mock import patch, MagicMock

from elikopy.infrastructure.file_manager import FileManager
from elikopy.infrastructure.exceptions import FileOperationError


class TestFileManager:
    """Test cases for FileManager class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.file_manager = FileManager(base_path=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_init_with_base_path(self):
        """Test FileManager initialization with base path"""
        fm = FileManager(base_path=self.temp_dir)
        assert fm.base_path == self.temp_dir
        assert fm.temp_dirs == []
    
    def test_init_without_base_path(self):
        """Test FileManager initialization without base path"""
        fm = FileManager()
        assert fm.base_path is None
        assert fm.temp_dirs == []
    
    def test_create_directory_structure_simple(self):
        """Test creating simple directory structure"""
        structure = {
            "data": "data_dir",
            "results": "results_dir"
        }
        
        created_paths = self.file_manager.create_directory_structure(structure)
        
        assert "data" in created_paths
        assert "results" in created_paths
        assert created_paths["data"].exists()
        assert created_paths["results"].exists()
        assert created_paths["data"].is_dir()
        assert created_paths["results"].is_dir()
    
    def test_create_directory_structure_nested(self):
        """Test creating nested directory structure"""
        structure = {
            "project": {
                "data": "raw_data",
                "analysis": "analysis_results",
                "logs": "log_files"
            }
        }
        
        created_paths = self.file_manager.create_directory_structure(structure)
        
        assert "project" in created_paths
        assert "project/data" in created_paths
        assert "project/analysis" in created_paths
        assert "project/logs" in created_paths
        
        # Check all directories exist
        for path in created_paths.values():
            assert path.exists()
            assert path.is_dir()
    
    def test_create_directory_structure_with_custom_base_path(self):
        """Test creating directory structure with custom base path"""
        custom_base = self.temp_dir / "custom"
        structure = {"test": "test_dir"}
        
        created_paths = self.file_manager.create_directory_structure(
            structure, base_path=custom_base
        )
        
        assert created_paths["test"] == custom_base / "test_dir"
        assert created_paths["test"].exists()
    
    def test_create_directory_structure_no_base_path_error(self):
        """Test error when no base path is provided"""
        fm = FileManager()  # No base path
        structure = {"test": "test_dir"}
        
        with pytest.raises(ValueError, match="Base path not specified"):
            fm.create_directory_structure(structure)
    
    @patch('elikopy.infrastructure.file_manager.FileManager._create_directory_with_permissions')
    def test_create_directory_structure_permission_error(self, mock_create_dir):
        """Test handling of permission errors during directory creation"""
        mock_create_dir.side_effect = PermissionError("Permission denied")
        structure = {"test": "test_dir"}
        
        with pytest.raises(FileOperationError):
            self.file_manager.create_directory_structure(structure)
    
    def test_copy_with_validation_success(self):
        """Test successful file copy with validation"""
        # Create source file
        src_file = self.temp_dir / "source.txt"
        src_content = "Test content for file copy"
        src_file.write_text(src_content)
        
        # Copy file
        dst_file = self.temp_dir / "destination.txt"
        result = self.file_manager.copy_with_validation(src_file, dst_file)
        
        assert result is True
        assert dst_file.exists()
        assert dst_file.read_text() == src_content
    
    def test_copy_with_validation_source_not_found(self):
        """Test copy with non-existent source file"""
        src_file = self.temp_dir / "nonexistent.txt"
        dst_file = self.temp_dir / "destination.txt"
        
        with pytest.raises(FileOperationError, match="Source file not found"):
            self.file_manager.copy_with_validation(src_file, dst_file)
    
    def test_copy_with_validation_creates_destination_directory(self):
        """Test that copy creates destination directory if needed"""
        # Create source file
        src_file = self.temp_dir / "source.txt"
        src_file.write_text("test content")
        
        # Copy to nested destination
        dst_file = self.temp_dir / "nested" / "dir" / "destination.txt"
        result = self.file_manager.copy_with_validation(src_file, dst_file)
        
        assert result is True
        assert dst_file.exists()
        assert dst_file.parent.exists()
    
    @patch('shutil.copy2')
    def test_copy_with_validation_integrity_failure(self, mock_copy):
        """Test handling of integrity validation failure"""
        # Create source file
        src_file = self.temp_dir / "source.txt"
        src_file.write_text("original content")
        
        # Mock copy to create different content
        def mock_copy_func(src, dst):
            Path(dst).write_text("corrupted content")
        
        mock_copy.side_effect = mock_copy_func
        
        dst_file = self.temp_dir / "destination.txt"
        
        with pytest.raises(FileOperationError, match="File integrity validation failed"):
            self.file_manager.copy_with_validation(src_file, dst_file)
        
        # Destination file should be cleaned up
        assert not dst_file.exists()
    
    @patch('os.access')
    def test_copy_with_validation_source_not_readable(self, mock_access):
        """Test copy with unreadable source file"""
        mock_access.return_value = False
        
        # Create source file
        src_file = self.temp_dir / "source.txt"
        src_file.write_text("test content")
        
        dst_file = self.temp_dir / "destination.txt"
        
        with pytest.raises(FileOperationError, match="Source file not readable"):
            self.file_manager.copy_with_validation(src_file, dst_file)
    
    def test_calculate_file_hash(self):
        """Test file hash calculation"""
        # Create test file
        test_file = self.temp_dir / "test.txt"
        test_content = "Test content for hash calculation"
        test_file.write_text(test_content)
        
        # Calculate hash
        hash1 = self.file_manager._calculate_file_hash(test_file)
        hash2 = self.file_manager._calculate_file_hash(test_file)
        
        # Hash should be consistent
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
    
    def test_calculate_file_hash_nonexistent_file(self):
        """Test hash calculation for non-existent file"""
        nonexistent_file = self.temp_dir / "nonexistent.txt"
        
        with pytest.raises(FileOperationError, match="Failed to calculate hash"):
            self.file_manager._calculate_file_hash(nonexistent_file)
    
    def test_create_temp_directory(self):
        """Test temporary directory creation"""
        temp_dir = self.file_manager.create_temp_directory(prefix="test_")
        
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        assert temp_dir in self.file_manager.temp_dirs
        assert temp_dir.name.startswith("test_")
    
    def test_cleanup_temporary_files_specific(self):
        """Test cleanup of specific temporary directory"""
        # Create temporary directory
        temp_dir = self.file_manager.create_temp_directory()
        
        # Create some files in it
        (temp_dir / "test_file.txt").write_text("test content")
        
        assert temp_dir.exists()
        assert temp_dir in self.file_manager.temp_dirs
        
        # Clean up specific directory
        self.file_manager.cleanup_temporary_files(temp_dir)
        
        assert not temp_dir.exists()
        assert temp_dir not in self.file_manager.temp_dirs
    
    def test_cleanup_temporary_files_all(self):
        """Test cleanup of all temporary directories"""
        # Create multiple temporary directories
        temp_dir1 = self.file_manager.create_temp_directory(prefix="test1_")
        temp_dir2 = self.file_manager.create_temp_directory(prefix="test2_")
        
        assert len(self.file_manager.temp_dirs) == 2
        
        # Clean up all
        self.file_manager.cleanup_temporary_files()
        
        assert not temp_dir1.exists()
        assert not temp_dir2.exists()
        assert len(self.file_manager.temp_dirs) == 0
    
    def test_cleanup_temporary_files_readonly(self):
        """Test cleanup of read-only temporary files"""
        # Create temporary directory with read-only file
        temp_dir = self.file_manager.create_temp_directory()
        readonly_file = temp_dir / "readonly.txt"
        readonly_file.write_text("readonly content")
        readonly_file.chmod(stat.S_IREAD)
        
        # Cleanup should handle read-only files
        self.file_manager.cleanup_temporary_files(temp_dir)
        
        assert not temp_dir.exists()
    
    def test_safe_delete_file(self):
        """Test safe deletion of file"""
        # Create test file
        test_file = self.temp_dir / "test_file.txt"
        test_file.write_text("test content")
        
        assert test_file.exists()
        
        result = self.file_manager.safe_delete(test_file)
        
        assert result is True
        assert not test_file.exists()
    
    def test_safe_delete_directory(self):
        """Test safe deletion of directory"""
        # Create test directory with files
        test_dir = self.temp_dir / "test_dir"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")
        
        assert test_dir.exists()
        
        result = self.file_manager.safe_delete(test_dir)
        
        assert result is True
        assert not test_dir.exists()
    
    def test_safe_delete_nonexistent(self):
        """Test safe deletion of non-existent path"""
        nonexistent_path = self.temp_dir / "nonexistent"
        
        result = self.file_manager.safe_delete(nonexistent_path)
        
        assert result is True  # Should return True for non-existent paths
    
    def test_safe_delete_readonly(self):
        """Test safe deletion of read-only files"""
        # Create read-only file
        readonly_file = self.temp_dir / "readonly.txt"
        readonly_file.write_text("readonly content")
        readonly_file.chmod(stat.S_IREAD)
        
        result = self.file_manager.safe_delete(readonly_file)
        
        assert result is True
        assert not readonly_file.exists()
    
    def test_get_file_size(self):
        """Test getting file size"""
        # Create test file
        test_file = self.temp_dir / "test_file.txt"
        test_content = "Test content with specific length"
        test_file.write_text(test_content)
        
        size = self.file_manager.get_file_size(test_file)
        
        assert size == len(test_content.encode('utf-8'))
    
    def test_get_file_size_nonexistent(self):
        """Test getting size of non-existent file"""
        nonexistent_file = self.temp_dir / "nonexistent.txt"
        
        with pytest.raises(FileOperationError, match="File not found"):
            self.file_manager.get_file_size(nonexistent_file)
    
    def test_get_directory_size(self):
        """Test getting directory size"""
        # Create test directory with files
        test_dir = self.temp_dir / "test_dir"
        test_dir.mkdir()
        
        file1 = test_dir / "file1.txt"
        file2 = test_dir / "file2.txt"
        content1 = "Content of file 1"
        content2 = "Content of file 2 with more text"
        
        file1.write_text(content1)
        file2.write_text(content2)
        
        size = self.file_manager.get_directory_size(test_dir)
        expected_size = len(content1.encode('utf-8')) + len(content2.encode('utf-8'))
        
        assert size == expected_size
    
    def test_get_directory_size_nonexistent(self):
        """Test getting size of non-existent directory"""
        nonexistent_dir = self.temp_dir / "nonexistent_dir"
        
        with pytest.raises(FileOperationError, match="Directory not found"):
            self.file_manager.get_directory_size(nonexistent_dir)
    
    def test_get_directory_size_not_directory(self):
        """Test getting directory size for a file"""
        # Create test file
        test_file = self.temp_dir / "test_file.txt"
        test_file.write_text("test content")
        
        with pytest.raises(FileOperationError, match="Not a directory"):
            self.file_manager.get_directory_size(test_file)
    
    def test_get_directory_size_nested(self):
        """Test getting directory size with nested structure"""
        # Create nested directory structure
        test_dir = self.temp_dir / "test_dir"
        nested_dir = test_dir / "nested"
        nested_dir.mkdir(parents=True)
        
        # Create files at different levels
        (test_dir / "file1.txt").write_text("content1")
        (nested_dir / "file2.txt").write_text("content2")
        
        size = self.file_manager.get_directory_size(test_dir)
        expected_size = len("content1".encode('utf-8')) + len("content2".encode('utf-8'))
        
        assert size == expected_size
    
    def test_destructor_cleanup(self):
        """Test that destructor cleans up temporary files"""
        # Create file manager with temporary directories
        fm = FileManager()
        temp_dir1 = fm.create_temp_directory()
        temp_dir2 = fm.create_temp_directory()
        
        assert temp_dir1.exists()
        assert temp_dir2.exists()
        
        # Simulate destructor call
        fm.__del__()
        
        # Temporary directories should be cleaned up
        assert not temp_dir1.exists()
        assert not temp_dir2.exists()
    
    def test_create_directory_with_permissions(self):
        """Test directory creation with proper permissions"""
        test_dir = self.temp_dir / "perm_test"
        
        self.file_manager._create_directory_with_permissions(test_dir)
        
        assert test_dir.exists()
        assert test_dir.is_dir()
        
        # Check permissions (on Unix-like systems)
        if os.name != 'nt':  # Not Windows
            mode = test_dir.stat().st_mode
            assert mode & 0o755 == 0o755
    
    def test_remove_readonly_and_delete_file(self):
        """Test removing read-only permissions and deleting file"""
        # Create read-only file
        readonly_file = self.temp_dir / "readonly.txt"
        readonly_file.write_text("readonly content")
        readonly_file.chmod(stat.S_IREAD)
        
        self.file_manager._remove_readonly_and_delete(readonly_file)
        
        assert not readonly_file.exists()
    
    def test_remove_readonly_and_delete_directory(self):
        """Test removing read-only permissions and deleting directory"""
        # Create directory with read-only files
        readonly_dir = self.temp_dir / "readonly_dir"
        readonly_dir.mkdir()
        readonly_file = readonly_dir / "readonly.txt"
        readonly_file.write_text("readonly content")
        readonly_file.chmod(stat.S_IREAD)
        
        self.file_manager._remove_readonly_and_delete(readonly_dir)
        
        assert not readonly_dir.exists()


class TestFileManagerIntegration:
    """Integration tests for FileManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.file_manager = FileManager(base_path=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow(self):
        """Test complete file management workflow"""
        # 1. Create directory structure
        structure = {
            "project": {
                "data": "raw_data",
                "results": "processed_data",
                "temp": "temporary_files"
            }
        }
        
        created_paths = self.file_manager.create_directory_structure(structure)
        
        # 2. Create and copy files
        source_file = self.temp_dir / "source_data.txt"
        source_content = "Important research data"
        source_file.write_text(source_content)
        
        dest_file = created_paths["project/data"] / "copied_data.txt"
        copy_success = self.file_manager.copy_with_validation(source_file, dest_file)
        
        assert copy_success
        assert dest_file.read_text() == source_content
        
        # 3. Create temporary files
        temp_dir = self.file_manager.create_temp_directory(prefix="analysis_")
        temp_file = temp_dir / "temp_results.txt"
        temp_file.write_text("Temporary analysis results")
        
        # 4. Check file sizes
        source_size = self.file_manager.get_file_size(source_file)
        dest_size = self.file_manager.get_file_size(dest_file)
        project_size = self.file_manager.get_directory_size(created_paths["project"])
        
        assert source_size == dest_size
        assert project_size >= dest_size
        
        # 5. Clean up temporary files
        self.file_manager.cleanup_temporary_files()
        
        assert not temp_dir.exists()
        assert len(self.file_manager.temp_dirs) == 0
        
        # 6. Verify main files still exist
        assert dest_file.exists()
        assert created_paths["project"].exists()
    
    def test_error_recovery_workflow(self):
        """Test error handling and recovery in file operations"""
        # Test with invalid paths and permissions
        
        # 1. Try to copy non-existent file
        with pytest.raises(FileOperationError):
            self.file_manager.copy_with_validation(
                self.temp_dir / "nonexistent.txt",
                self.temp_dir / "dest.txt"
            )
        
        # 2. Try to get size of non-existent file
        with pytest.raises(FileOperationError):
            self.file_manager.get_file_size(self.temp_dir / "nonexistent.txt")
        
        # 3. Try to create directory structure without base path
        fm_no_base = FileManager()
        with pytest.raises(ValueError):
            fm_no_base.create_directory_structure({"test": "dir"})
        
        # 4. Verify file manager is still functional after errors
        test_file = self.temp_dir / "recovery_test.txt"
        test_file.write_text("Recovery test")
        
        size = self.file_manager.get_file_size(test_file)
        assert size > 0