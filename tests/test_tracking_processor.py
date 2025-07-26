"""
Unit tests for TrackingProcessor
"""

import numpy as np
from pathlib import Path

from elikopy.processing.tracking import (
    TrackingProcessor, TrackingConfig, StreamlinesResult, SIFTResult
)
from elikopy.core.base import ProcessingStatus


class TestTrackingConfig:
    """Test TrackingConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TrackingConfig()
        assert config.algorithm == "deterministic"
        assert config.step_size == 0.5
        assert config.max_angle == 30.0
        assert config.min_length == 20.0
        assert config.max_length == 200.0
        assert config.num_streamlines == 100000
        assert config.apply_sift is True
        assert config.use_mrtrix is True
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = TrackingConfig(
            algorithm="probabilistic",
            step_size=0.25,
            num_streamlines=50000
        )
        assert config.algorithm == "probabilistic"
        assert config.step_size == 0.25
        assert config.num_streamlines == 50000


class TestTrackingProcessor:
    """Test TrackingProcessor class"""
    
    def test_initialization(self):
        """Test processor initialization"""
        processor = TrackingProcessor()
        assert processor.config is not None
        assert isinstance(processor.config, TrackingConfig)
    
    def test_initialization_with_config(self):
        """Test processor initialization with custom config"""
        config = TrackingConfig(algorithm="probabilistic")
        processor = TrackingProcessor(config)
        assert processor.config.algorithm == "probabilistic"
    
    def test_configure(self):
        """Test processor configuration"""
        processor = TrackingProcessor()
        config_dict = {
            "algorithm": "probabilistic",
            "step_size": 0.25,
            "num_streamlines": 50000
        }
        processor.configure(config_dict)
        
        assert processor.config.algorithm == "probabilistic"
        assert processor.config.step_size == 0.25
        assert processor.config.num_streamlines == 50000
    
    def test_get_default_config(self):
        """Test getting default configuration"""
        processor = TrackingProcessor()
        config = processor.get_default_config()
        
        assert isinstance(config, dict)
        assert "algorithm" in config
        assert "step_size" in config
        assert "num_streamlines" in config
        assert config["algorithm"] == "deterministic"
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config"""
        processor = TrackingProcessor()
        config = {
            "algorithm": "deterministic",
            "step_size": 0.5,
            "max_angle": 30.0,
            "min_length": 10.0,
            "max_length": 100.0
        }
        assert processor.validate_config(config) is True
    
    def test_validate_config_invalid_algorithm(self):
        """Test configuration validation with invalid algorithm"""
        processor = TrackingProcessor()
        config = {"algorithm": "invalid_algorithm"}
        assert processor.validate_config(config) is False
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs"""
        processor = TrackingProcessor()
        
        peaks_data = {
            'peaks': np.random.rand(10, 10, 10, 3, 3),
            'peak_values': np.random.rand(10, 10, 10, 3),
            'peak_indices': np.random.randint(0, 100, (10, 10, 10, 3))
        }
        mask = np.ones((10, 10, 10), dtype=bool)
        affine = np.eye(4)
        
        result = processor.validate_inputs(
            peaks_data=peaks_data,
            mask=mask,
            affine=affine
        )
        assert result is True
    
    def test_validate_inputs_missing_peaks(self):
        """Test input validation with missing peaks data"""
        processor = TrackingProcessor()
        mask = np.ones((10, 10, 10), dtype=bool)
        affine = np.eye(4)
        
        result = processor.validate_inputs(
            peaks_data=None,
            mask=mask,
            affine=affine
        )
        assert result is False
    
    def test_validate_inputs_missing_mask(self):
        """Test input validation with missing mask"""
        processor = TrackingProcessor()
        peaks_data = {
            'peaks': np.random.rand(10, 10, 10, 3, 3),
            'peak_values': np.random.rand(10, 10, 10, 3),
            'peak_indices': np.random.randint(0, 100, (10, 10, 10, 3))
        }
        affine = np.eye(4)
        
        result = processor.validate_inputs(
            peaks_data=peaks_data,
            mask=None,
            affine=affine
        )
        assert result is False


class TestStreamlinesResult:
    """Test StreamlinesResult dataclass"""
    
    def test_creation(self):
        """Test StreamlinesResult creation"""
        streamlines = [np.random.rand(50, 3) for _ in range(10)]
        length_stats = {"mean": 25.0, "std": 5.0}
        
        result = StreamlinesResult(
            streamlines=streamlines,
            streamline_count=10,
            length_stats=length_stats,
            algorithm_used="deterministic"
        )
        
        assert result.streamlines == streamlines
        assert result.streamline_count == 10
        assert result.length_stats == length_stats
        assert result.algorithm_used == "deterministic"


class TestSIFTResult:
    """Test SIFTResult dataclass"""
    
    def test_creation(self):
        """Test SIFTResult creation"""
        streamlines = [np.random.rand(50, 3) for _ in range(5)]
        
        result = SIFTResult(
            streamlines=streamlines,
            streamline_count=5,
            original_count=10,
            reduction_factor=0.5
        )
        
        assert result.streamlines == streamlines
        assert result.streamline_count == 5
        assert result.original_count == 10
        assert result.reduction_factor == 0.5