"""
ElikopyConfig class - Configuration management

This module provides comprehensive configuration management for ElikoPy,
including validation, default templates, and support for YAML/JSON files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import warnings


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails"""
    pass

@dataclass
class DTIConfig:
    """Configuration for DTI processing
    
    Attributes:
        fit_method: Tensor fitting method (WLS, OLS, NLLS)
        mask_threshold: Threshold for brain mask creation
        compute_metrics: List of DTI metrics to compute
    """
    fit_method: str = "WLS"  # WLS, OLS, NLLS
    mask_threshold: float = 0.2
    compute_metrics: List[str] = field(default_factory=lambda: ["FA", "MD", "AD", "RD"])
    
    def validate(self) -> List[str]:
        """Validate DTI configuration
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate fit method
        valid_methods = ["WLS", "OLS", "NLLS"]
        if self.fit_method not in valid_methods:
            errors.append(f"DTI fit_method must be one of {valid_methods}, got '{self.fit_method}'")
        
        # Validate mask threshold
        if not 0.0 <= self.mask_threshold <= 1.0:
            errors.append(f"DTI mask_threshold must be between 0.0 and 1.0, got {self.mask_threshold}")
        
        # Validate metrics
        valid_metrics = ["FA", "MD", "AD", "RD", "MO", "RGB"]
        invalid_metrics = [m for m in self.compute_metrics if m not in valid_metrics]
        if invalid_metrics:
            errors.append(f"DTI compute_metrics contains invalid metrics: {invalid_metrics}. Valid metrics: {valid_metrics}")
        
        return errors


@dataclass
class NODDIConfig:
    """Configuration for NODDI processing
    
    Attributes:
        fit_method: NODDI fitting method (amico, noddi-python)
        mask_threshold: Threshold for brain mask creation
        compute_metrics: List of NODDI metrics to compute
    """
    fit_method: str = "amico"  # amico, noddi-python
    mask_threshold: float = 0.2
    compute_metrics: List[str] = field(default_factory=lambda: ["ICVF", "ODI", "ISOVF"])
    
    def validate(self) -> List[str]:
        """Validate NODDI configuration
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate fit method
        valid_methods = ["amico", "noddi-python"]
        if self.fit_method not in valid_methods:
            errors.append(f"NODDI fit_method must be one of {valid_methods}, got '{self.fit_method}'")
        
        # Validate mask threshold
        if not 0.0 <= self.mask_threshold <= 1.0:
            errors.append(f"NODDI mask_threshold must be between 0.0 and 1.0, got {self.mask_threshold}")
        
        # Validate metrics
        valid_metrics = ["ICVF", "ODI", "ISOVF", "FISO"]
        invalid_metrics = [m for m in self.compute_metrics if m not in valid_metrics]
        if invalid_metrics:
            errors.append(f"NODDI compute_metrics contains invalid metrics: {invalid_metrics}. Valid metrics: {valid_metrics}")
        
        return errors


@dataclass
class CSDConfig:
    """Configuration for CSD processing
    
    Attributes:
        response_method: Response function estimation method
        sh_order: Spherical harmonics order
        mask_threshold: Threshold for brain mask creation
        multi_shell: Whether to use multi-shell data
    """
    response_method: str = "tournier"  # tournier, tax, dhollander
    sh_order: int = 8
    mask_threshold: float = 0.2
    multi_shell: bool = False
    
    def validate(self) -> List[str]:
        """Validate CSD configuration
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate response method
        valid_methods = ["tournier", "tax", "dhollander"]
        if self.response_method not in valid_methods:
            errors.append(f"CSD response_method must be one of {valid_methods}, got '{self.response_method}'")
        
        # Validate spherical harmonics order
        if self.sh_order < 2 or self.sh_order > 12 or self.sh_order % 2 != 0:
            errors.append(f"CSD sh_order must be an even number between 2 and 12, got {self.sh_order}")
        
        # Validate mask threshold
        if not 0.0 <= self.mask_threshold <= 1.0:
            errors.append(f"CSD mask_threshold must be between 0.0 and 1.0, got {self.mask_threshold}")
        
        return errors


@dataclass
class MSMTCSDConfig:
    """Configuration for MSMT-CSD processing
    
    Attributes:
        response_method: Multi-tissue response function estimation method
        sh_order: Spherical harmonics order
        mask_threshold: Threshold for brain mask creation
    """
    response_method: str = "dhollander"  # dhollander, msmt-5tt
    sh_order: int = 8
    mask_threshold: float = 0.2
    
    def validate(self) -> List[str]:
        """Validate MSMT-CSD configuration
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate response method
        valid_methods = ["dhollander", "msmt-5tt"]
        if self.response_method not in valid_methods:
            errors.append(f"MSMT-CSD response_method must be one of {valid_methods}, got '{self.response_method}'")
        
        # Validate spherical harmonics order
        if self.sh_order < 2 or self.sh_order > 12 or self.sh_order % 2 != 0:
            errors.append(f"MSMT-CSD sh_order must be an even number between 2 and 12, got {self.sh_order}")
        
        # Validate mask threshold
        if not 0.0 <= self.mask_threshold <= 1.0:
            errors.append(f"MSMT-CSD mask_threshold must be between 0.0 and 1.0, got {self.mask_threshold}")
        
        return errors


@dataclass
class TrackingConfig:
    """Configuration for tractography
    
    Attributes:
        algorithm: Tracking algorithm (deterministic, probabilistic)
        step_size: Step size in mm
        max_angle: Maximum turning angle in degrees
        min_length: Minimum streamline length in mm
        max_length: Maximum streamline length in mm
        num_seeds: Number of seeds for tracking
        seed_mask: Seeding mask type
        apply_sift: Whether to apply SIFT filtering
        sift_term_count: Target streamline count after SIFT
    """
    algorithm: str = "deterministic"  # deterministic, probabilistic
    step_size: float = 0.5
    max_angle: float = 30.0
    min_length: float = 20.0
    max_length: float = 200.0
    num_seeds: int = 10000
    seed_mask: str = "wm"  # wm, interface, gmwmi
    apply_sift: bool = True
    sift_term_count: int = 5000
    
    def validate(self) -> List[str]:
        """Validate tracking configuration
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate algorithm
        valid_algorithms = ["deterministic", "probabilistic"]
        if self.algorithm not in valid_algorithms:
            errors.append(f"Tracking algorithm must be one of {valid_algorithms}, got '{self.algorithm}'")
        
        # Validate step size
        if not 0.1 <= self.step_size <= 2.0:
            errors.append(f"Tracking step_size must be between 0.1 and 2.0 mm, got {self.step_size}")
        
        # Validate max angle
        if not 5.0 <= self.max_angle <= 90.0:
            errors.append(f"Tracking max_angle must be between 5.0 and 90.0 degrees, got {self.max_angle}")
        
        # Validate lengths
        if self.min_length <= 0:
            errors.append(f"Tracking min_length must be positive, got {self.min_length}")
        if self.max_length <= self.min_length:
            errors.append(f"Tracking max_length ({self.max_length}) must be greater than min_length ({self.min_length})")
        
        # Validate seed count
        if self.num_seeds <= 0:
            errors.append(f"Tracking num_seeds must be positive, got {self.num_seeds}")
        
        # Validate seed mask
        valid_masks = ["wm", "interface", "gmwmi"]
        if self.seed_mask not in valid_masks:
            errors.append(f"Tracking seed_mask must be one of {valid_masks}, got '{self.seed_mask}'")
        
        # Validate SIFT parameters
        if self.apply_sift and self.sift_term_count <= 0:
            errors.append(f"Tracking sift_term_count must be positive when apply_sift is True, got {self.sift_term_count}")
        
        return errors


@dataclass
class ConnectivityConfig:
    """Configuration for connectivity analysis
    
    Attributes:
        atlas: Atlas to use for connectivity analysis
        measure: Connectivity measure to compute
        normalize: Whether to normalize connectivity values
    """
    atlas: str = "aal"  # aal, desikan, schaefer100, schaefer200, schaefer400
    measure: str = "count"  # count, density, length
    normalize: bool = True
    
    def validate(self) -> List[str]:
        """Validate connectivity configuration
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate atlas
        valid_atlases = ["aal", "desikan", "schaefer100", "schaefer200", "schaefer400"]
        if self.atlas not in valid_atlases:
            errors.append(f"Connectivity atlas must be one of {valid_atlases}, got '{self.atlas}'")
        
        # Validate measure
        valid_measures = ["count", "density", "length", "mean_length"]
        if self.measure not in valid_measures:
            errors.append(f"Connectivity measure must be one of {valid_measures}, got '{self.measure}'")
        
        return errors


@dataclass
class FingerprintingConfig:
    """Configuration for microstructure fingerprinting
    
    Attributes:
        dictionary_path: Path to fingerprinting dictionary
        mask_threshold: Threshold for brain mask creation
        compute_metrics: List of fingerprinting metrics to compute
    """
    dictionary_path: Optional[str] = None
    mask_threshold: float = 0.2
    compute_metrics: List[str] = field(default_factory=lambda: ["fvf", "diameter", "orientation"])
    
    def validate(self) -> List[str]:
        """Validate fingerprinting configuration
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate dictionary path
        if self.dictionary_path is not None:
            dict_path = Path(self.dictionary_path)
            if not dict_path.exists():
                errors.append(f"Fingerprinting dictionary_path does not exist: {self.dictionary_path}")
        
        # Validate mask threshold
        if not 0.0 <= self.mask_threshold <= 1.0:
            errors.append(f"Fingerprinting mask_threshold must be between 0.0 and 1.0, got {self.mask_threshold}")
        
        # Validate metrics
        valid_metrics = ["fvf", "diameter", "orientation", "dispersion", "kappa"]
        invalid_metrics = [m for m in self.compute_metrics if m not in valid_metrics]
        if invalid_metrics:
            errors.append(f"Fingerprinting compute_metrics contains invalid metrics: {invalid_metrics}. Valid metrics: {valid_metrics}")
        
        return errors


@dataclass
class SchedulerConfig:
    """Configuration for job scheduler
    
    Attributes:
        type: Scheduler type (slurm, local)
        cpus_per_task: Number of CPUs per task
        mem_per_cpu: Memory per CPU in GB
        time_limit: Time limit in HH:MM:SS format
        partition: SLURM partition name
        account: SLURM account name
        use_gpu: Whether to use GPU acceleration
        gpu_count: Number of GPUs to request
    """
    type: str = "slurm"  # slurm, local
    cpus_per_task: int = 4
    mem_per_cpu: int = 4  # GB
    time_limit: str = "24:00:00"
    partition: Optional[str] = None
    account: Optional[str] = None
    use_gpu: bool = False
    gpu_count: int = 0
    
    def validate(self) -> List[str]:
        """Validate scheduler configuration
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate scheduler type
        valid_types = ["slurm", "local"]
        if self.type not in valid_types:
            errors.append(f"Scheduler type must be one of {valid_types}, got '{self.type}'")
        
        # Validate CPU count
        if self.cpus_per_task <= 0:
            errors.append(f"Scheduler cpus_per_task must be positive, got {self.cpus_per_task}")
        
        # Validate memory
        if self.mem_per_cpu <= 0:
            errors.append(f"Scheduler mem_per_cpu must be positive, got {self.mem_per_cpu}")
        
        # Validate time limit format
        import re
        time_pattern = r'^\d{1,2}:\d{2}:\d{2}$'
        if not re.match(time_pattern, self.time_limit):
            errors.append(f"Scheduler time_limit must be in HH:MM:SS format, got '{self.time_limit}'")
        
        # Validate GPU settings
        if self.use_gpu and self.gpu_count <= 0:
            errors.append(f"Scheduler gpu_count must be positive when use_gpu is True, got {self.gpu_count}")
        
        return errors


@dataclass
class OutputConfig:
    """Configuration for output settings
    
    Attributes:
        derivatives_name: Name for derivatives directory
        save_intermediates: Whether to save intermediate processing files
        compress_outputs: Whether to compress output files
        verbose_logging: Whether to enable verbose logging
    """
    derivatives_name: str = "elikopy"
    save_intermediates: bool = False
    compress_outputs: bool = True
    verbose_logging: bool = False
    
    def validate(self) -> List[str]:
        """Validate output configuration
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate derivatives name
        if not self.derivatives_name or not self.derivatives_name.strip():
            errors.append("Output derivatives_name cannot be empty")
        
        # Check for valid directory name characters
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', self.derivatives_name):
            errors.append(f"Output derivatives_name contains invalid characters: '{self.derivatives_name}'. Use only letters, numbers, underscores, and hyphens.")
        
        return errors


@dataclass
class ElikopyConfig:
    """Main configuration class for ElikoPy"""
    study_name: str = "elikopy_study"
    dti: DTIConfig = field(default_factory=DTIConfig)
    noddi: NODDIConfig = field(default_factory=NODDIConfig)
    csd: CSDConfig = field(default_factory=CSDConfig)
    msmt_csd: MSMTCSDConfig = field(default_factory=MSMTCSDConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    fingerprinting: FingerprintingConfig = field(default_factory=FingerprintingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'ElikopyConfig':
        """Load configuration from file
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Returns:
            ElikopyConfig object
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If file format is unsupported or invalid
            ConfigValidationError: If configuration validation fails
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            # Load configuration based on file extension
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                except ImportError:
                    raise ImportError("PyYAML is required to load YAML configuration files. Install with: pip install PyYAML")
                
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
                    
            elif config_file.suffix.lower() == '.json':
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_file.suffix}. Supported formats: .yaml, .yml, .json")
            
            if config_dict is None:
                config_dict = {}
                
        except (yaml.YAMLError if 'yaml' in locals() else Exception, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse configuration file {config_file}: {e}")
        
        # Create config object
        config = cls.from_dict(config_dict)
        
        # Validate configuration
        validation_errors = config.validate()
        if validation_errors:
            error_msg = f"Configuration validation failed for {config_file}:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise ConfigValidationError(error_msg)
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ElikopyConfig':
        """Create config from dictionary
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ElikopyConfig object
        """
        # Create base config
        config = cls()
        
        # Update with provided values
        config.update(config_dict)
        
        return config
    
    def to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Raises:
            ValueError: If file format is unsupported
            ConfigValidationError: If configuration validation fails
        """
        config_file = Path(config_path)
        
        # Validate configuration before saving
        validation_errors = self.validate()
        if validation_errors:
            error_msg = f"Cannot save invalid configuration:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise ConfigValidationError(error_msg)
        
        # Convert to dictionary
        config_dict = self.to_dict()
        
        # Create parent directory if it doesn't exist
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save based on file extension
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                except ImportError:
                    raise ImportError("PyYAML is required to save YAML configuration files. Install with: pip install PyYAML")
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)
                    
            elif config_file.suffix.lower() == '.json':
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, sort_keys=False)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_file.suffix}. Supported formats: .yaml, .yml, .json")
                
        except (yaml.YAMLError if 'yaml' in locals() else Exception, OSError) as e:
            raise ValueError(f"Failed to save configuration file {config_file}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary
        
        Returns:
            Configuration dictionary
        """
        import dataclasses
        import json
        
        # Convert to dictionary using dataclasses
        config_dict = dataclasses.asdict(self)
        
        return config_dict
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with values from dictionary
        
        Args:
            config_dict: Configuration dictionary
        """
        # Update top-level attributes
        for key, value in config_dict.items():
            if key in self.__dataclass_fields__:
                if isinstance(value, dict):
                    # Update nested dataclass
                    current_value = getattr(self, key)
                    if hasattr(current_value, '__dataclass_fields__'):
                        for nested_key, nested_value in value.items():
                            if nested_key in current_value.__dataclass_fields__:
                                setattr(current_value, nested_key, nested_value)
                    else:
                        setattr(self, key, value)
                else:
                    # Update simple attribute
                    setattr(self, key, value)
    
    def validate(self) -> List[str]:
        """Validate configuration
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate study name
        if not self.study_name or not self.study_name.strip():
            errors.append("study_name cannot be empty")
        
        # Validate each configuration section
        errors.extend(self.dti.validate())
        errors.extend(self.noddi.validate())
        errors.extend(self.csd.validate())
        errors.extend(self.msmt_csd.validate())
        errors.extend(self.tracking.validate())
        errors.extend(self.connectivity.validate())
        errors.extend(self.fingerprinting.validate())
        errors.extend(self.scheduler.validate())
        errors.extend(self.output.validate())
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid
        
        Returns:
            True if configuration is valid
        """
        return len(self.validate()) == 0
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        # Create a new instance with default values
        default_config = ElikopyConfig()
        
        # Convert to dictionary
        return default_config.to_dict()
    
    @classmethod
    def create_default_template(cls, template_path: Union[str, Path], 
                               format: str = "yaml") -> None:
        """Create a default configuration template file
        
        Args:
            template_path: Path where to save the template
            format: File format ('yaml' or 'json')
            
        Raises:
            ValueError: If format is unsupported
        """
        if format.lower() not in ['yaml', 'yml', 'json']:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")
        
        # Create default configuration
        default_config = cls()
        
        # Determine file extension
        template_file = Path(template_path)
        if not template_file.suffix:
            if format.lower() == 'json':
                template_file = template_file.with_suffix('.json')
            else:
                template_file = template_file.with_suffix('.yaml')
        
        # Save template
        default_config.to_file(template_file)
    
    @classmethod
    def create_minimal_template(cls, template_path: Union[str, Path], 
                               processing_types: List[str] = None,
                               format: str = "yaml") -> None:
        """Create a minimal configuration template with only specified processing types
        
        Args:
            template_path: Path where to save the template
            processing_types: List of processing types to include (e.g., ['dti', 'noddi'])
            format: File format ('yaml' or 'json')
            
        Raises:
            ValueError: If format is unsupported or processing type is invalid
        """
        if format.lower() not in ['yaml', 'yml', 'json']:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")
        
        if processing_types is None:
            processing_types = ['dti']
        
        # Validate processing types
        valid_types = ['dti', 'noddi', 'csd', 'msmt_csd',
                      'tracking', 'connectivity', 'fingerprinting']
        invalid_types = [t for t in processing_types if t not in valid_types]
        if invalid_types:
            raise ValueError(f"Invalid processing types: {invalid_types}. Valid types: {valid_types}")
        
        # Create minimal configuration
        config_dict = {
            'study_name': 'my_study',
            'scheduler': {
                'type': 'local',
                'cpus_per_task': 4
            },
            'output': {
                'derivatives_name': 'elikopy'
            }
        }
        
        # Add requested processing configurations
        default_config = cls()
        for proc_type in processing_types:
            if hasattr(default_config, proc_type):
                section_config = getattr(default_config, proc_type)
                config_dict[proc_type] = section_config.__dict__.copy()
        
        # Determine file extension
        template_file = Path(template_path)
        if not template_file.suffix:
            if format.lower() == 'json':
                template_file = template_file.with_suffix('.json')
            else:
                template_file = template_file.with_suffix('.yaml')
        
        # Create parent directory if needed
        template_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save template
        try:
            if template_file.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                except ImportError:
                    raise ImportError("PyYAML is required to save YAML templates. Install with: pip install PyYAML")
                
                with open(template_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)
                    
            elif template_file.suffix.lower() == '.json':
                with open(template_file, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, sort_keys=False)
                    
        except (yaml.YAMLError if 'yaml' in locals() else Exception, OSError) as e:
            raise ValueError(f"Failed to save template file {template_file}: {e}")
    
    def print_validation_report(self) -> None:
        """Print a detailed validation report"""
        errors = self.validate()
        
        if not errors:
            print("✓ Configuration is valid")
        else:
            print(f"✗ Configuration has {len(errors)} error(s):")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            'study_name': self.study_name,
            'enabled_processing': {
                'dti': True,  # Always available
                'noddi': True,  # Always available
                'csd': True,  # Always available
                'msmt_csd': True,  # Always available
                'tracking': True,  # Always available
                'connectivity': True,  # Always available
                'fingerprinting': self.fingerprinting.dictionary_path is not None
            },
            'scheduler_type': self.scheduler.type,
            'output_name': self.output.derivatives_name,
            'validation_status': 'valid' if self.is_valid() else 'invalid'
        }