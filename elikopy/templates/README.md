# ElikoPy Configuration Templates

This directory contains pre-configured templates for common ElikoPy use cases.

## Available Templates

### 1. `default_config.yaml`
Complete configuration with all processing options and their default values. Use this as a starting point for comprehensive diffusion MRI analysis.

**Includes:**
- All preprocessing options
- DTI, NODDI, CSD, and MSMT-CSD processing
- Tractography and connectivity analysis
- Microstructure fingerprinting (requires dictionary)
- SLURM scheduler configuration

### 2. `minimal_dti_config.yaml`
Minimal configuration for DTI-only analysis with local processing.

**Includes:**
- DTI processing with standard metrics (FA, MD, AD, RD)
- Local scheduler (no HPC required)
- Basic output settings

### 3. `hpc_config.yaml`
Configuration optimized for high-performance computing environments.

**Includes:**
- Multiple processing types (DTI, NODDI, tractography, connectivity)
- SLURM scheduler with increased resources
- High-resolution connectivity analysis
- Verbose logging and intermediate file saving

## Using Templates

### Method 1: Copy and Modify
```bash
cp elikopy/templates/minimal_dti_config.yaml my_config.yaml
# Edit my_config.yaml as needed
```

### Method 2: CLI Tool
```bash
# Create default template
python -m elikopy.cli.config_cli create-template my_config.yaml

# Create minimal template with specific processing types
python -m elikopy.cli.config_cli create-template --minimal --processing dti noddi my_config.yaml

# Validate configuration
python -m elikopy.cli.config_cli validate my_config.yaml --verbose

# Convert between formats
python -m elikopy.cli.config_cli convert my_config.yaml my_config.json
```

### Method 3: Python API
```python
from elikopy.core.config import ElikopyConfig

# Create default configuration
config = ElikopyConfig()

# Create minimal configuration
ElikopyConfig.create_minimal_template(
    "my_config.yaml", 
    processing_types=['dti', 'noddi']
)

# Load and validate configuration
config = ElikopyConfig.from_file("my_config.yaml")
if config.is_valid():
    print("Configuration is valid!")
else:
    config.print_validation_report()
```

## Configuration Sections

### Study Settings
```yaml
study_name: my_study  # Name for your study
```

### Processing Options

#### DTI (Diffusion Tensor Imaging)
```yaml
dti:
  fit_method: WLS          # WLS, OLS, or NLLS
  mask_threshold: 0.2      # Brain mask threshold (0.0-1.0)
  compute_metrics:         # Metrics to compute
    - FA                   # Fractional Anisotropy
    - MD                   # Mean Diffusivity
    - AD                   # Axial Diffusivity
    - RD                   # Radial Diffusivity
```

#### NODDI (Neurite Orientation Dispersion and Density Imaging)
```yaml
noddi:
  fit_method: amico        # amico or noddi-python
  mask_threshold: 0.2      # Brain mask threshold
  compute_metrics:
    - ICVF                 # Intracellular Volume Fraction
    - ODI                  # Orientation Dispersion Index
    - ISOVF                # Isotropic Volume Fraction
```

#### Tractography
```yaml
tracking:
  algorithm: deterministic  # deterministic or probabilistic
  step_size: 0.5           # Step size in mm
  max_angle: 30.0          # Maximum turning angle in degrees
  min_length: 20.0         # Minimum streamline length in mm
  max_length: 200.0        # Maximum streamline length in mm
  num_seeds: 10000         # Number of seeds
  apply_sift: true         # Apply SIFT filtering
```

### Scheduler Configuration

#### Local Processing
```yaml
scheduler:
  type: local
  cpus_per_task: 4
```

#### SLURM/HPC Processing
```yaml
scheduler:
  type: slurm
  cpus_per_task: 8
  mem_per_cpu: 8           # GB per CPU
  time_limit: "48:00:00"   # HH:MM:SS format
  partition: compute       # SLURM partition
  account: my_account      # SLURM account (optional)
  use_gpu: false           # Enable GPU acceleration
```

### Output Settings
```yaml
output:
  derivatives_name: elikopy     # Name for derivatives directory
  save_intermediates: false    # Save intermediate processing files
  compress_outputs: true       # Compress output files
  verbose_logging: false       # Enable verbose logging
```

## Validation

All configurations are automatically validated when loaded. Common validation errors include:

- **Invalid parameter values**: e.g., mask_threshold outside 0.0-1.0 range
- **Unsupported methods**: e.g., invalid fit_method for DTI
- **Missing required files**: e.g., fingerprinting dictionary not found
- **Incompatible settings**: e.g., GPU requested but count is 0

Use the validation tools to check your configuration:

```bash
python -m elikopy.cli.config_cli validate my_config.yaml --verbose
```

## Best Practices

1. **Start with templates**: Use provided templates as starting points
2. **Validate early**: Always validate configurations before processing
3. **Use version control**: Track configuration changes in git
4. **Document changes**: Add comments to explain custom settings
5. **Test locally**: Test configurations with small datasets first
6. **Resource planning**: Adjust scheduler settings based on data size

## Troubleshooting

### Common Issues

1. **YAML parsing errors**: Check indentation and syntax
2. **File not found**: Verify paths are correct and files exist
3. **Permission errors**: Ensure write permissions for output directories
4. **Resource limits**: Adjust memory/CPU requests for your system

### Getting Help

- Check validation messages for specific error details
- Use `--verbose` flag for detailed configuration summaries
- Refer to the main ElikoPy documentation for processing details