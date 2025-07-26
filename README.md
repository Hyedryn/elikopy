# ElikoPy

[![Documentation Status](https://readthedocs.org/projects/elikopy/badge/?version=latest)](https://elikopy.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/elikopy?label=pypi%20package)](https://pypi.org/project/elikopy//) ![GitHub repo size](https://img.shields.io/github/repo-size/Hyedryn/elikopy) [![DOI](https://zenodo.org/badge/296056994.svg)](https://zenodo.org/doi/10.5281/zenodo.10514465)

ElikoPy is a Python library designed to simplify the processing of diffusion MRI data for microstructural analysis. It provides BIDS-compliant data handling, integrates with qsiprep preprocessing outputs, and supports HPC environments for large-scale processing.

## Key Features

- **Microstructural Modeling**: DTI, NODDI, CSD, MSMT-CSD, and microstructure fingerprinting
- **Tractography**: Streamline generation, SIFT filtering, and connectivity analysis  
- **BIDS Compliance**: Full support for BIDS derivatives specification
- **HPC Integration**: Native SLURM job scheduling and parallel processing
- **QSIPrep Integration**: Direct support for qsiprep preprocessing outputs

## Installation

### From PyPI (Recommended)

```bash
pip install elikopy
```

### From Source

```bash
git clone https://github.com/Hyedryn/elikopy.git
cd elikopy
pip install -e .
```

### Dependencies

ElikoPy requires Python ≥3.8 and depends on several neuroimaging libraries:

- **Core**: numpy, scipy, scikit-learn, scikit-image
- **Neuroimaging**: dipy, nibabel, pybids
- **External Tools**: mrtrix3 (optional, for advanced tractography)

## Quick Start

### Basic Study Setup

```python
import elikopy
from pathlib import Path

# Initialize study with BIDS dataset
study = elikopy.ElikopyStudy(
    bids_path="/path/to/bids/dataset",
    derivatives_path="/path/to/derivatives",
    study_name="my_diffusion_study"
)

# Load subjects
study.load_subjects(subject_list=["sub-01", "sub-02"])

# Configure processing
config = elikopy.ElikopyConfig()
config.dti.fit_method = "WLS"
config.noddi.enabled = True
config.tracking.algorithm = "probabilistic"

# Set up processor
processor = elikopy.ElikopyProcessor(study, config)
```

### DTI Processing

```python
# Configure DTI processing
config = elikopy.ElikopyConfig()
config.dti.fit_method = "WLS"  # Weighted Least Squares
config.dti.compute_metrics = ["FA", "MD", "AD", "RD"]
config.dti.mask_threshold = 0.2

# Process DTI for all subjects
results = processor.process_dti(
    subjects=["sub-01", "sub-02"],
    sessions=["ses-01"]
)

# Access results
for subject_id, result in results.items():
    print(f"DTI processing for {subject_id}: {result.status}")
    if result.success:
        print(f"  FA map: {result.outputs['FA']}")
        print(f"  MD map: {result.outputs['MD']}")
```

### NODDI Processing

```python
# Configure NODDI processing
config = elikopy.ElikopyConfig()
config.noddi.enabled = True
config.noddi.model_type = "NODDI_WATSON"
config.noddi.parallel_diffusivity = 1.7e-3
config.noddi.isotropic_diffusivity = 3.0e-3

# Process NODDI
results = processor.process_noddi(
    subjects=["sub-01"],
    sessions=["ses-01"]
)

# Access NODDI metrics
for subject_id, result in results.items():
    if result.success:
        print(f"NODDI results for {subject_id}:")
        print(f"  ICVF: {result.outputs['ICVF']}")
        print(f"  ISOVF: {result.outputs['ISOVF']}")
        print(f"  OD: {result.outputs['OD']}")
```

### Tractography and Connectivity

```python
# Configure tractography
config = elikopy.ElikopyConfig()
config.tracking.algorithm = "probabilistic"
config.tracking.n_streamlines = 1000000
config.tracking.step_size = 0.5
config.tracking.max_angle = 30

# Configure connectivity analysis
config.connectivity.atlas_name = "schaefer_400"
config.connectivity.weighting = "count"
config.connectivity.output_formats = ["csv", "json"]

# Process tractography
tracking_results = processor.process_tracking(
    subjects=["sub-01"],
    sessions=["ses-01"]
)

# Process connectivity
connectivity_results = processor.process_connectivity(
    subjects=["sub-01"],
    sessions=["ses-01"],
    atlas_name="schaefer_400"
)

# Access connectivity matrix
for subject_id, result in connectivity_results.items():
    if result.success:
        matrix = result.outputs['connectivity_matrix']
        print(f"Connectivity matrix shape: {matrix.shape}")
```

### Working with BIDS Data

```python
# Access BIDS data directly
bids_handler = study.bids_handler

# Get DWI files for a subject
dwi_files = bids_handler.get_dwi_files("sub-01", "ses-01")
for dwi_file in dwi_files:
    print(f"DWI: {dwi_file.path}")
    print(f"  b-values: {dwi_file.bval_path}")
    print(f"  b-vectors: {dwi_file.bvec_path}")

# Get anatomical files
anat_files = bids_handler.get_anatomical_files("sub-01", "ses-01")
for anat_file in anat_files:
    print(f"Anatomical: {anat_file.path}")

# Validate BIDS dataset
validation_result = bids_handler.validate_dataset()
if validation_result.is_valid:
    print("BIDS dataset is valid")
else:
    print("BIDS validation errors:")
    for error in validation_result.errors:
        print(f"  - {error}")
```

### HPC/SLURM Integration

```python
# Configure for HPC environment
config = elikopy.ElikopyConfig()
config.hpc.enabled = True
config.hpc.scheduler = "slurm"
config.hpc.partition = "gpu"
config.hpc.time_limit = "24:00:00"
config.hpc.memory = "32GB"
config.hpc.cpus_per_task = 8

# Submit jobs to SLURM
job_ids = processor.submit_jobs(
    processing_steps=["dti", "noddi", "tracking"],
    subjects=["sub-01", "sub-02", "sub-03"],
    sessions=["ses-01"]
)

# Monitor job status
for job_id in job_ids:
    status = processor.get_job_status(job_id)
    print(f"Job {job_id}: {status}")
```

### Configuration Management

```python
# Load configuration from file
config = elikopy.ElikopyConfig.from_file("config.yaml")

# Create configuration with custom settings
config = elikopy.ElikopyConfig(
    dti=elikopy.DTIConfig(
        fit_method="WLS",
        compute_metrics=["FA", "MD", "AD", "RD"]
    ),
    noddi=elikopy.NODDIConfig(
        enabled=True,
        model_type="NODDI_WATSON"
    ),
    tracking=elikopy.TrackingConfig(
        algorithm="probabilistic",
        n_streamlines=500000
    )
)

# Save configuration
config.save("my_config.yaml")

# Validate configuration
validation_errors = config.validate()
if validation_errors:
    print("Configuration errors:")
    for error in validation_errors:
        print(f"  - {error}")
```

### Running a Complete Study from YAML Config

The most efficient way to run ElikoPy is using a YAML configuration file that defines all processing parameters:

#### 1. Create a comprehensive config file

```yaml
# study_config.yaml
study:
  bids_path: "/path/to/bids/dataset"
  derivatives_path: "/path/to/derivatives"
  study_name: "my_diffusion_study"
  subjects: ["sub-01", "sub-02", "sub-03"]
  sessions: ["ses-01"]

processing:
  steps: ["dti", "noddi", "tracking", "connectivity"]
  
dti:
  fit_method: "WLS"
  mask_threshold: 0.2
  compute_metrics: ["FA", "MD", "AD", "RD"]
  
noddi:
  enabled: true
  model_type: "NODDI_WATSON"
  parallel_diffusivity: 1.7e-3
  isotropic_diffusivity: 3.0e-3
  
tracking:
  algorithm: "probabilistic"
  n_streamlines: 1000000
  step_size: 0.5
  max_angle: 30
  
connectivity:
  atlas_name: "schaefer_400"
  weighting: "count"
  output_formats: ["csv", "json"]
  
hpc:
  enabled: true
  scheduler: "slurm"
  partition: "compute"
  time_limit: "24:00:00"
  memory: "32GB"
  cpus_per_task: 8
```

#### 2. Run the complete study

```python
import elikopy

def run_study_from_config(config_path):
    """Run complete diffusion study from YAML configuration"""
    
    # Load configuration
    config = elikopy.ElikopyConfig.from_file(config_path)
    
    # Initialize study
    study = elikopy.ElikopyStudy(
        bids_path=config.study.bids_path,
        derivatives_path=config.study.derivatives_path,
        study_name=config.study.study_name
    )
    
    # Load subjects
    study.load_subjects(
        subject_list=config.study.subjects,
        sessions=config.study.sessions
    )
    
    # Create processor
    processor = elikopy.ElikopyProcessor(study, config)
    
    # Run processing steps
    all_results = {}
    
    for step in config.processing.steps:
        print(f"Running {step} processing...")
        
        if step == "dti":
            results = processor.process_dti(
                subjects=config.study.subjects,
                sessions=config.study.sessions
            )
            all_results['dti'] = results
            
        elif step == "noddi" and config.noddi.enabled:
            results = processor.process_noddi(
                subjects=config.study.subjects,
                sessions=config.study.sessions
            )
            all_results['noddi'] = results
            
        elif step == "tracking":
            results = processor.process_tracking(
                subjects=config.study.subjects,
                sessions=config.study.sessions
            )
            all_results['tracking'] = results
            
        elif step == "connectivity":
            results = processor.process_connectivity(
                subjects=config.study.subjects,
                sessions=config.study.sessions,
                atlas_name=config.connectivity.atlas_name
            )
            all_results['connectivity'] = results
    
    return all_results

# Run the study
results = run_study_from_config("study_config.yaml")

# Print summary
for step, step_results in results.items():
    successful = sum(1 for r in step_results.values() if r.success)
    total = len(step_results)
    print(f"{step.upper()}: {successful}/{total} subjects processed successfully")
```

#### 3. Command-line execution script

Create a simple script to run studies from the command line:

```python
#!/usr/bin/env python3
# run_elikopy_study.py

import argparse
import sys
from pathlib import Path
import elikopy

def main():
    parser = argparse.ArgumentParser(description="Run ElikoPy diffusion processing study")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without processing")
    parser.add_argument("--subjects", nargs="+", help="Override subjects list")
    parser.add_argument("--sessions", nargs="+", help="Override sessions list")
    
    args = parser.parse_args()
    
    # Load and validate configuration
    try:
        config = elikopy.ElikopyConfig.from_file(args.config)
        validation_errors = config.validate()
        
        if validation_errors:
            print("Configuration validation errors:")
            for error in validation_errors:
                print(f"  - {error}")
            sys.exit(1)
            
        print(f"Configuration loaded successfully from {args.config}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Override subjects/sessions if provided
    if args.subjects:
        config.study.subjects = args.subjects
    if args.sessions:
        config.study.sessions = args.sessions
    
    if args.dry_run:
        print("Dry run - configuration is valid")
        print(f"Would process {len(config.study.subjects)} subjects")
        print(f"Processing steps: {config.processing.steps}")
        return
    
    # Run the study
    try:
        results = run_study_from_config(args.config)
        
        # Print final summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        
        for step, step_results in results.items():
            successful = sum(1 for r in step_results.values() if r.success)
            total = len(step_results)
            print(f"{step.upper()}: {successful}/{total} subjects successful")
            
            # Show failed subjects
            failed = [subj for subj, r in step_results.items() if not r.success]
            if failed:
                print(f"  Failed subjects: {', '.join(failed)}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

#### 4. Usage examples

```bash
# Run complete study
python run_elikopy_study.py study_config.yaml

# Validate configuration without processing
python run_elikopy_study.py study_config.yaml --dry-run

# Override subjects list
python run_elikopy_study.py study_config.yaml --subjects sub-01 sub-02

# Override both subjects and sessions
python run_elikopy_study.py study_config.yaml --subjects sub-01 --sessions ses-01 ses-02
```

This approach provides:
- **Reproducible processing** with version-controlled configs
- **Easy parameter tuning** by editing YAML files
- **Batch processing** of multiple subjects
- **HPC integration** with SLURM job submission
- **Error handling** and progress reporting

### Advanced Usage: Custom Processing Pipeline

```python
# Create custom processing pipeline
class CustomPipeline:
    def __init__(self, study, config):
        self.study = study
        self.config = config
        self.processor = elikopy.ElikopyProcessor(study, config)
    
    def run_full_pipeline(self, subjects, sessions):
        """Run complete diffusion processing pipeline"""
        results = {}
        
        # Step 1: DTI processing
        print("Processing DTI...")
        dti_results = self.processor.process_dti(subjects, sessions)
        results['dti'] = dti_results
        
        # Step 2: NODDI processing (if enabled)
        if self.config.noddi.enabled:
            print("Processing NODDI...")
            noddi_results = self.processor.process_noddi(subjects, sessions)
            results['noddi'] = noddi_results
        
        # Step 3: Tractography
        print("Processing tractography...")
        tracking_results = self.processor.process_tracking(subjects, sessions)
        results['tracking'] = tracking_results
        
        # Step 4: Connectivity analysis
        print("Processing connectivity...")
        connectivity_results = self.processor.process_connectivity(
            subjects, sessions, atlas_name=self.config.connectivity.atlas_name
        )
        results['connectivity'] = connectivity_results
        
        return results

# Use custom pipeline
pipeline = CustomPipeline(study, config)
all_results = pipeline.run_full_pipeline(
    subjects=["sub-01", "sub-02"],
    sessions=["ses-01"]
)
```

## Configuration Files

ElikoPy supports YAML configuration files for reproducible processing:

```yaml
# config.yaml
dti:
  fit_method: "WLS"
  mask_threshold: 0.2
  compute_metrics: ["FA", "MD", "AD", "RD"]

noddi:
  enabled: true
  model_type: "NODDI_WATSON"
  parallel_diffusivity: 1.7e-3
  isotropic_diffusivity: 3.0e-3

tracking:
  algorithm: "probabilistic"
  n_streamlines: 1000000
  step_size: 0.5
  max_angle: 30

connectivity:
  atlas_name: "schaefer_400"
  weighting: "count"
  output_formats: ["csv", "json"]

hpc:
  enabled: false
  scheduler: "slurm"
  partition: "compute"
  time_limit: "12:00:00"
  memory: "16GB"
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_dti_processor.py

# Run with coverage
python -m pytest tests/ --cov=elikopy
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Publications & Citations

If you use ElikoPy in your research, please cite it using the package DOI [![DOI](https://zenodo.org/badge/296056994.svg)](https://zenodo.org/doi/10.5281/zenodo.10514465).

## License

ElikoPy is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE) for details.
