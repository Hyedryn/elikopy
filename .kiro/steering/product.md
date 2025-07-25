# ElikoPy Product Overview

ElikoPy is a Python library for diffusion MRI processing and microstructural analysis. The library is designed to work with qsiprep preprocessing outputs and focuses on BIDS-compliant data handling and derivative storage.

## Core Purpose
- Simplify diffusion imaging processing for microstructural analysis
- Provide BIDS-compliant data access and derivative storage
- Support HPC/SLURM environments for large-scale processing
- Integrate with qsiprep preprocessing outputs rather than implementing custom preprocessing

## Key Features
- **Microstructural Modeling**: DTI, NODDI, CSD, MSMT-CSD, and microstructure fingerprinting
- **Tractography**: Streamline generation, SIFT filtering, and connectivity analysis
- **BIDS Compliance**: Full support for BIDS derivatives specification
- **HPC Integration**: Native SLURM job scheduling and parallel processing
- **QSIPrep Integration**: Direct support for qsiprep preprocessing outputs

## Target Users
- Neuroimaging researchers processing diffusion MRI data
- Research groups requiring standardized, reproducible diffusion analysis pipelines
- HPC users needing scalable diffusion processing workflows

## Current Status
The library is undergoing comprehensive refactoring to improve modularity, BIDS compliance, and maintainability while preserving existing functionality.