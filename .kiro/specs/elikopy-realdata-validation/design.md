# Design Document: ElikoPy Real Data Validation

## Overview

The real data validation system is designed to comprehensively test the refactored ElikoPy codebase against actual diffusion MRI datasets. This validation framework will serve as an integration testing suite that verifies the coherence and functionality of all refactored components when processing real neuroimaging data.

The system follows a modular testing approach that mirrors the ElikoPy architecture, testing each layer (data handling, processing algorithms, infrastructure) both independently and in integration scenarios. The validation framework will provide detailed reporting on component functionality, performance comparisons, and identification of any regressions introduced during refactoring.

## Architecture

### Core Components

The validation system is structured around four main validation modules:

1. **Data Validation Module**: Tests BIDS handling, data loading, and validation classes
2. **Processing Validation Module**: Tests all processing algorithms (DTI, NODDI, CSD, etc.)
3. **Infrastructure Validation Module**: Tests configuration, logging, and scheduling components
4. **Integration Validation Module**: Tests end-to-end workflows and cross-component interactions

### Validation Framework Architecture

```
RealDataValidator (Main Controller)
├── DataComponentValidator
│   ├── BIDSHandlerValidator
│   ├── DerivativesValidator
│   └── DataValidatorTester
├── ProcessingComponentValidator
│   ├── DTIProcessorValidator
│   ├── NODDIProcessorValidator
│   ├── CSDProcessorValidator
│   └── ConnectivityValidator
├── InfrastructureComponentValidator
│   ├── ConfigValidator
│   ├── LoggingValidator
│   └── SchedulerValidator
├── IntegrationValidator
│   ├── EndToEndWorkflowValidator
│   └── CrossModuleValidator
└── ValidationReporter
    ├── ComponentReportGenerator
    ├── PerformanceAnalyzer
    └── RegressionDetector
```

### Design Rationale

**Modular Testing Approach**: Each validation module corresponds to a major component of the ElikoPy architecture, allowing for isolated testing and clear identification of issues within specific subsystems.

**Real Data Focus**: Unlike unit tests that use synthetic data, this validation system specifically uses real diffusion MRI datasets to ensure the refactored code handles real-world data complexities and edge cases.

**Comprehensive Reporting**: The validation system generates detailed reports that not only identify failures but also provide performance metrics and regression analysis to demonstrate the success of the refactoring effort.

## Components and Interfaces

### RealDataValidator (Main Controller)

```python
class RealDataValidator:
    def __init__(self, test_data_path: str, config_path: str)
    def run_full_validation(self) -> ValidationReport
    def run_component_validation(self, component: str) -> ComponentReport
    def run_integration_validation(self) -> IntegrationReport
```

**Purpose**: Orchestrates the entire validation process, coordinates between different validation modules, and manages test data and configuration.

### DataComponentValidator

```python
class DataComponentValidator:
    def validate_bids_handler(self, dataset_path: str) -> BIDSValidationResult
    def validate_derivatives_handling(self, derivatives_path: str) -> DerivativesValidationResult
    def validate_data_integrity_checks(self, dataset_path: str) -> DataValidationResult
```

**Purpose**: Tests all data-related classes including BIDSHandler, derivatives management, and data validation utilities with real BIDS datasets.

### ProcessingComponentValidator

```python
class ProcessingComponentValidator:
    def validate_dti_processing(self, dwi_data: str) -> DTIValidationResult
    def validate_noddi_processing(self, dwi_data: str) -> NODDIValidationResult
    def validate_csd_processing(self, dwi_data: str) -> CSDValidationResult
    def validate_connectivity_analysis(self, processed_data: str) -> ConnectivityValidationResult
```

**Purpose**: Tests all processing algorithms with real diffusion data to ensure they produce valid results and maintain expected functionality after refactoring.

### InfrastructureComponentValidator

```python
class InfrastructureComponentValidator:
    def validate_configuration_management(self, config_scenarios: List[str]) -> ConfigValidationResult
    def validate_logging_system(self, workflow_events: List[str]) -> LoggingValidationResult
    def validate_scheduler_functionality(self, job_scenarios: List[str]) -> SchedulerValidationResult
```

**Purpose**: Tests infrastructure components that support the processing workflows, ensuring they handle real-world scenarios correctly.

### IntegrationValidator

```python
class IntegrationValidator:
    def validate_end_to_end_workflows(self, workflow_configs: List[str]) -> WorkflowValidationResult
    def validate_cross_module_interactions(self, interaction_scenarios: List[str]) -> InteractionValidationResult
```

**Purpose**: Tests complete processing pipelines and verifies that refactored components work together seamlessly.

### ValidationReporter

```python
class ValidationReporter:
    def generate_component_report(self, results: List[ComponentResult]) -> ComponentReport
    def generate_performance_analysis(self, metrics: PerformanceMetrics) -> PerformanceReport
    def generate_regression_analysis(self, baseline: BaselineMetrics, current: CurrentMetrics) -> RegressionReport
```

**Purpose**: Generates comprehensive reports documenting validation results, performance comparisons, and regression analysis.

## Data Models

### ValidationResult Base Class

```python
@dataclass
class ValidationResult:
    component_name: str
    test_name: str
    status: ValidationStatus  # PASS, FAIL, WARNING
    execution_time: float
    error_details: Optional[str]
    performance_metrics: Dict[str, Any]
    timestamp: datetime
```

### ComponentValidationResult

```python
@dataclass
class ComponentValidationResult(ValidationResult):
    sub_tests: List[ValidationResult]
    integration_status: ValidationStatus
    regression_detected: bool
    performance_comparison: Optional[PerformanceComparison]
```

### ValidationReport

```python
@dataclass
class ValidationReport:
    overall_status: ValidationStatus
    component_results: List[ComponentValidationResult]
    integration_results: List[IntegrationValidationResult]
    performance_summary: PerformanceSummary
    regression_summary: RegressionSummary
    recommendations: List[str]
    generated_at: datetime
```

### Test Data Configuration

```python
@dataclass
class TestDataConfig:
    bids_dataset_path: str
    subjects: List[str]
    sessions: List[str]
    processing_configs: Dict[str, str]
    expected_outputs: Dict[str, str]
    baseline_metrics: Optional[str]
```

## Error Handling

### Validation Exception Hierarchy

```python
class ValidationError(ElikopyError):
    """Base exception for validation errors"""
    pass

class ComponentValidationError(ValidationError):
    """Raised when a specific component fails validation"""
    def __init__(self, component: str, test: str, details: str)

class IntegrationValidationError(ValidationError):
    """Raised when integration tests fail"""
    def __init__(self, workflow: str, components: List[str], details: str)

class DataValidationError(ValidationError):
    """Raised when test data is invalid or inaccessible"""
    def __init__(self, data_path: str, issue: str)
```

### Error Recovery Strategy

**Graceful Degradation**: When individual component tests fail, the validation system continues with other components and clearly reports which tests could not be completed.

**Detailed Error Capture**: All exceptions are captured with full stack traces, component context, and data state information to facilitate debugging of refactored code.

**Isolation of Failures**: Component validation failures are isolated to prevent cascading failures in other validation modules.

## Testing Strategy

### Test Data Requirements

**Real BIDS Dataset**: The validation system requires access to a real BIDS-compliant diffusion MRI dataset with multiple subjects and sessions to test various data scenarios.

**Baseline Metrics**: For regression testing, baseline performance metrics from the original codebase are stored and compared against refactored code performance.

**Configuration Variants**: Multiple configuration scenarios are tested to ensure the refactored code handles different processing parameters correctly.

### Validation Test Categories

1. **Functional Validation**: Verifies that refactored components produce expected outputs
2. **Performance Validation**: Compares processing times and resource usage between original and refactored code
3. **Integration Validation**: Tests component interactions and data flow between modules
4. **Regression Validation**: Identifies any functionality lost during refactoring
5. **Error Handling Validation**: Tests that error conditions are handled appropriately

### Test Execution Strategy

**Parallel Execution**: Independent component validations run in parallel to reduce total validation time.

**Incremental Validation**: The system supports running validation on specific components or workflows for targeted testing during development.

**Continuous Validation**: The validation framework can be integrated into CI/CD pipelines for ongoing validation of code changes.

### Validation Metrics

**Coverage Metrics**: Tracks which components, functions, and code paths are exercised during validation.

**Performance Metrics**: Measures execution time, memory usage, and computational efficiency.

**Quality Metrics**: Assesses output quality, numerical accuracy, and BIDS compliance of generated derivatives.

**Reliability Metrics**: Measures consistency of results across multiple runs and different data inputs.

## Implementation Considerations

### Real Data Handling

**Data Privacy**: The validation system includes mechanisms to ensure test data is handled securely and in compliance with data sharing agreements.

**Data Preparation**: Automated scripts prepare test datasets by selecting appropriate subjects and ensuring data quality for validation purposes.

**Resource Management**: The validation system manages computational resources efficiently, especially when processing large real datasets.

### Performance Optimization

**Caching Strategy**: Intermediate results are cached to avoid redundant processing during iterative validation runs.

**Resource Monitoring**: The system monitors CPU, memory, and disk usage during validation to identify performance bottlenecks.

**Scalability**: The validation framework is designed to scale with dataset size and can utilize HPC resources when available.

### Reporting and Documentation

**Interactive Reports**: Validation reports include interactive elements for exploring results and drilling down into specific failures.

**Automated Documentation**: The system automatically generates documentation of validation procedures and results for compliance and audit purposes.

**Integration with Development Workflow**: Validation results are formatted for integration with development tools and CI/CD systems.