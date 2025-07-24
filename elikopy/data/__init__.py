"""
Data module for ElikoPy
======================

This module handles BIDS data access, validation, and derivatives management:
- BIDSHandler: BIDS data access and organization
- DataValidator: Data validation utilities
- DerivativesManager: BIDS derivatives management
"""

from elikopy.data.bids_handler import BIDSHandler
from elikopy.data.validator import DataValidator
from elikopy.data.derivatives import DerivativesManager

__all__ = [
    'BIDSHandler',
    'DataValidator', 
    'DerivativesManager'
]