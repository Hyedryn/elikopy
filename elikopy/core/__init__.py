"""
Core module for ElikoPy
=======================

This module contains the main API classes for ElikoPy:
- ElikopyStudy: Main entry point for study management
- ElikopyProcessor: Processing orchestration
- ElikopyConfig: Configuration management
"""

from elikopy.core.study import ElikopyStudy
from elikopy.core.processor import ElikopyProcessor
from elikopy.core.config import ElikopyConfig

__all__ = [
    'ElikopyStudy',
    'ElikopyProcessor',
    'ElikopyConfig'
]