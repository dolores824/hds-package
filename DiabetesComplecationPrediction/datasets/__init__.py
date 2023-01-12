"""
This module is for loading the sample datasets.
"""

from .base import load_csv, diabetes_cvd_risk, diabetes_IgAN_risk

__all__ = [
    'load_csv', 'diabetes_cvd_risk', 'diabetes_IgAN_risk'
]