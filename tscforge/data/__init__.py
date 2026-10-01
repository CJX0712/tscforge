"""Data layer: synthetic generators + stratified split.

Author: 晨星 (CJX0712)
"""

from tscforge.data.generators import (
    DATASET_REGISTRY,
    make_order_freq2a,
    make_order_freq2b,
    make_order_freq2c,
    make_order_freq3a,
    make_order_freq3b,
)
from tscforge.data.split import stratified_split

__all__ = [
    "DATASET_REGISTRY",
    "make_order_freq2a",
    "make_order_freq2b",
    "make_order_freq2c",
    "make_order_freq3a",
    "make_order_freq3b",
    "stratified_split",
]
