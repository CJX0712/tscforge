"""TscForge: deterministic time series classification.

TinyRocket (pure numpy ROCKET variant) + PRISM flagship
(Pooling-fused RIdge with dtw-diStance Merging).

Author: 晨星 (CJX0712)
"""

import os

# Thread pinning MUST happen before numpy is imported anywhere in the process
# so that BLAS reduction order is deterministic (single-threaded).
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

__version__ = "0.1.0"
__author__ = "晨星"
