"""Notebook kernel startup configuration.

VS Code starts Jupyter kernels with ``notebooks/`` as the working directory,
so Python imports this module before user notebook code. Keep native numerical
thread settings here because setting them in the first notebook cell can be too
late after ipykernel or extensions have imported compiled libraries.
"""

from __future__ import annotations

import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_WARNINGS", "FALSE")
os.environ.setdefault("KMP_SETTINGS", "FALSE")
