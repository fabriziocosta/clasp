from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


@pytest.fixture
def toy_adata() -> AnnData:
    rng = np.random.default_rng(7)
    centers = np.array([[0, 0], [3, 0], [0, 3], [3, 3]], dtype=float)
    X_parts = []
    labels = []
    batches = []
    for batch_id, shift in [("batch_a", 0.0), ("batch_b", 0.35)]:
        for label_id, center in [("type_1", centers[0]), ("type_2", centers[3])]:
            low_dim = rng.normal(center + shift, 0.2, size=(8, 2))
            noise = rng.normal(0, 0.05, size=(8, 8))
            X_parts.append(np.hstack([low_dim, noise]))
            labels.extend([label_id] * 8)
            batches.extend([batch_id] * 8)
    X = np.vstack(X_parts)
    adata = AnnData(X, obs=pd.DataFrame({"batch": batches, "label": labels}))
    return adata
