import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pydeseq2.utils import load_example_data


@pytest.fixture
def test_counts():
    return load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )


@pytest.fixture
def test_metadata():
    return load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )


@pytest.fixture
def test_adata(test_counts, test_metadata):
    return ad.AnnData(X=test_counts, obs=test_metadata)


@pytest.fixture
def test_adata_minimal():
    n_obs = 80
    n_donors = n_obs // 4
    obs = pd.DataFrame(
        {
            "condition": ["A", "B"] * (n_obs // 2),
            "donor": sum(([f"D{i}"] * n_donors for i in range(n_obs // n_donors)), []),
            "other": (["X"] * (n_obs // 4)) + (["Y"] * ((3 * n_obs) // 4)),
            "pairing": sum(([str(i), str(i)] for i in range(n_obs // 2)), []),
            "continuous": np.random.uniform(0, 1) * 4000,
        },
    )
    var = pd.DataFrame(index=["gene1", "gene2"])
    rng = np.random.default_rng(9)  # make tests deterministic
    group1 = rng.negative_binomial(20, 0.1, n_obs // 2)  # large mean
    group2 = rng.negative_binomial(5, 0.5, n_obs // 2)  # small mean

    condition_data = np.empty((n_obs,), dtype=group1.dtype)
    condition_data[0::2] = group1
    condition_data[1::2] = group2

    donor_data = np.empty((n_obs,), dtype=group1.dtype)
    donor_data[0:n_donors] = group2[:n_donors]
    donor_data[n_donors : (2 * n_donors)] = group1[n_donors:]

    donor_data[(2 * n_donors) : (3 * n_donors)] = group2[:n_donors]
    donor_data[(3 * n_donors) :] = group1[n_donors:]

    X = np.vstack([condition_data, donor_data]).T
    return ad.AnnData(X=X, obs=obs, var=var)
