import anndata as ad
import pandas as pd
import numpy as np
import pytest
from pydeseq2.utils import load_example_data

from multi_condition_comparisions.tl.de import StatsmodelsDE


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
    obs = pd.DataFrame(
        [
            ["A", "D1", "X"],
            ["B", "D1", "X"],
            ["A", "D2", "Y"],
            ["B", "D2", "Y"],
        ],
        columns=["condition", "donor", "other"],
    )
    var = pd.DataFrame(index=["gene1", "gene2"])
    X = np.array(
        [
            # gene1 differs between condition A/B by approx. FC=2
            # gene2 differs between donor D1/D2 by approx FC=2
            # [gene1, gene2]
            [10, 200],
            [20, 220],
            [12, 400],
            [25, 420],
        ]
    )
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def statsmodels_stub(test_adata):
    return StatsmodelsDE(adata=test_adata, design="~condition")
