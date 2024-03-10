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
    obs = pd.DataFrame(
        [
            ["A", "D1", "X", 0.2],
            ["B", "D1", "X", 0.7],
            ["A", "D2", "Y", 1.6],
            ["B", "D2", "Y", 42],
            ["A", "D3", "Y", 212],
            ["B", "D3", "Y", 6023],
        ],
        columns=["condition", "donor", "other", "continuous"],
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
            [13, 600],
            [24, 620],
        ]
    )
    return ad.AnnData(X=X, obs=obs, var=var)
