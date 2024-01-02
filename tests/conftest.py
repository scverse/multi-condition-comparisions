import anndata as ad
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
def statsmodels_stub(test_adata):
    return StatsmodelsDE(adata=test_adata, design="~condition")
