import anndata as ad
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
