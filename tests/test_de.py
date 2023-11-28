import anndata as ad
import pytest
from pydeseq2.utils import load_example_data

import multi_condition_comparisions


def test_package_has_version():
    assert multi_condition_comparisions.__version__ is not None


@pytest.fixture
def test_adata():
    counts = load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )

    metadata = load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )

    return ad.AnnData(X=counts, obs=metadata)


@pytest.param("method")
def test_de(method):
    raise NotImplementedError
