import anndata as ad
import numpy as np
import pytest
from pydeseq2.utils import load_example_data

import multi_condition_comparisions
from multi_condition_comparisions.tl.de import MethodBase


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


@pytest.param("method_class", [])
def test_de(method_class: MethodBase):
    """Check that the method can be initialized and fitted and that the test_contrast
    method returns a dataframe with the correct number of rows"""
    method = method_class(adata=test_adata, design="~condition")
    method.fit()
    res_df = method.test_contrast(np.array([0, 1]))
    assert len(res_df) == test_adata.n_vars
