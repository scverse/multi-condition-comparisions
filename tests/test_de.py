import anndata as ad
import numpy as np
import pytest
from pydeseq2.utils import load_example_data

import multi_condition_comparisions
from multi_condition_comparisions.tl.de import BaseMethod, PyDESeq2DE, StatsmodelsDE


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


@pytest.mark.parametrize("method_class", [StatsmodelsDE])
def test_de(test_adata, method_class: BaseMethod):
    """Check that the method can be initialized and fitted and that the test_contrast
    method returns a dataframe with the correct number of rows"""
    method = method_class(adata=test_adata, design="~condition")
    method.fit()
    res_df = method.test_contrasts(np.array([0, 1]))
    assert len(res_df) == test_adata.n_vars

def test_pydeseq2de(test_adata):
    """Check that the pyDESeq2 method can be initialized and fitted and that the test_contrast
    method returns a dataframe with the correct number of rows.

    Now this is a separate  
    
    """
    method = PyDESeq2DE(adata=test_adata, design="~condition")
    method.fit()
    res_df = method.test_contrasts(['condition', 'A', 'B'])
    assert len(res_df) == test_adata.n_vars

def test_pydeseq2de2(test_adata):
    """Check that the pyDESeq2 method can be initialized with a different covariate name and fitted and that the test_contrast
    method returns a dataframe with the correct number of rows.

    Now this is a separate  
    
    """
    test_adata.obs['condition1'] = test_adata.obs['condition'].copy()
    method = PyDESeq2DE(adata=test_adata, design="~condition1+group")   
    method.fit()
    res_df = method.test_contrasts(['condition1', 'A', 'B'])
    assert len(res_df) == test_adata.n_vars

