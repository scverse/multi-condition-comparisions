import numpy as np
import pytest
import statsmodels.api as sm
from pandas import testing as tm

import multi_condition_comparisions
from multi_condition_comparisions.tl.de import BaseMethod, PyDESeq2DE, StatsmodelsDE


def test_package_has_version():
    assert multi_condition_comparisions.__version__ is not None


@pytest.mark.parametrize(
    "method_class,kwargs",
    [
        # OLS
        (StatsmodelsDE, {}),
        # Negative Binomial
        (
            StatsmodelsDE,
            {"regression_model": sm.GLM, "family": sm.families.NegativeBinomial()},
        ),
    ],
)
def test_de(test_adata, method_class: BaseMethod, kwargs):
    """Check that the method can be initialized and fitted, and perform basic checks on
    the result of test_contrasts."""
    method = method_class(adata=test_adata, design="~condition")
    method.fit(**kwargs)
    res_df = method.test_contrasts(np.array([0, 1]))
    # Check that the result has the correct number of rows
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
    # Check that the index of the result matches the var_names of the adata
    tm.assert_index_equal(test_adata.var_names, res_df.index, check_order=False, check_names=False)
    # Check that there is a p-value column
    assert "pvalue" in res_df.columns
    # Check that p-values are between 0 and 1
    assert np.all((0 <= res_df["pvalue"]) & (res_df["pvalue"] <= 1))
