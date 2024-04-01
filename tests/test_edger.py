import numpy as np
from pandas import testing as tm
from pandas.core.api import DataFrame as DataFrame

from multi_condition_comparisions.methods import EdgeR


def test_edger_simple(test_adata):
    """Check that the EdgeR method can be

    1. Initialized
    2. Fitted
    3. and that test_contrast returns a DataFrame with the correct number of rows.
    """
    method = EdgeR(adata=test_adata, design="~condition")
    method.fit()
    res_df = method.test_contrasts(["condition", "A", "B"])

    assert len(res_df) == test_adata.n_vars


def test_edger_complex(test_adata):
    """Check that the EdgeR method can be initialized with a different covariate name and fitted and that the test_contrast
    method returns a dataframe with the correct number of rows.
    """
    test_adata.obs["condition1"] = test_adata.obs["condition"].copy()
    method = EdgeR(adata=test_adata, design="~condition1+group")
    method.fit()
    res_df = method.test_contrasts(["condition1", "A", "B"])

    assert len(res_df) == test_adata.n_vars
    # Check that the index of the result matches the var_names of the AnnData object
    tm.assert_index_equal(test_adata.var_names, res_df.index, check_order=False, check_names=False)

    expected_columns = {"pvals", "pvals_adj", "logfoldchanges"}
    assert expected_columns.issubset(set(res_df.columns))
    assert np.all((0 <= res_df["pvals"]) & (res_df["pvals"] <= 1))
