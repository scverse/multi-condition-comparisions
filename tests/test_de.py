import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from pandas import testing as tm
from pandas.core.api import DataFrame as DataFrame

import multi_condition_comparisions
from multi_condition_comparisions.methods import EdgeR, PyDESeq2, SimpleComparisonBase, Statsmodels, TTest, WilcoxonTest


def test_package_has_version():
    assert multi_condition_comparisions.__version__ is not None


@pytest.mark.parametrize(
    "method_class,kwargs",
    [
        # OLS
        (Statsmodels, {}),
        # Negative Binomial
        (
            Statsmodels,
            {"regression_model": sm.GLM, "family": sm.families.NegativeBinomial()},
        ),
    ],
)
def test_statsmodels(test_adata, method_class, kwargs):
    """Check that the method can be initialized and fitted, and perform basic checks on
    the result of test_contrasts."""
    method = method_class(adata=test_adata, design="~condition")  # type: ignore
    method.fit(**kwargs)
    res_df = method.test_contrasts(np.array([0, 1]))
    # Check that the result has the correct number of rows
    assert len(res_df) == test_adata.n_vars


def test_pydeseq2_simple(test_adata):
    """Check that the pyDESeq2 method can be

    1. Initialized
    2. Fitted
    3. and that test_contrast returns a DataFrame with the correct number of rows.
    """
    method = PyDESeq2(adata=test_adata, design="~condition")
    method.fit()
    res_df = method.test_contrasts(["condition", "A", "B"])

    assert len(res_df) == test_adata.n_vars


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


def test_pydeseq2_complex(test_adata):
    """Check that the pyDESeq2 method can be initialized with a different covariate name and fitted and that the test_contrast
    method returns a dataframe with the correct number of rows.
    """
    test_adata.obs["condition1"] = test_adata.obs["condition"].copy()
    method = PyDESeq2(adata=test_adata, design="~condition1+group")
    method.fit()
    res_df = method.test_contrasts(["condition1", "A", "B"])

    assert len(res_df) == test_adata.n_vars
    # Check that the index of the result matches the var_names of the AnnData object
    tm.assert_index_equal(test_adata.var_names, res_df.index, check_order=False, check_names=False)

    expected_columns = {"pvals", "pvals_adj", "logfoldchanges"}
    assert expected_columns.issubset(set(res_df.columns))
    assert np.all((0 <= res_df["pvals"]) & (res_df["pvals"] <= 1))


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


@pytest.mark.parametrize("paired_by", [None, "pairings"])
def test_wilcoxon(test_adata, paired_by):
    if paired_by is not None:
        test_adata.obs[paired_by] = list(range(sum(test_adata.obs["condition"] == "A"))) * 2
    res_df = WilcoxonTest.compare_groups(
        adata=test_adata, column="condition", baseline="A", groups_to_compare="B", paired_by=paired_by
    )
    assert np.all((0 <= res_df["pvals"]) & (res_df["pvals"] <= 1))  # TODO: which of these should actually be <.05?


@pytest.mark.parametrize("paired_by", [None, "pairings"])
def test_t(test_adata, paired_by):
    if paired_by is not None:
        test_adata.obs[paired_by] = list(range(sum(test_adata.obs["condition"] == "A"))) * 2
    res_df = TTest.compare_groups(
        adata=test_adata, column="condition", baseline="A", groups_to_compare="B", paired_by=paired_by
    )
    assert np.all((0 <= res_df["pvals"]) & (res_df["pvals"] <= 1))  # TODO: which of these should actually be <.05?


@pytest.mark.parametrize("seed", range(10))
def test_simple_comparison_pairing(test_adata_minimal, seed):
    """Test that paired samples are properly matched in a paired test"""

    class MockSimpleComparison(SimpleComparisonBase):
        @staticmethod
        def _test():
            return None

        def _compare_single_group(
            self, baseline_idx: np.ndarray, comparison_idx: np.ndarray, *, paired: bool = False, **kwargs
        ) -> DataFrame:
            assert paired
            x0 = self.adata[baseline_idx, :]
            x1 = self.adata[comparison_idx, :]
            assert np.all(x0.obs["condition"] == "A")
            assert np.all(x1.obs["condition"] == "B")
            return pd.DataFrame()

    rng = np.random.default_rng(seed)
    shuffle_adata_idx = rng.permutation(test_adata_minimal.obs_names)
    tmp_adata = test_adata_minimal[shuffle_adata_idx, :].copy()

    MockSimpleComparison.compare_groups(
        tmp_adata, column="condition", baseline="A", groups_to_compare=["B"], paired_by="donor"
    )
