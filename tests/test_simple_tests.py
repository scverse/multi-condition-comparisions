import numpy as np
import pandas as pd
import pytest
from pandas.core.api import DataFrame as DataFrame

from multi_condition_comparisions.methods import SimpleComparisonBase, TTest, WilcoxonTest


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
            assert np.all(x0.obs["pairing"].values == x1.obs["pairing"].values)
            return pd.DataFrame()

    rng = np.random.default_rng(seed)
    shuffle_adata_idx = rng.permutation(test_adata_minimal.obs_names)
    tmp_adata = test_adata_minimal[shuffle_adata_idx, :].copy()

    MockSimpleComparison.compare_groups(
        tmp_adata, column="condition", baseline="A", groups_to_compare=["B"], paired_by="pairing"
    )
