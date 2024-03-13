"""Simple tests such as t-test, wilcoxon"""

import warnings
from abc import abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import scipy.stats
from anndata import AnnData
from pandas.core.api import DataFrame as DataFrame
from scipy.sparse import issparse
from tqdm.auto import tqdm

from ._base import MethodBase


class SimpleComparisonBase(MethodBase):
    @staticmethod
    @abstractmethod
    def _get_test_fun(paired) -> Callable:
        """Return a function with the same signature as e.g. `scipy.stats.wilcoxon`"""
        ...

    def _compare_single_group(self, baseline_idx: np.ndarray, comparison_idx: np.ndarray, *, paired: bool) -> DataFrame:
        """
        Perform a single comparison between two groups.

        Parameters
        ----------
        baseline_idx
            numeric indices indicating which observations are in the baseline group
        comparison_idx
            numeric indices indicating which observations are in the comparison/treatment group
        paired
            whether or not to perform a paired test. Note that in the case of a paired test,
            the indices must be ordered such that paired observations appear at the same position.
        """
        if paired:
            assert len(baseline_idx) == len(comparison_idx), "For a paired test, indices must be of the same length"
        test_fun = self._get_test_fun(paired)

        x0 = self.data[baseline_idx, :]
        x1 = self.data[comparison_idx, :]

        # In the following loop, we are doing a lot of column slicing -- which is significantly
        # more efficient in csc format.
        if issparse(self.data):
            x0 = x0.tocsc()
            x1 = x1.tocsc()

        res = []
        for var in tqdm(self.adata.var_names):
            tmp_x0 = x0[:, self.adata.var_names == var]
            tmp_x0 = np.asarray(tmp_x0.todense()).flatten() if issparse(tmp_x0) else tmp_x0.flatten()
            tmp_x1 = x1[:, self.adata.var_names == var]
            tmp_x1 = np.asarray(tmp_x1.todense()).flatten() if issparse(tmp_x1) else tmp_x1.flatten()
            pval = test_fun(x=tmp_x0, y=tmp_x1).pvalue
            mean_x0 = np.asarray(np.mean(x0, axis=0)).flatten().astype(dtype=float)
            mean_x1 = np.asarray(np.mean(x1, axis=0)).flatten().astype(dtype=float)
            res.append({"variable": var, "pvals": pval, "fold_change": np.log(mean_x1) - np.log(mean_x0)})
        return pd.DataFrame(res).sort_values("pvals").set_index("variable")

    @classmethod
    def compare_groups(
        cls,
        adata: AnnData,
        column: str,
        baseline: str,
        groups_to_compare: str | Sequence[str],
        *,
        paired_by: str | None = None,
        mask: str | None = None,
        layer: str | None = None,
        fit_kwargs: dict = None,
        test_kwargs: dict = None,
    ) -> DataFrame:
        if test_kwargs is None:
            test_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}
        if len(fit_kwargs) or len(test_kwargs):
            warnings.warn("Simple tests do not use fit or test kwargs", UserWarning, stacklevel=2)
        paired = paired_by is not None
        if paired:
            adata = adata.copy()[adata.obs.sort_values(paired_by).index, :]
        model = cls(adata, mask=mask, layer=layer)
        if isinstance(groups_to_compare, str):
            groups_to_compare = [groups_to_compare]

        res_dfs = []
        for group_to_compare in groups_to_compare:
            comparison_idx = np.where(adata.obs[column] == group_to_compare)[0]
            if baseline is None:
                baseline_idx = np.where(adata.obs[column] != group_to_compare)[0]
            else:
                baseline_idx = np.where(adata.obs[column] == baseline)[0]
            res_dfs.append(
                model._compare_single_group(baseline_idx, comparison_idx, paired=paired).assign(
                    comparison=f"{group_to_compare}_vs_{baseline if baseline is not None else 'rest'}"
                )
            )
        return pd.concat(res_dfs)


class WilcoxonTest(SimpleComparisonBase):
    """Perform a unpaired or paired Wilcoxon test.

    (the former is also known as "Mann-Whitney U test", the latter as "wilcoxon signed rank test")
    """

    @staticmethod
    def _get_test_fun(paired) -> Callable:
        if paired:
            return scipy.stats.wilcoxon
        else:
            return scipy.stats.mannwhitneyu


class TTest(SimpleComparisonBase):
    """Perform a unpaired or paired T-test"""

    @staticmethod
    def _get_test_fun(paired) -> Callable:
        if paired:
            return scipy.stats.ttest_rel
        else:
            return scipy.stats.ttest_ind
