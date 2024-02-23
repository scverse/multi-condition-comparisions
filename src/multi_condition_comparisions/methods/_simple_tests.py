"""Simple tests such as t-test, wilcoxon"""

from abc import abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.stats
from anndata import AnnData
from pandas.core.api import DataFrame as DataFrame
from scipy.sparse import issparse
from tqdm.auto import tqdm

from ._base import MethodBase


class SimpleComparisonBase(MethodBase):
    @abstractmethod
    def _compare_single_group(
        self, baseline_idx: np.ndarray, comparison_idx: np.ndarray, *, paired: bool = False
    ) -> pd.DataFrame:
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
        ...

    @classmethod
    def compare_groups(
        cls,
        adata: AnnData,
        column: str,
        baseline: str | None = None,
        groups_to_compare: str | Sequence[str] | None = None,
        *,
        paired_by: str = None,
        mask: str | None = None,
        layer: str | None = None,
    ) -> DataFrame:
        model = cls(adata, mask=mask, layer=layer)
        if isinstance(groups_to_compare, str):
            groups_to_compare = [groups_to_compare]
        if groups_to_compare is None:
            groups_to_compare = list(model.adata.obs[column].unique())

        if paired_by is not None:
            raise NotImplementedError("TODO: reorder the indices accordingly and pass to _compare_single_group")

        res_dfs = []
        for group_to_compare in groups_to_compare:
            comparison_idx, _ = np.where(model.adata.obs[column] == group_to_compare)
            if baseline is None:
                baseline_idx, _ = np.where(model.adata.obs[column] != group_to_compare)
            else:
                baseline_idx, _ = np.where(model.adata.obs[column] == baseline)
            res_dfs.append(
                model._compare_single_group(baseline_idx, comparison_idx).assign(
                    comparison=f"{group_to_compare}_vs_{baseline if baseline is not None else 'rest'}"
                )
            )
        return pd.concat(res_dfs)


class WilcoxonTest(SimpleComparisonBase):
    """Perform a unpaired or paired Wilcoxon test.

    (the former is also known as "Mann-Whitney U test", the latter as "wilcoxon signed rank test")
    """

    def _compare_single_group(
        self, baseline_idx: np.ndarray, comparison_idx: np.ndarray, *, paired: bool = False
    ) -> DataFrame:
        if paired:
            assert len(baseline_idx) == len(comparison_idx), "For a paired test, indices must be of the same length"
            test_fun = scipy.stats.wilcoxon
        else:
            test_fun = scipy.stats.mannwhitneyu

        # TODO can be made more efficient by converting CSR/CSC matrices accordingly
        x0 = self.data[baseline_idx, :]
        x1 = self.data[comparison_idx, :]
        res = []
        for var in tqdm(self.adata.var_names):
            tmp_x0 = x0[:, self.adata.var_names == var]
            tmp_x0 = np.asarray(x0.todense()).flatten() if issparse(x0) else x0.flatten()
            tmp_x1 = x1[:, self.adata.var_names == var]
            tmp_x1 = np.asarray(x1.todense()).flatten() if issparse(x1) else x1.flatten()
            pval = test_fun(x=tmp_x0, y=tmp_x1).pvalue
            res.append({"variable": var, "pvalue": pval, "fold_change": "TODO"})
        return pd.DataFrame(res).sort_values("pvalue").set_index("variable")
