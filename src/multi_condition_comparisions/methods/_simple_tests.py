"""Simple tests such as t-test, wilcoxon"""

from collections.abc import Sequence

from anndata import AnnData
from pandas.core.api import DataFrame as DataFrame

from ._base import MethodBase


class WilcoxonTest(MethodBase):
    @classmethod
    def compare_groups(
        cls,
        adata: AnnData,
        variable: str,
        baseline: str | None = None,
        groups_to_compare: str | Sequence[str] | None = None,
        *,
        paired_by: str = None,
        mask: str | None = None,
        layer: str | None = None,
    ) -> DataFrame:
        """
        Perform a unpaired or paired Wilcoxon test (the latter is also known as 'Mann-Whitney' test).

        TODO: should probably reuse the docstring from the parent method?

        Parameters
        ----------
        adata
            AnnData object
        variable
            variable from X or obs to compare
        baseline
            baseline value (one category from variable). If set to "None" this refers to "all other categories".
        groups_to_compare
            One or multiple categories from variable to compare against baseline. Setting this to None refers to
            "all categories"
        paired_by
            Column from `obs` that contains information about paired sample (e.g. subject_id)
        mask
            Subset anndata by a boolean mask stored in this column in `.obs` before making any tests
        layer
            Use this layer instead of `.X`.

        Returns
        -------
        Pandas dataframe with results ordered by significance. If multiple comparisons were performed this
        is indicated in an additional column.
        """
        cls(adata, mask=mask, layer=layer)
