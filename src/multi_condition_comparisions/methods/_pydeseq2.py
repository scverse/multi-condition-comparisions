import os
import re
import warnings

import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

from multi_condition_comparisions._util import check_is_integer_matrix

from ._base import LinearModelBase


class PyDESeq2(LinearModelBase):
    """Differential expression test using a PyDESeq2"""

    def _check_counts(self):
        check_is_integer_matrix(self.data)

    def fit(self, **kwargs) -> pd.DataFrame:
        """
        Fit dds model using pydeseq2.

        Note: this creates its own AnnData object for downstream processing.

        Params:
        ----------
        **kwargs
            Keyword arguments specific to DeseqDataSet(), except for `n_cpus` which will use all available CPUs minus one if the argument is not passed.
        """
        inference = DefaultInference(n_cpus=kwargs.pop("n_cpus", os.cpu_count() - 1))
        covars = self.design.columns.tolist()
        if "Intercept" not in covars:
            warnings.warn(
                "Warning: Pydeseq is hard-coded to use Intercept, please include intercept into the model", stacklevel=2
            )
        processed_covars = list({re.sub(r"\[T\.(.*)\]", "", col) for col in covars if col != "Intercept"})
        dds = DeseqDataSet(
            adata=self.adata, design_factors=processed_covars, refit_cooks=True, inference=inference, **kwargs
        )
        # workaround code to insert design array
        des_mtx_cols = dds.obsm["design_matrix"].columns
        dds.obsm["design_matrix"] = self.design
        if dds.obsm["design_matrix"].shape[1] == len(des_mtx_cols):
            dds.obsm["design_matrix"].columns = des_mtx_cols.copy()

        dds.deseq2()
        self.dds = dds

    def _test_single_contrast(self, contrast: list[str], alpha=0.05, **kwargs) -> pd.DataFrame:
        """
        Conduct a specific test and returns a Pandas DataFrame.

        Parameters
        ----------
        contrasts:
            list of three strings of the form
            ["variable", "tested level", "reference level"]
        alpha: p value threshold used for controlling fdr with
        independent hypothesis weighting
        kwargs: extra arguments to pass to DeseqStats()
        """
        stat_res = DeseqStats(self.dds, contrast=contrast, alpha=alpha, **kwargs)
        # Calling `.summary()` is required to fill the `results_df` data frame
        stat_res.summary()
        res_df = (
            pd.DataFrame(stat_res.results_df)
            .rename(columns={"pvalue": "p_value", "padj": "adj_p_value", "log2FoldChange": "log_fc"})
            .sort_values("p_value")
        )
        res_df.index.name = "variable"
        res_df = res_df.reset_index()
        return res_df
