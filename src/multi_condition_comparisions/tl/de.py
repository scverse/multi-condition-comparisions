from typing import List
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.regression.linear_model
from anndata import AnnData
from formulaic import model_matrix
from formulaic.model_matrix import ModelMatrix
from tqdm.auto import tqdm

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

class BaseMethod(ABC):
    def __init__(
        self, adata: AnnData, design: str | np.ndarray, mask: str | None = None, layer: str | None = None, **kwargs
    ):
        """
        Initialize the method

        Parameters
        ----------
        adata
            AnnData object, usually pseudobulked.
        design
            Model design. Can be either a design matrix, a formulaic formula.
        mask
            a column in adata.var that contains a boolean mask with selected features.
        **kwargs
            Keyword arguments specific to the method implementation
        """
        self.adata = adata
        if mask is not None:
            self.adata = self.adata[:, self.adata.var[mask]]
        self.layer = layer
        if isinstance(design, str):
            self.design = model_matrix(design, adata.obs)
        else:
            self.design = design

    @property
    def variables(self):
        """Get the names of the variables used in the model definition"""
        return self.design.model_spec.variables_by_source["data"]

    @abstractmethod
    def fit(self, **kwargs) -> None:
        """
        Fit the model

        Parameters
        ----------
        **kwargs
            Additional arguments for fitting the specific method.
        """
        ...

    @abstractmethod
    def _test_single_contrast(self, contrast, **kwargs) -> pd.DataFrame:
        ...

    def test_contrasts(self, contrasts: dict[str, np.ndarray] | np.ndarray, **kwargs) -> pd.DataFrame:
        """
        Conduct a specific test

        Parameters
        ----------
        contrasts:
            either a single contrast, or a dictionary of contrasts where the key is the name for that particular contrast.
            Each contrast can be either a vector of coefficients (the most general case), a string, or a some fancy DSL
            (details still need to be figured out).

            or a tuple withe three elements contrasts = ("condition", "control", "treatment")
        """
        if not isinstance(contrasts, dict):
            contrasts = {None: contrasts}
        results = []
        for name, contrast in contrasts.items():
            results.append(self._test_single_contrast(contrast, **kwargs).assign(contrast=name))
        return pd.concat(results)

    def test_reduced(self, modelB: "BaseMethod") -> pd.DataFrame:
        """
        Test against a reduced model

        Parameters
        ----------
        modelB:
            the reduced model against which to test.

        Example:
        --------
        ```
        modelA = Model().fit()
        modelB = Model().fit()
        modelA.test_reduced(modelB)
        ```
        """
        raise NotImplementedError

    def cond(self, **kwargs) -> np.ndarray:
        """
        The intention is to make contrasts using this function as in glmGamPoi

        >>> res <- test_de(fit, contrast = cond(cell = "B cells", condition = "stim") - cond(cell = "B cells", condition = "ctrl"))

        Parameters
        ----------
        **kwargs

        """
        if not isinstance(self.design, ModelMatrix):
            raise RuntimeError(
                "Building contrasts with `cond` only works if you specified the model using a "
                "formulaic formula. Please manually provide a contrast vector."
            )

        # TODO pre-fill the dictionary with baseline values for keys that are not specified
        # TODO not sure how that works for continuous variables
        # for factor, factor_info in self.design.design_info.factor_infos.items():
        #     pass

        return self.design.model_spec.get_model_matrix(pd.DataFrame([kwargs]))

    def contrast(self, column: str, baseline: str, group_to_compare: str) -> np.ndarray:
        """
        Build a simple contrast for pairwise comparisons.

        This is equivalent to

        ```
        model.cond(<column> = baseline) - model.cond(<column> = group_to_compare)
        ```
        """
        return self.cond(**{column: baseline}) - self.cond(**{column: group_to_compare})


class StatsmodelsDE(BaseMethod):
    """Differential expression test using a statsmodels linear regression"""

    def fit(self):
        """Fit the OLS model"""
        self.models = []
        for var in tqdm(self.adata.var_names):
            mod = statsmodels.regression.linear_model.OLS(
                sc.get.obs_df(self.adata, keys=[var], layer=self.layer)[var], self.design
            )
            mod = mod.fit()
            self.models.append(mod)

    def _test_single_contrast(self, contrast, **kwargs) -> pd.DataFrame:
        res = []
        for var, mod in zip(tqdm(self.adata.var_names), self.models):
            t_test = mod.t_test(contrast)
            res.append(
                {
                    "variable": var,
                    "pvalue": t_test.pvalue,
                    "tvalue": t_test.tvalue.item(),
                    "sd": t_test.sd.item(),
                    "fold_change": t_test.effect.item(),
                }
            )
        return pd.DataFrame(res).sort_values("pvalue")


class PyDESeq2DE(BaseMethod):
    """Differential expression test using a PyDESeq2"""

    def fit(self, **kwargs) -> pd.DataFrame:
        '''
        Fit dds model using pydeseq2. Note: this creates its own adata object for downstream. 

        Params:
        ----------
        **kwargs
            Keyword arguments specific to DeseqDataSet()
        '''
        
        inference = DefaultInference(n_cpus=3)
        dds = DeseqDataSet(adata=self.adata, design_factors="condition", refit_cooks=True, inference=inference, **kwargs)
        dds.obsm['design_matrix'] = pd.DataFrame(self.design, index = self.adata.obs_names.copy())
        #implement correct naming of the columns in design matrix for
        # downstream 
        dds.deseq2()
        self.dds = dds
        
    def _test_single_contrast(self, contrast: List[str],  alpha = 0.05, **kwargs) -> pd.DataFrame:
            """
            Conduct a specific test and returns a data frame

            Parameters
            ----------
            contrasts:
                list of three strings of the form 
                ["variable", "tested level", "reference level"]
            alpha: p value threshold used for controlling fdr with 
            independent hypothesis weighting  
            kwargs: extra arguments to pass to DeseqStats()
            """
            stat_res = DeseqStats(self.dds, contrast = contrast, alpha=alpha, **kwargs)
            stat_res.summary()
            stat_res.p_values
            return pd.DataFrame(stat_res.results_df).sort_values("padj")