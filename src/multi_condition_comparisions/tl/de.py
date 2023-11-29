from typing import List
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
from anndata import AnnData
from formulaic import model_matrix
from formulaic.model_matrix import ModelMatrix
from pandas import DataFrame
from tqdm.auto import tqdm

import rpy2.robjects.pandas2ri
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import STAP
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
# if you have a different version of rpy2 you may not need these two lines
rpy2.robjects.pandas2ri.activate()
rpy2.robjects.numpy2ri.activate()
from scanpy import logging
from scipy.sparse import issparse


class BaseMethod(ABC):
    def __init__(
        self,
        adata: AnnData,
        design: str | np.ndarray,
        mask: str | None = None,
        layer: str | None = None,
        **kwargs,
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
            A column in adata.var that contains a boolean mask with selected features.
        layer
            Layer to use in fit(). If None, use the X matrix.
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

    def fit(
        self,
        regression_model: sm.OLS | sm.GLM = sm.OLS,
        **kwargs,
    ) -> None:
        """
        Fit the specified regression model.

        Parameters
        ----------
        regression_model
            A statsmodels regression model class, either OLS or GLM. Defaults to OLS.

        **kwargs
            Additional arguments for fitting the specific method. In particular, this
            is where you can specify the family for GLM.

        Example
        -------
        >>> import statsmodels.api as sm
        >>> model = StatsmodelsDE(adata, design="~condition")
        >>> model.fit(sm.GLM, family=sm.families.NegativeBinomial(link=sm.families.links.Log()))
        >>> results = model.test_contrasts(np.array([0, 1]))
        """
        self.models = []
        for var in tqdm(self.adata.var_names):
            mod = regression_model(
                sc.get.obs_df(self.adata, keys=[var], layer=self.layer)[var],
                self.design,
                **kwargs,
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

class DESeq2DE(BaseMethod):
    """Differential expression test using DESeq2 (R/BioC implementation)"""

    def fit(self) -> DataFrame:
        '''
        Run differential expression using DESeq2.
        '''

        ## Get anndata components
        data_X = self.adata.X.toarray().copy()
        data_obs = self.adata.obs.copy()
        data_vars = self.adata.var_names.copy()

        ## Set up the R code
        deseq2_fit_str = f'''
            library(DESeq2)
            
            fit_deseq <- function(args){{
                data_X <- t(args[[1]])
                data_obs <- args[[2]]
                rownames(data_X) <- args[[3]]
                
                design <- as.matrix(args[[4]])

                # Prepare the data
                deseq_data <- DESeqDataSetFromMatrix(countData = data_X,
                                                    colData = data_obs,
                                                    design = design)

                # Fit the DESeq2 model
                deseq_model <- DESeq(deseq_data)
                return(deseq_model)
                }}
            '''

        r_pkg = STAP(deseq2_fit_str, "r_pkg")
        
        # Run DE
        de_model = r_pkg.fit_deseq([data_X, data_obs, data_vars, self.design])
        self.de_model = de_model

    def _test_single_contrast(self, contrast, **kwargs) -> DataFrame:
        '''
        Run differential expression using DESeq2.
        '''

        deseq2_test_str = f'''
            library(DESeq2)
            test_deseq <- function(args){{
                deseq_model <- args[[1]]
                contrast <- args[[2]]
                # Make the comparison
                deseq_result <- as.data.frame( results(deseq_model, contrast=contrast) )

                #print(deseq_result[1:5,])  For debugging
                #print(rownames(deseq_result)[1:5]) For debugging

                return(deseq_result)
            }}
            '''
        r_pkg = STAP(deseq2_test_str, "r_pkg")
        de_res_df = r_pkg.test_deseq([self.de_model, contrast])

        with (ro.default_converter + pandas2ri.converter).context():
            pd_result = ro.conversion.get_conversion().rpy2py(de_res_df)

        pd_result["var_name"] = pd_result.index
        pd_result = pd_result.reset_index(drop=True)
        
        return pd_result
        return pd.DataFrame(res).sort_values("pvalue").set_index("variable")
      

class EdgeRDE(BaseMethod):
    """Differential expression test using EdgeR"""

    def fit(self, **kwargs): #adata, design, mask, layer
        '''
        Fit model using edgeR. Note: this creates its own adata object for downstream. 

        Params:
        ----------
        **kwargs
            Keyword arguments specific to glmQLFit()
        '''
        
        ## For running in notebook
        #pandas2ri.activate()
        #rpy2.robjects.numpy2ri.activate()
        
        ## -- Check installations
        try:
            import rpy2.robjects.pandas2ri
            import rpy2.robjects.numpy2ri
            from rpy2.robjects.packages import importr
            from rpy2.robjects import pandas2ri, numpy2ri
            from rpy2.robjects.conversion import localconverter
            from rpy2 import robjects as ro
            
            pandas2ri.activate()
            rpy2.robjects.numpy2ri.activate()
            
        except ImportError:
            raise ImportError("edger requires rpy2 to be installed. ")
            
        try:
            base = importr("base")
            edger = importr("edgeR")
            stats = importr("stats")
            limma = importr("limma")
            blasctl = importr("RhpcBLASctl")
            bcparallel = importr("BiocParallel")
        except ImportError:
            raise ImportError(
                    "edgeR requires a valid R installation with the following packages: "
                    "edgeR, BiocParallel, RhpcBLASctl"
                )

        ## -- Feature selection
        #if mask is not None:
        #    self.adata = self.adata[:,~self.adata.var[mask]]
        
        ## -- Convert dataframe
        with localconverter(ro.default_converter + numpy2ri.converter):
            expr = self.adata.X if self.layer is None else self.adata.layers[self.layer]
            if issparse(expr):
                expr = expr.T.toarray()
            else:
                expr = expr.T

        expr_r = ro.conversion.py2rpy(pd.DataFrame(expr, 
                                                   index=self.adata.var_names, 
                                                   columns=self.adata.obs_names))

        ## -- Convert to DGE
        dge = edger.DGEList(counts=expr_r, 
                            samples=self.adata.obs)
        
        ## -- Run EdgeR
        logging.info("Calculating NormFactors")
        dge = edger.calcNormFactors(dge)

        logging.info("Estimating Dispersions")
        dge = edger.estimateDisp(dge, design=self.design)

        logging.info("Fitting linear model")
        fit = edger.glmQLFit(dge, design=self.design, **kwargs)

        ## -- Save object
        ro.globalenv["fit"] = fit
        #self.adata.uns["fit"] = fit
        self.fit = fit
        
        
        
    def _test_single_contrast(self, contrast: List[str]) -> pd.DataFrame:
        """
        Conduct test for each contrast and return a data frame

        Parameters
        ----------
        contrast:
            numpy array of integars indicating contrast
            i.e. [-1, 0, 1, 0, 0]
        """
        
        ## For running in notebook
        #pandas2ri.activate()
        #rpy2.robjects.numpy2ri.activate()
        
        ## -- To do:
        ##  parse **kwargs to R function
        ##  Fix mask for .fit()
        
        ## -- Check installations
        try:
            import rpy2.robjects.pandas2ri
            import rpy2.robjects.numpy2ri
            from rpy2.robjects.packages import importr
            from rpy2.robjects import pandas2ri, numpy2ri
            from rpy2.robjects.conversion import localconverter
            from rpy2 import robjects as ro
            
        except ImportError:
            raise ImportError("edger requires rpy2 to be installed. ")
            
        try:
            base = importr("base")
            edger = importr("edgeR")
            stats = importr("stats")
            limma = importr("limma")
            blasctl = importr("RhpcBLASctl")
            bcparallel = importr("BiocParallel")
        except ImportError:
            raise ImportError(
                    "edgeR requires a valid R installation with the following packages: "
                    "edgeR, BiocParallel, RhpcBLASctl"
                )

        ## -- Get fit object
        fit = self.fit
        
        ## -- Convert vector to R
        contrast_vec_r = ro.conversion.py2rpy(np.asarray(contrast))
        ro.globalenv["contrast_vec"] = contrast_vec_r
        
        ## -- Test contrast with R
        ro.r(
            """
            test = edgeR::glmQLFTest(fit, contrast=contrast_vec)
            de_res =  edgeR::topTags(test, n=Inf, adjust.method="BH", sort.by="PValue")$table 
            """
        )
        
        ## -- Convert results to pandas
        de_res = ro.conversion.rpy2py(ro.globalenv["de_res"])

        return de_res

