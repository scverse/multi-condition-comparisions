import re
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import statsmodels
import statsmodels.api as sm
from anndata import AnnData
from formulaic import model_matrix
from formulaic.model_matrix import ModelMatrix
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from scanpy import logging
from scipy.sparse import issparse, spmatrix
from tqdm.auto import tqdm


class BaseMethod(ABC):
    """Base method for DE testing that is implemented per DE test."""

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
            Model design. Can be either a design matrix, a formulaic formula.Formulaic formula in the format 'x + z' or '~x+z'.
        mask
            A column in adata.var that contains a boolean mask with selected features.
        layer
            Layer to use in fit(). If None, use the X array.
        **kwargs
            Keyword arguments specific to the method implementation
        """
        self.adata = adata
        if mask is not None:
            self.adata = self.adata[:, self.adata.var[mask]]

        # Do some sanity checks on the input. Do them after the mask is applied.

        # Check that counts have no NaN or Inf values.
        if np.any(np.logical_or(adata.X < 0, np.isnan(self.adata.X))) or np.any(np.isinf(self.adata.X)):
            raise ValueError("Counts cannot contain negative, NaN or Inf values.")
        # Check that counts have numeric values.
        if not np.issubdtype(self.adata.X.dtype, np.number):
            raise ValueError("Counts must be numeric.")

        self._check_counts()

        self.layer = layer
        if isinstance(design, str):
            self.design = model_matrix(design, adata.obs)
        else:
            self.design = design

    def _check_count_matrix(self, array: np.ndarray | spmatrix, tolerance: float = 1e-6) -> bool:
        if issparse(array):
            if not array.data.dtype.kind == "i":
                raise ValueError("Non-zero elements of the matrix must be integers.")

            if not np.all(np.abs(array.data - np.round(array.data)) < tolerance):
                raise ValueError("Non-zero elements of the matrix must be close to integer values.")
        else:
            if not array.dtype.kind == "i" or not np.all(np.abs(array - np.round(array)) < tolerance):
                raise ValueError("Matrix must be a count matrix.")
        if (array < 0).sum() > 0:
            raise ValueError("Non.zero elements of the matrix must be postiive.")

        return True

    @property
    def variables(self):
        """Get the names of the variables used in the model definition"""
        return self.design.model_spec.variables_by_source["data"]

    @abstractmethod
    def _check_counts(self) -> bool:
        """
        Check that counts are valid for the specific method.

        Different methods may have different requirements.

        Returns
        -------
        bool
            True if counts are valid, False otherwise.
        """
        ...

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
    def _test_single_contrast(self, contrast, **kwargs) -> pd.DataFrame: ...

    def test_contrasts(self, contrasts: list[str] | dict[str, np.ndarray] | np.ndarray, **kwargs) -> pd.DataFrame:
        """
        Conduct a specific test.  Please use :method:`contrast` to build the contrasts instead of building it on your own.

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
        results_list = []
        for name, contrast in contrasts.items():
            results_list.append(self._test_single_contrast(contrast, **kwargs).assign(contrast=name))

        results = pd.concat(results_list)
        results.rename(
            columns={"pvalue": "pvals", "padj": "pvals_adj", "log2FoldChange": "logfoldchanges"}, inplace=True
        )

        return results_df

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

        # TODO this is hacky - reach out to formulaic authors how to do this properly
        def _get_var_from_colname(colname):
            regex = re.compile(r"^.+\[T\.(.+)\]$")
            return regex.search(colname).groups()[0]

        if not isinstance(self.design, ModelMatrix):
            raise RuntimeError(
                "Building contrasts with `cond` only works if you specified the model using a "
                "formulaic formula. Please manually provide a contrast vector."
            )
        cond_dict = kwargs
        for var in self.variables:
            var_type = self.design.model_spec.encoder_state[var][0].value
            if var_type == "categorical":
                all_categories = set(self.design.model_spec.encoder_state[var][1]["categories"])
            if var in kwargs:
                if var_type == "categorical" and kwargs[var] not in all_categories:
                    raise ValueError(
                        f"You specified a non-existant category for {var}. Possible categories: {', '.join(all_categories)}"
                    )
            else:
                # fill with default values
                if var_type != "categorical":
                    cond_dict[var] = 0
                else:
                    var_cols = self.design.columns[self.design.columns.str.startswith(f"{var}[")]

                    present_categories = {_get_var_from_colname(x) for x in var_cols}
                    dropped_category = all_categories - present_categories
                    assert len(dropped_category) == 1
                    cond_dict[var] = next(iter(dropped_category))

        df = pd.DataFrame([kwargs])

        return self.design.model_spec.get_model_matrix(df)

    def contrast(self, column: str, baseline: str, group_to_compare: str) -> object:
        """Build a simple contrast for pairwise comparisons.  In the future all methods should be able to accept the output of :method:`StatsmodelsDE.contrast` but alas a big TODO."""
        return [column, baseline, group_to_compare]


class StatsmodelsDE(BaseMethod):
    """Differential expression test using a statsmodels linear regression"""

    def _check_counts(self) -> bool:
        return self._check_count_matrix(self.adata.X)

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
        for var, mod in zip(tqdm(self.adata.var_names), self.models, strict=False):
            t_test = mod.t_test(contrast)
            res.append(
                {
                    "variable": var,
                    "pvalue": t_test.pvalue,
                    "tvalue": t_test.tvalue.item(),
                    "sd": t_test.sd.item(),
                    "logfoldchanges": t_test.effect.item(),
                    "padj": statsmodels.stats.multitest.fdrcorrection(np.array([t_test.pvalue]))[1].item(),
                }
            )
        return pd.DataFrame(res).sort_values("pvalue").set_index("variable")

    def contrast(self, column: str, baseline: str, group_to_compare: str) -> np.ndarray:
        """
        Build a simple contrast for pairwise comparisons.

        This is equivalent to

        ```
        model.cond(<column> = baseline) - model.cond(<column> = group_to_compare)
        ```
        """
        return self.cond(**{column: baseline}) - self.cond(**{column: group_to_compare})


class PyDESeq2DE(BaseMethod):
    """Differential expression test using a PyDESeq2"""

    def _check_counts(self) -> bool:
        return self._check_count_matrix(self.adata.X)

    def fit(self, **kwargs) -> pd.DataFrame:
        """
        Fit dds model using pydeseq2.

        Note: this creates its own AnnData object for downstream processing.

        Params:
        ----------
        **kwargs
            Keyword arguments specific to DeseqDataSet()
        """
        inference = DefaultInference(n_cpus=3)
        covars = self.design.columns.tolist()
        if "Intercept" not in covars:
            warnings.warn(
                "Warning: Pydeseq is hard-coded to use Intercept, please include intercept into the model", stacklevel=2
            )
        processed_covars = [col.split("[")[0] for col in covars if col != "Intercept"]
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
        stat_res = DeseqStats(self.dds, contrast=contrast, alpha=alpha, **kwargs)
        stat_res.summary()
        stat_res.p_values
        return pd.DataFrame(stat_res.results_df).sort_values("padj")


class EdgeRDE(BaseMethod):
    """Differential expression test using EdgeR"""

    def _check_counts(self) -> bool:
        return self._check_count_matrix(self.adata.X)

    def fit(self, **kwargs):  # adata, design, mask, layer
        """
        Fit model using edgeR.

        Note: this creates its own adata object for downstream.

        Params:
        ----------
        **kwargs
            Keyword arguments specific to glmQLFit()
        '''

        ## -- Check installations
        """
        # For running in notebook
        # pandas2ri.activate()
        # rpy2.robjects.numpy2ri.activate()

        try:
            import rpy2.robjects.numpy2ri
            import rpy2.robjects.pandas2ri
            from rpy2 import robjects as ro
            from rpy2.robjects import numpy2ri, pandas2ri
            from rpy2.robjects.conversion import localconverter
            from rpy2.robjects.packages import importr

            pandas2ri.activate()
            rpy2.robjects.numpy2ri.activate()

        except ImportError:
            raise ImportError("edger requires rpy2 to be installed. ") from None

        try:
            edger = importr("edgeR")
        except ImportError:
            raise ImportError(
                "edgeR requires a valid R installation with the following packages: " "edgeR, BiocParallel, RhpcBLASctl"
            ) from None

        ## -- Convert dataframe
        # Feature selection
        # if mask is not None:
        #    self.adata = self.adata[:,~self.adata.var[mask]]

        # Convert dataframe
        with localconverter(ro.default_converter + numpy2ri.converter):
            expr = self.adata.X if self.layer is None else self.adata.layers[self.layer]
            if issparse(expr):
                expr = expr.T.toarray()
            else:
                expr = expr.T

        expr_r = ro.conversion.py2rpy(pd.DataFrame(expr, index=self.adata.var_names, columns=self.adata.obs_names))

        # Convert to DGE
        dge = edger.DGEList(counts=expr_r, samples=self.adata.obs)

        # Run EdgeR
        logging.info("Calculating NormFactors")
        dge = edger.calcNormFactors(dge)

        logging.info("Estimating Dispersions")
        dge = edger.estimateDisp(dge, design=self.design)

        logging.info("Fitting linear model")
        fit = edger.glmQLFit(dge, design=self.design, **kwargs)

        # Save object
        ro.globalenv["fit"] = fit
        self.fit = fit

    def _test_single_contrast(self, contrast: list[str], **kwargs) -> pd.DataFrame:
        """
        Conduct test for each contrast and return a data frame

        Parameters
        ----------
        contrast:
            numpy array of integars indicating contrast
            i.e. [-1, 0, 1, 0, 0]
        """
        ## -- Check installations
        # For running in notebook
        # pandas2ri.activate()
        # rpy2.robjects.numpy2ri.activate()

        # ToDo:
        #  parse **kwargs to R function
        #  Fix mask for .fit()

        try:
            import rpy2.robjects.numpy2ri
            import rpy2.robjects.pandas2ri  # noqa: F401
            from rpy2 import robjects as ro
            from rpy2.robjects import numpy2ri, pandas2ri  # noqa: F401
            from rpy2.robjects.conversion import localconverter  # noqa: F401
            from rpy2.robjects.packages import importr

        except ImportError:
            raise ImportError("edger requires rpy2 to be installed.") from None

        try:
            importr("edgeR")
        except ImportError:
            raise ImportError(
                "edgeR requires a valid R installation with the following packages: " "edgeR, BiocParallel, RhpcBLASctl"
            ) from None

        # Convert vector to R, which drops a category like `self.design_matrix` to use the intercept for the left out.
        contrast_vec = [0] * len(self.design.columns)
        make_contrast_column_key = lambda ind: f"{contrast[0]}[T.{contrast[ind]}]"
        for index in [1, 2]:
            if make_contrast_column_key(index) in self.design.columns:
                contrast_vec[self.design.columns.to_list().index(f"{contrast[0]}[T.{contrast[index]}]")] = 1
        contrast_vec_r = ro.conversion.py2rpy(np.asarray(contrast_vec))
        ro.globalenv["contrast_vec"] = contrast_vec_r

        # Test contrast with R
        ro.r(
            """
            test = edgeR::glmQLFTest(fit, contrast=contrast_vec)
            de_res =  edgeR::topTags(test, n=Inf, adjust.method="BH", sort.by="PValue")$table
            """
        )

        # Convert results to pandas
        de_res = ro.conversion.rpy2py(ro.globalenv["de_res"])

        return de_res


class BaseTwoSampleNonParametricTest(BaseMethod):
    name: str

    def test_method(self, *args, **kwargs):
        """`test_method` must be implemented as a non-parametric, two-sample test, with an interface like `Callable[[np.ndarray, np.ndarray, ...], object]`. See e.g., https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html"""
        raise NotImplementedError(
            "`test_method` must be implemented as a non-parametric, two-sample test, with an interface like `Callable[[np.ndarray, np.ndarray, ...], object]`. See e.g., https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html"
        )

    def _check_counts(self) -> bool:
        return True  # later? is this correct?

    def fit(self):
        """`fit` method here is only provided for API completeness and does nothing."""
        warnings.warn(f"There is nothing to fit in a {self.name} test.", stacklevel=2)
        pass

    def _test_single_contrast(self, contrast, **kwargs) -> pd.DataFrame:
        res = []
        if len(contrast) != 3:
            raise ValueError("Contrast can only have three elements for Mann-Whitney U test.")
        for var in tqdm(self.adata.var_names):
            adata0 = self.adata[self.adata.obs[contrast[0]] == contrast[1], var].copy()
            adata1 = self.adata[self.adata.obs[contrast[0]] == contrast[2], var].copy()

            x0 = adata0.X if self.layer is None else adata0.layers[self.layer]
            x1 = adata1.X if self.layer is None else adata1.layers[self.layer]
            pval = self.test_method(
                x=np.asarray(x0.todense()).flatten() if scipy.sparse.issparse(x0) else x0.flatten(),
                y=np.asarray(x1.todense()).flatten() if scipy.sparse.issparse(x1) else x1.flatten(),
                **kwargs,
            ).pvalue
            mean_x0 = np.asarray(np.mean(x0, axis=0)).flatten().astype(dtype=float)
            mean_x1 = np.asarray(np.mean(x1, axis=0)).flatten().astype(dtype=float)
            res.append(
                {
                    "variable": var,
                    "pvalue": pval,
                    "fold_change": np.log(mean_x1) - np.log(mean_x0),
                }
            )
        return pd.DataFrame(res).sort_values("pvalue").set_index("variable")


class MannWhitneyUTest(BaseTwoSampleNonParametricTest):
    """Mann-Whitney U Test via scipy's :func:`~scipy.stats.mannwhitneyu`."""

    name: str = "Mann-Whitney U Test"

    def test_method(self, *args, **kwargs):
        return scipy.stats.mannwhitneyu(*args, **kwargs)


class WilcoxonTest(BaseTwoSampleNonParametricTest):
    """Wilcoxon Test via scipy's :func:`~scipy.stats.mannwhitneyu`."""

    name: str = "Wilcoxon Test"

    def test_method(self, *args, **kwargs):
        return scipy.stats.wilcoxon(*args, **kwargs)
