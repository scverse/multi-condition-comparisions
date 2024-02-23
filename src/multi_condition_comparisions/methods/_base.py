import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from anndata import AnnData
from formulaic import model_matrix
from formulaic.model_matrix import ModelMatrix
from scipy.sparse import issparse, spmatrix


@dataclass
class Contrast:
    """Simple contrast for comparison between groups"""

    column: str
    baseline: str
    group_to_compare: str


ContrastType = Contrast | tuple[str, str, str]


class MethodBase(ABC):
    @abstractmethod
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
    ) -> pd.DataFrame:
        ...
        """
        Compare between groups in a specified column.

        This interface is expected to be provided by all methods. Methods can provide other interfaces
        on top, see e.g. {class}`LinearModelBase`.

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


class LinearModelBase(MethodBase):
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
    ) -> pd.DataFrame:
        raise NotImplementedError

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
    def _test_single_contrast(self, contrast, **kwargs) -> pd.DataFrame:
        ...

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
        results = []
        for name, contrast in contrasts.items():
            results.append(self._test_single_contrast(contrast, **kwargs).assign(contrast=name))

        results_df = pd.concat(results)
        results_df.rename(
            columns={"pvalue": "pvals", "padj": "pvals_adj", "log2FoldChange": "logfoldchanges"}, inplace=True
        )

        return results_df

    def test_reduced(self, modelB: "MethodBase") -> pd.DataFrame:
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

        >>> res < -test_de(
        ...     fit, contrast=cond(cell="B cells", condition="stim") - cond(cell="B cells", condition="ctrl")
        ... )

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
