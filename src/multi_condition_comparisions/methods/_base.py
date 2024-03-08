import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from anndata import AnnData
from formulaic import model_matrix
from formulaic.model_matrix import ModelMatrix


@dataclass
class Contrast:
    """Simple contrast for comparison between groups"""

    column: str
    baseline: str
    group_to_compare: str


ContrastType = Contrast | tuple[str, str, str]


class MethodBase(ABC):
    def __init__(
        self,
        adata: AnnData,
        *,
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

        self.layer = layer

        # Do some sanity checks on the input. Do them after the mask is applied.
        # Check that counts have no NaN or Inf values.
        if np.any(~np.isfinite(self.data)):
            raise ValueError("Counts cannot contain negative, NaN or Inf values.")
        # Check that counts have numeric values.
        if not np.issubdtype(self.adata.X.dtype, np.number):
            raise ValueError("Counts must be numeric.")

    @property
    def data(self):
        """Get the data matrix from anndata this object was initalized with (X or layer)"""
        if self.layer is None:
            return self.adata.X
        else:
            return self.adata.layer[self.layer]

    @classmethod
    @abstractmethod
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
    ) -> pd.DataFrame:
        """
        Compare between groups in a specified column.

        This interface is expected to be provided by all methods. Methods can provide other interfaces
        on top, see e.g. {class}`LinearModelBase`. This is a high-level interface that is kept simple on purpose and
        only supports comparisons between groups on a single column at a time. For more complex designs,
        please use the LinearModel method classes directly.

        Parameters
        ----------
        adata
            AnnData object
        column
            column in obs that contains the grouping information
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
        ...


class LinearModelBase(MethodBase):
    """Base class for DE methods that have a linear model interface (i.e. support design matrices and contrast testing)"""

    def __init__(
        self,
        adata: AnnData,
        design: str | np.ndarray,
        *,
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
        super().__init__(adata, mask=mask, layer=layer)
        self._check_counts()

        if isinstance(design, str):
            self.design = model_matrix(design, adata.obs)
        else:
            self.design = design

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
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "TODO: should be possible to just initialize a model, and build a contrast. `cls` provides access to the current subclass"
        )

    @property
    def variables(self):
        """Get the names of the variables used in the model definition"""
        try:
            return self.design.model_spec.variables_by_source["data"]
        except AttributeError:
            raise ValueError(
                "Retrieving variables is only possible if the model was initialized using a formula."
            ) from None

    @abstractmethod
    def _check_counts(self) -> None:
        """
        Check that counts are valid for the specific method.

        Different methods may have different requirements.

        Raises
        ------
        ValueError
            if the data matrix does not comply with the expectations
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
