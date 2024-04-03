from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from anndata import AnnData

from multi_condition_comparisions._util import check_is_numeric_matrix
from multi_condition_comparisions._util.formulaic import Factor, get_factor_storage_and_materializer


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
        Initialize the method.

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

        # Check after mask has been applied.
        check_is_numeric_matrix(self.data)

    @property
    def data(self):
        """Get the data matrix from anndata this object was initalized with (X or layer)."""
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
        baseline: str,
        groups_to_compare: str | Sequence[str] | None,
        *,
        paired_by: str | None = None,
        mask: str | None = None,
        layer: str | None = None,
        fit_kwargs: Mapping = MappingProxyType({}),
        test_kwargs: Mapping = MappingProxyType({}),
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
            baseline value (one category from variable).
        groups_to_compare
            One or multiple categories from variable to compare against baseline. Setting this to None refers to
            "all other categories"
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

        self.materializer = None
        self.factor_storage = None
        """Object to store metadata of formulaic factors which is used for building contrasts later. If a design matrix
        is passed directly, this remains None."""

        if isinstance(design, str):
            self.factor_storage, materializer_class = get_factor_storage_and_materializer()
            self.materializer = materializer_class(adata.obs)
            self.design = self.materializer.get_model_matrix(design)
            self.materializer.stop_recording()
        else:
            self.design = design

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
        fit_kwargs: Mapping = MappingProxyType({}),
        test_kwargs: Mapping = MappingProxyType({}),
    ) -> pd.DataFrame:
        """
        Compare between groups in a specified column.

        This is a high-level interface that is kept simple on purpose and
        only supports comparisons between groups on a single column at a time.
        For more complex designs, please use the LinearModel method classes directly.

        Parameters
        ----------
        adata
            AnnData object
        column
            column in obs that contains the grouping information
        baseline
            baseline value (one category from variable).
        groups_to_compare
            One or multiple categories from variable to compare against baseline. Setting this to None refers to
            "all other categories"
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
        design = f"~{column}"
        if paired_by is not None:
            design += f"+{paired_by}"
        if isinstance(groups_to_compare, str):
            groups_to_compare = [groups_to_compare]
        model = cls(adata, design=design, mask=mask, layer=layer)

        model.fit(**fit_kwargs)

        de_res = model.test_contrasts(
            {
                group_to_compare: model.contrast(column=column, baseline=baseline, group_to_compare=group_to_compare)
                for group_to_compare in groups_to_compare
            },
            **test_kwargs,
        )

        return de_res

    @property
    def variables(self) -> set:
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
    def _test_single_contrast(self, contrast: Sequence[float], **kwargs) -> pd.DataFrame: ...

    def test_contrasts(self, contrasts: Sequence[float] | dict[str, Sequence[float]], **kwargs) -> pd.DataFrame:
        """
        Perform a comparison as specified in a contrast vector.

        The contrast vector is a numeric vector with one element for each column in the design matrix.
        We recommend building the contrast vector using {func}`~LinearModelBase.contrast` or
        {func}`~LinearModelBase.cond`.

        Multiple comparisons can be specified as a dictionary of contrast vectors.

        Parameters
        ----------
        contrasts:
            Either a numeric contrast vector, or a dictionary of numeric contrast vectors. The dictionary keys
            are added in a column named `contrast` in the result dataframe.
        kwargs
            are passed to the respective implementation
        """
        if not isinstance(contrasts, dict):
            contrasts = {None: contrasts}
        results = []
        for name, contrast in contrasts.items():
            results.append(self._test_single_contrast(contrast, **kwargs).assign(contrast=name))

        results_df = pd.concat(results)

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
        Get a contrast vector representing a specific condition.

        The resulting contrast vector can be combined using arithmetic operations to generate a contrast
        vector for a specific comparison, e.g. for a simple comparison treated vs. control

        >>> contrast = model.cond(condition="treated") - model.cond(condition="ctrl")

        When using an interaction term `cell * condition`, a comparison within a specific cell-type can be
        easily built as follows:

        >>> contrast = model.cond(cell="B cells", condition="ctrl") - cond(cell="B cells", condition="treated")

        This way of building contrast vectors is inspired by [glmGamPoi](https://bioconductor.org/packages/release/bioc/html/glmGamPoi.html).

        Parameters
        ----------
        **kwargs
            column/value pairs

        Returns
        -------
        A contrast vector that aligns to the columns of the design matrix.
        """
        if self.factor_storage is None:
            raise RuntimeError(
                "Building contrasts with `cond` only works if you specified the model using a "
                "formulaic formula. Please manually provide a contrast vector."
            )

        cond_dict = kwargs

        if not set(cond_dict.keys()).issubset(self.variables):
            raise ValueError(
                "You specified a variable that is not part of the model. Available variables: "
                + ",".join(self.variables)
            )

        # We now fill `cond_dict` such that it is equivalent to a data row from `adata.obs`.
        # This data can then be passed to the `get_model_matrix` of formulaic to retreive a correpsonding
        # contrast vector.
        # To do so, we keep the values that were already specified, and fill all other values with the default category
        # (the one that is usually dropped from the model for being redundant).
        #
        # `model_spec.variable_terms` is a mapping from variable to a set of terms. Unless a variable is used twice in the
        # same formula (which for don't support for now), it contains exactly one element.
        for var, term in self.design.model_spec.variable_terms.items():
            if len(term) != 1:
                raise RuntimeError(
                    "Ambiguous variable! Building contrasts with model.cond only works "
                    "when each variable occurs only once per formula"
                )
            term = term.pop()
            term_metadata = self.factor_storage[term]
            if var in cond_dict:
                # In this case we keep the specified value in the dict, but we verify that it's a valid category
                if term_metadata.kind == Factor.Kind.CATEGORICAL and cond_dict[var] not in term_metadata.categories:
                    raise ValueError(
                        f"You specified a non-existant category for {var}. Possible categories: {', '.join(term_metadata.categories)}"
                    )
            else:
                # fill with default values
                if term_metadata.kind == Factor.Kind.CATEGORICAL:
                    cond_dict[var] = term_metadata.base
                else:
                    cond_dict[var] = 0

        df = pd.DataFrame([kwargs])
        return self.design.model_spec.get_model_matrix(df).iloc[0]

    def contrast(self, column: str, baseline: str, group_to_compare: str) -> pd.Series:
        """
        Build a simple contrast for pairwise comparisons.

        This is an alias for

        >>> model.cond(column=group_to_compare) - model.cond(column=baseline)

        Parameters
        ----------
        column
            column in adata.obs to test on
        baseline
            baseline category (denominator)
        group_to_compare
            category to compare against baseline (nominator)

        Returns
        -------
        Numeric contrast vector
        """
        return self.cond(**{column: group_to_compare}) - self.cond(**{column: baseline})
