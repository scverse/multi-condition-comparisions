from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from multi_condition_comparisions._util import check_is_numeric_matrix
from multi_condition_comparisions._util.formulaic import (
    AmbiguousAttributeError,
    Factor,
    FactorMetadata,
    get_factor_storage_and_materializer,
    resolve_ambiguous,
)


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

        self.factor_storage: dict[str, list[FactorMetadata]] = None
        """Object to store metadata of formulaic factors which is used for building contrasts later. If a design matrix
        is passed directly, this remains None."""

        self.variable_to_factors: dict[str, set[str]] = None
        """Stores mapping from variables to formulaic factors"""

        if isinstance(design, str):
            # Use custom formulaic materializer that will record factor metadata while creating the model matrix
            self.factor_storage, self.variable_to_factors, materializer_class = get_factor_storage_and_materializer()
            self.design = materializer_class(adata.obs, record_factor_metadata=True).get_model_matrix(design)
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
        # self.variables only contains "data-variables", but no "factor variables", such as `C`
        for var in self.variables:
            # If the variable is specified in the cond_dict explicitly, we just keep it as is.
            # We still verify that it's a valid category, otherwise simple typos are not caught and lead to
            # zeros in the design matrix.
            if var in cond_dict:
                self._check_category(var, cond_dict[var])

            # If the variable is not specified, we want to fill it with its default value (i.e. the base category)
            else:
                cond_dict[var] = self._get_default_value(var)

        df = pd.DataFrame([kwargs])
        return self.design.model_spec.get_model_matrix(df).iloc[0]

    def _get_factor_metadata_for_variable(self, var) -> list[FactorMetadata]:
        """
        Get the Metadata objects of all factors defined by a given variable.

        A variable can refer to one or multiple factors. Either if a variable is specified
        multiple times in the model (e.g. ~ var + C(var); ~ continuous + np.log(continuous))
        or when there's an interaction term (e.g. ~ A * B ==> terms `A`, `B`, `A:B`).

        Some Factors can have multiple metadata objects, because they are specified
        multiple times in the formula, or forumlaic resolves them multiple times internally.

        Even if there are multiple factors per variable, they could still contain the same metadata and we
        can unambiguously retreive metadata for a given variable, e.g. using the `_util.formulaic.resolve_ambiguous`
        function.

        Returns
        -------
        a list of FactorMetadata objects
        """
        factors = self.variable_to_factors[var]

        return list(chain.from_iterable(self.factor_storage[f] for f in factors))

    def _get_default_value(self, var) -> Any:
        r"""
        Get the default value (base category) for variable `var`.

        Special cases:
         * If the variable is continuous, return 0.
         * For a variable without reduced rank (as it happens for the first variable in a no-intercept model `~0 + xxx`)
           the function returns the null string "\0". This is to ensure that when passed to `formulaic.get_model_matrix`,
           a row is returned that sets all columns of that variable to 0. It could be any string that is not
           a valid category. It cannot be `None`, because this results in failure of `get_model_matrix`, therefore the
           null string was chosen because of clear semantics and it being surely not a valid category.

        Raises
        ------
        ValueError
            If there is no way to unambiguously infer the base category from the formulaic formula.

        Returns
        -------
        The default value of the given variable.
        """
        factor_metadata = self._get_factor_metadata_for_variable(var)
        if resolve_ambiguous(factor_metadata, "kind") == Factor.Kind.CATEGORICAL:
            # In this case it can be ambiguous for some formulas -> Tell the user to specify the variable explicitly
            try:
                tmp_base = resolve_ambiguous(factor_metadata, "base")
            except AmbiguousAttributeError as e:
                raise ValueError(
                    f"Could not automatically resolve base category for variable {var}. Please specify it explicity in `model.cond`."
                ) from e

            # if tmp_base is None (no-intercept model), set it to the NUL string \0.
            # In principle, it could be any string that is not a valid category, but it cannot be None
            # because this leads to an error in `get_model_matrix`
            return tmp_base if tmp_base is not None else "\0"
        else:
            # Set to zero for continuous variables
            return 0

    def _check_category(self, var, value) -> None:
        """
        Check if `value` is a valid category of the factor defined by `var`.

        If `var` is a continuous variable this passes silently.

        Raises
        ------
        ValueError
            If `value` is not a valid category.
        """
        factor_metadata = self._get_factor_metadata_for_variable(var)

        # Getting the categories should never be non-ambiguous. If it happens, it's an edge case we don't know about (-> let it fail)
        tmp_categories = resolve_ambiguous(factor_metadata, "categories")
        if resolve_ambiguous(factor_metadata, "kind") == Factor.Kind.CATEGORICAL and value not in tmp_categories:
            raise ValueError(
                f"You specified a non-existant category for {var}. Possible categories: {', '.join(tmp_categories)}"
            )

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
