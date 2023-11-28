from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd
from formulaic import model_matrix
import formulaic.model_matrix
from anndata import AnnData


class MethodBase(ABC):
    def __init__(self, adata: AnnData, design: str | np.ndarray, mask: str, layer: str | None = None, **kwargs):
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
        self.adata = adata[:, adata.var[mask]]
        self.layer = layer
        if isinstance(design, str):
            self.design = model_matrix(design, adata.obs)
        else:
            self.design = design

    @abstractmethod
    def fit(self, **kwargs) -> None:
        """
        Fit the model

        Parameters
        ----------
        **kwargs
            Additional arguments for fitting the specific method.
        """
        pass

    @abstractmethod
    def _test_single_contrast(self, contrast, **kwargs) -> pd.DataFrame:
        pass

    def test_contrast(self, contrasts: dict[str, str] | str, **kwargs) -> pd.DataFrame:
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
        for name, contrast in contrasts.items():
            self._test_single_contrast(contrast, **kwargs)

    def test_reduced(self, modelB: "DeMethod") -> pd.DataFrame:
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
        if not isinstance(self.design, formulaic.model_matrix.ModelMatrix):
            raise RuntimeError(
                "Building contrasts with `cond` only works if you specified the model using a "
                "formulaic formula. Please manually provide a contrast vector."
            )
        # for factor, factor_info in self.design.design_info.factor_infos.items():
        #     pass

        # patsy.build_design_matrices(self.design.design_info, kwargs)
        raise NotImplementedError

    def contrast(self, column: str, baseline: str, group_to_compare: str) -> np.ndarray:
        """
        Build a simple contrast for pairwise comparisons.

        This is equivalent to

        ```
        model.cond(<column> = baseline) - model.cond(<column> = group_to_compare)
        ```
        """
        return self.cond(**{column: baseline}) - self.cond(**{column: group_to_compare})
