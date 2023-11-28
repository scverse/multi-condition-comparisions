from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd
import patsy
from anndata import AnnData


class DeMethod(ABC):
    def __init__(self, adata: AnnData, design: str | np.ndarray | Literal["patsy"], mask: str):
        """
        Initialize the method

        Parameters
        ----------
        adata
            AnnData object, usually pseudobulked.
        design
            Model design. Can be either a design matrix, a patsy formula.
        mask
            a column in adata.var that contains a boolean mask with selected features.
        """
        self.adata = adata
        self.design = patsy.dmmatrix(design, adata.obs)

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
        for factor, factor_info in self.design.design_info.factor_infos.items():
            pass

        patsy.build_design_matrices(self.design.design_info, kwargs)
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
