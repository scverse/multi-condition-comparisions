from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData


class DeMethod(ABC):
    def __init__(self, adata: AnnData):
        """
        Initialize the method

        Parameters
        ----------
        adata
            AnnData object, usually pseudobulked
        """
        self.adata = adata

    @abstractmethod
    def fit(self, design: str | np.ndarray | Literal["patsy"], mask: str, **kwargs) -> None:
        """
        Fit the model

        Parameters
        ----------
        design
            Model design. Can be either a design matrix, a patsy formula
        mask
            a column in adata.var that contains a boolean mask with selected features
        **kwargs
            Additional arguments for the specific method
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
        """
        for name, contrast in contrasts.items():
            self._test_single_contrast(contrast, **kwargs)

    def test_reduced(self, modelB: "DeMethod") -> pd.DataFrame:
        """
        Test against a reduced model

        Example:
        --------
        ```
        modelA = Model().fit()
        modelB = Model().fit()
        modelA.test_reduced(modelB)
        ```
        """
        raise NotImplementedError
