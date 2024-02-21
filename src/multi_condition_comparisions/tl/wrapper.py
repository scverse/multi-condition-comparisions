from typing import Any, Literal, TypedDict

import numpy as np
from anndata import AnnData

from multi_condition_comparisions.tl.de import EdgeRDE, PyDESeq2DE, StatsmodelsDE

MethodRegistry: dict[str, Any] = {"DESeq": PyDESeq2DE, "edgeR": EdgeRDE, "statsmodels": StatsmodelsDE}


class Contrast(TypedDict):
    """Contrast typed dict."""

    column: str
    baseline: str
    group_to_compare: str


def run_de(
    adata: AnnData,
    contrasts: list[Contrast],
    method: Literal["DESeq", "edgeR", "statsmodels"],
    design: str | np.ndarray | None = None,
    mask: str | None = None,
    layer: str | None = None,
    **kwargs,
):
    """
    Wrapper function to run differential expression analysis.

    Params:
    ----------
    adata
        AnnData object, usually pseudobulked.
    contrasts
        Contrasts to perform.
    method
        Method to perform DE.
    design (optional)
        Model design. Can be either a design matrix, a formulaic formula. If None, contrasts should be provided.
    mask (optional)
        A column in adata.var that contains a boolean mask with selected features.
    layer (optional)
        Layer to use in fit(). If None, use the X matrix.
    **kwargs
        Keyword arguments specific to the method implementation.
    """
    ## Initialise object
    model = MethodRegistry[method](adata, design, mask=mask, layer=layer)

    ## Fit model
    model.fit(**kwargs)

    ## Test contrasts
    de_res = model.test_contrasts(np.vstack([model.contrast(**contrast) for contrast in contrasts]), **kwargs)

    return de_res
