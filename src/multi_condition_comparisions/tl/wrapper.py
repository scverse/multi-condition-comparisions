from typing import List

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


def run_de(adata: AnnData,
           contrasts: str | List[str] | dict[str, np.ndarray],
           method: str,
           design: str | np.ndarray | None = None,
           mask: str | None = None,
           layer: str | None = None, **kwargs):
    '''
    Wrapper function to run differential expression analysis.

    Params:
    ----------
    adata
        AnnData object, usually pseudobulked.
    contrasts
        Columns of .obs to perform contrasts with.
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
    '''
    
    
    ## TODO: Extract design matrix with Cond function
    if design is not None:
        design = 
        
    ## TODO: Extract contrasts based on Cond function
    if not isinstance(contrasts, dict):
        contrasts = {} 

    
    ## TODO: Get pseudobulk adata with pseudobulk function
    pb_adata =
    
    ## Initialise object
    pb_adata = BaseMethod(pb_adata, design, mask = mask, layer= layer)
    
    ## Fit model
    pb_adata.fit(**kwargs)
    
    ## Test contrasts
    de_res = pb_adata.test_contrasts(contrasts, **kwargs)

    
    ## TODO: Standardise column names
    
    
    return de_res

