from typing import Union, Optional, Sequence, Tuple, Iterable, Dict, Any
import numpy as np
import pandas as pd

from anndata import AnnData
from numpy.typing import ArrayLike


def _run_de_pydeseq2(
        adata: AnnData,
        design: ArrayLike, 
        contrast: ArrayLike,      
    ) -> pd.DataFrame:
    '''
    Run differential expression using pydeseq2.

    Params:
    -------
    adata: AnnData
        Annotated data matrix.
    design: Union[str, ArrayLike]
        Design matrix with the same number of rows as adata.X.
    contrast: ArrayLike
        Binary vector specifying cont

    Returns:
    --------
    pd.DataFrame
        Differential expression results
    '''


def _run_de_deseq2(
        adata: AnnData,
        design: ArrayLike, 
        contrast: str,      
    ) -> pd.DataFrame:
    '''
    Run differential expression using pydeseq2.

    Params:
    -------
    adata: AnnData
        Annotated data matrix.
    design: Union[str, ArrayLike]
        Design. Either a matrix with the same number of rows as adata.X or a string of the design formula, or .
    contrast: str
        Contrast to test. Must be a string of the form 'condition1 - condition2'.

    Returns:
    --------
    pd.DataFrame
        Differential expression results
    '''