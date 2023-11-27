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
    Run differential expression using DESeq2.

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

    ## Get anndata components
    data_X = adata.X.toarray().copy()
    data_obs = adata.obs.copy()
    data_var = adata.var.copy()

    ## Set up the R code
    deseq2_str = f'''
        library(DESeq2)
        
        run_deseq <- function(args){{
            # Prepare the data
            deseq_data <- DESeqDataSetFromMatrix(countData = data_X,
                                                colData = data_obs,
                                                design = design)

            # Fit the DESeq2 model
            deseq_model <- DESeq(deseq_data)

            # Make the comparison
            deseq_result <- results(deseq_model, contrast=contrast, format="DataFrame")

            return(deseq_result)
        }}
        '''

    r_pkg = STAP(deseq2_str, "r_pkg")
   
    # this was needed for the code to run on jhub
    # if you have a different version of rpy2 you may not need these two lines
    rpy2.robjects.pandas2ri.activate()
    rpy2.robjects.numpy2ri.activate()
    
    out_filename = './de_results_DESeq2.csv'

    # Run DE
    args = [data_X, data_obs]
    de_res_df = r_pkg.run_de(args)

    # Relplace the index with the gene namess
    de_res_df = pd.read_csv(out_filename, index_col=0)
    de_res_df.index = de_res_df['gene_name']
    de_res_df.drop('name', axis=1, inplace=True)
    os.remove(out_filename)
    return(de_res_df)