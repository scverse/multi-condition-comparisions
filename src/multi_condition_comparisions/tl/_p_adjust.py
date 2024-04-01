import statsmodels.stats
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def fdr_correction(df: pd.DataFrame, pvalue_col="p_value", *, key_added="adj_p_value", inplace=False):
    """Adjust p-values in a DataFrame with test results using FDR correction."""
    if not inplace:
        df = df.copy()

    df[key_added] = statsmodels.stats.multitest.fdrcorrection(df[pvalue_col].values)[1]

    if not inplace:
        return df
