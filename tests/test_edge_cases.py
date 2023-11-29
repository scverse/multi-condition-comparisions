import anndata as ad
import numpy as np
import pytest

from multi_condition_comparisions.tl.de import StatsmodelsDE


@pytest.mark.parametrize("invalid_input", [np.nan, np.inf])
def test_invalid_inputs(invalid_input, test_counts, test_metadata):
    """Check that invalid inputs in MethodBase counts raise an error."""
    test_counts[0, 0] = invalid_input
    adata = ad.AnnData(X=test_counts, obs=test_metadata)
    with pytest.raises(ValueError):
        StatsmodelsDE(adata=adata, design="~condition")
