import anndata as ad
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from multi_condition_comparisions._util import check_is_integer_matrix
from multi_condition_comparisions.tl.de import Statsmodels


@pytest.mark.parametrize("invalid_input", [np.nan, np.inf, "foo"])
def test_invalid_inputs(invalid_input, test_counts, test_metadata):
    """Check that invalid inputs in MethodBase counts raise an error."""
    test_counts[0, 0] = invalid_input
    adata = ad.AnnData(X=test_counts, obs=test_metadata)
    with pytest.raises((ValueError, TypeError)):
        Statsmodels(adata=adata, design="~condition")


def test_valid_count_matrix():
    """Test with a valid count matrix."""
    matrix = np.array([[1, 2], [3, 4]])
    check_is_integer_matrix(matrix)


def test_valid_sparse_count_matrix():
    """Test with a valid sparse count matrix."""
    matrix = csr_matrix(np.array([[1, 0], [0, 2]]), dtype="int32")
    check_is_integer_matrix(matrix)


def test_negative_values():
    """Test with a matrix containing negative values."""
    matrix = np.array([[1, -2], [3, 4]])
    with pytest.raises(ValueError):
        check_is_integer_matrix(matrix)


def test_non_integer_values():
    """Test with a matrix containing non-integer values."""
    matrix = np.array([[1.5, 2], [3, 4]])
    with pytest.raises(ValueError):
        check_is_integer_matrix(matrix)


def test_matrix_with_nans():
    """Test with a matrix containing NaNs."""
    matrix = np.array([[1, np.nan], [3, 4]])
    with pytest.raises(ValueError):
        check_is_integer_matrix(matrix)


def test_sparse_matrix_with_nans():
    """Test with a sparse matrix containing NaNs."""
    matrix = csr_matrix([[1, np.nan], [3, 4]])
    with pytest.raises(ValueError):
        check_is_integer_matrix(matrix)
