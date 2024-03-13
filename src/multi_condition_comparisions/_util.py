import numpy as np
from scipy.sparse import issparse, spmatrix


def check_is_integer_matrix(array: np.ndarray | spmatrix, tolerance: float = 1e-6) -> None:
    """Check if a matrix container integers, or floats that are close to integers.

    Parameters
    ----------
    array
        dense or sparse matrix to check
    tolerance
        Values must be this close to ingegers

    Raises
    ------
    ValueError
        if the matrix contains valuese that are not close to integers
    """
    if issparse(array):
        if not array.data.dtype.kind == "i" or not np.all(np.abs(array.data - np.round(array.data)) < tolerance):
            raise ValueError("Non-zero elements of the matrix must be close to integer values.")
    else:
        if not array.dtype.kind == "i" or not np.all(np.abs(array - np.round(array)) < tolerance):
            raise ValueError("Matrix must be a count matrix.")
    if (array < 0).sum() > 0:
        raise ValueError("Non.zero elements of the matrix must be postiive.")
