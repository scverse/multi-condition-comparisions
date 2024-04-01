import numpy as np
import pytest

from multi_condition_comparisions.methods import AVAILABLE_METHODS


@pytest.mark.parametrize("method", AVAILABLE_METHODS)
@pytest.mark.parametrize("paired_by", ["pairing", None])
def test_unified(test_adata_minimal, method, paired_by):
    """
    Test that all methods implement the unified API.

    Here, we don't check the correctness of the results
    (we have the method-specific tests for that), but rather that the interface works
    as expected and the format of the resulting data frame is what we expect.

    TODO: tests for layers
    TODO: tests for mask
    """
    # case 1: Single group to compare
    res_df = method.compare_groups(
        adata=test_adata_minimal, column="condition", baseline="A", groups_to_compare="B", paired_by=paired_by
    )
    assert res_df.shape[0] == test_adata_minimal.shape[1], "The result dataframe must contain a value for each var name"
    assert {"variable", "p_value", "log_fc", "adj_p_value"} - set(
        res_df.columns
    ) == set(), "Mandated column names not in result df"
    assert np.all((0 <= res_df["p_value"]) & (res_df["p_value"] <= 1))
    assert np.all((0 <= res_df["adj_p_value"]) & (res_df["adj_p_value"] <= 1))
    assert np.all(res_df["adj_p_value"] >= res_df["p_value"])

    # case 2: multiple groups to compare
    res_df = method.compare_groups(
        adata=test_adata_minimal,
        column="donor",
        baseline="D0",
        groups_to_compare=["D1", "D2", "D3"],
        paired_by=paired_by,
    )
    assert (
        res_df.shape[0] == 3 * test_adata_minimal.shape[1]
    ), "The result dataframe must contain a value for each var name"
    assert {"variable", "p_value", "log_fc", "adj_p_value"} - set(
        res_df.columns
    ) == set(), "Mandated column names not in result df"
    assert np.all((0 <= res_df["p_value"]) & (res_df["p_value"] <= 1))
    assert np.all((0 <= res_df["adj_p_value"]) & (res_df["adj_p_value"] <= 1))
    assert np.all(res_df["adj_p_value"] >= res_df["p_value"])
