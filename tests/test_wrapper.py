from inspect import signature
from typing import get_args

import numpy as np
import pytest
from pandas import testing as tm

from multi_condition_comparisions.tl.de import MethodRegistry, run_de


def test_arg_types():
    run_de_args = set(get_args(signature(run_de).parameters["method"].annotation))
    assert run_de_args == set(MethodRegistry.keys()), run_de_args


@pytest.mark.parametrize(
    "method",
    list(MethodRegistry.keys()),
)
@pytest.mark.parametrize(
    "design,contrasts",
    [
        (
            "~condition",
            {
                "first": {"column": "condition", "baseline": "A", "group_to_compare": "B"},
            },
        ),
        (
            "~condition",
            {
                "first": {"column": "condition", "baseline": "A", "group_to_compare": "B"},
                "second": {"column": "condition", "baseline": "B", "group_to_compare": "A"},
            },
        ),
        (
            "~condition",
            {
                "first": ["condition", "A", "B"],
            },
        ),
        (
            "~condition",
            {
                "first": ["condition", "A", "B"],
                "second": ["condition", "B", "A"],
            },
        ),
        (
            "~condition1+group",
            {
                "first": {"column": "condition1", "baseline": "A", "group_to_compare": "B"},
            },
        ),
        (
            "~condition1+group",
            {
                "first": {"column": "condition1", "baseline": "A", "group_to_compare": "B"},
                "second": {"column": "condition1", "baseline": "B", "group_to_compare": "A"},
            },
        ),
        (
            "~condition1+group",
            {
                "first": ["condition1", "A", "B"],
            },
        ),
        (
            "~condition1+group",
            {
                "first": ["condition1", "A", "B"],
                "second": ["condition1", "B", "A"],
            },
        ),
    ],
)
def test_run_de(test_adata, method, design, contrasts):
    test_adata.obs["condition1"] = test_adata.obs["condition"].copy()
    res_df = run_de(adata=test_adata, method=method, contrasts=contrasts, design=design)

    assert len(res_df) == test_adata.n_vars * len(contrasts)
    # Check that the index of the result per group matches the var_names of the AnnData object
    tm.assert_index_equal(
        test_adata.var_names, res_df.index[: len(res_df) // len(contrasts)], check_order=False, check_names=False
    )

    expected_columns = {"pvals", "pvals_adj", "logfoldchanges"}
    assert expected_columns.issubset(set(res_df.columns.tolist()))
    assert np.all((0 <= res_df["pvals"]) & (res_df["pvals"] <= 1))
