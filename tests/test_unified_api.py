import pytest

from multi_condition_comparisions.methods import EdgeR, PyDESeq2, Statsmodels, TTest, WilcoxonTest


@pytest.mark.parametrize("method", [WilcoxonTest, TTest, Statsmodels, PyDESeq2, EdgeR])
@pytest.mark.parametrize("paired_by", ["pairing", None])
def test_unified(test_adata_minimal, method, paired_by):
    res_df = method.compare_groups(
        adata=test_adata_minimal, column="condition", baseline="A", groups_to_compare="B", paired_by=paired_by
    )
    if paired_by is None:
        assert res_df.loc["gene1"]["pvals"] < 0.005
        assert res_df.loc["gene2"]["pvals"] > 0.005
    else:
        assert res_df.loc["gene2"]["pvals"] > res_df.loc["gene1"]["pvals"]
    res_df = method.compare_groups(
        adata=test_adata_minimal,
        column="donor",
        baseline="D0",
        groups_to_compare=["D1", "D2", "D3"],
        paired_by=paired_by,
    )
    assert res_df.loc["gene2"]["pvals"].values[0] < 0.005
    assert res_df.loc["gene2"]["pvals"].values[2] < 0.005
    assert res_df.loc["gene2"]["pvals"].values[1] > 0.005  # D1 is from same distribution
