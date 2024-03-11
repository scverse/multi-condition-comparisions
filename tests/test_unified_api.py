import pytest

from multi_condition_comparisions.methods import EdgeR, PyDESeq2, Statsmodels, WilcoxonTest


@pytest.mark.parametrize("method", [WilcoxonTest, Statsmodels, PyDESeq2, EdgeR])
def test_unified(test_adata_minimal, method):
    res_df = method.compare_groups(adata=test_adata_minimal, column="condition", baseline="A", groups_to_compare="B")
    assert res_df.loc["gene1"]["pvals"] < 0.05
    assert res_df.loc["gene2"]["pvals"] > 0.05
    res_df = method.compare_groups(
        adata=test_adata_minimal, column="donor", baseline="D0", groups_to_compare=["D1", "D2", "D3"]
    )
    assert (res_df.loc["gene2"]["pvals"].values < 0.05).all()
