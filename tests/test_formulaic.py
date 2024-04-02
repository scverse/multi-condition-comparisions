import pytest

from multi_condition_comparisions._util.formulaic import get_factor_storage_and_materializer


@pytest.mark.parametrize(
    "formula",
    [
        "~ donor",
        "~ C(donor)",
        "~ C(donor, contr.treatment(base='D2'))",
        "~ C(donor, contr.sum)",
        "~ condition",
        "~ C(condition)",
        "~ C(condition, contr.treatment(base='B'))",
        "~ C(condition, contr.sum)",
        "~ 0 + condition",
        "~ condition + donor",
        "~ condition * donor",
        "~ condition + C(condition) + C(condition, contr.treatment(base='B'))",
        "~ condition + continuous + np.log(continuous)",
        "~ condition * donor + continuous",
    ],
)
def test_custom_materializer(test_adata_minimal, formula):
    """Test that the custom materializer correctly stores the baseline category."""
    factor_storage, materializer = get_factor_storage_and_materializer()
    materializer(test_adata_minimal.obs).get_model_matrix(formula)
    raise AssertionError(factor_storage)
