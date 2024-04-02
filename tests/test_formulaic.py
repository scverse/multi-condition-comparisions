from multi_condition_comparisions._util.formulaic import get_factor_storage_and_materializer


def test_custom_materializer(test_adata_minimal, formula):
    """Test that the custom materializer correctly stores the baseline category."""
    factor_storage, materializer = get_factor_storage_and_materializer()
    materializer(test_adata_minimal.obs).get_model_matrix(formula)
    raise AssertionError()
