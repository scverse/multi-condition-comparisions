import pytest

from multi_condition_comparisions._util.formulaic import get_factor_storage_and_materializer


@pytest.mark.parametrize(
    "formula,expected_factor_metadata",
    [
        [
            "~ donor",
            {"donor": {"reduced_rank": True, "custom_encoder": False, "base": ""}},
        ],
        [
            "~ C(donor)",
            {"donor": {"reduced_rank": True, "custom_encoder": True, "base": ""}},
        ],
        [
            "~ C(donor, contr.treatment(base='D2'))",
            {"donor": {"reduced_rank": True, "custom_encoder": False, "base": "D2"}},
        ],
        [
            "~ C(donor, contr.sum)",
            {"donor": {"reduced_rank": True, "custom_encoder": False, "base": ""}},
        ],
        [
            "~ condition",
            {"condition": {"reduced_rank": True, "custom_encoder": False, "base": ""}},
        ],
        [
            "~ C(condition)",
            {"condition": {"reduced_rank": True, "custom_encoder": True, "base": ""}},
        ],
        [
            "~ C(condition, contr.treatment(base='B'))",
            {"condition": {"reduced_rank": True, "custom_encoder": False, "base": "B"}},
        ],
        [
            "~ C(condition, contr.sum)",
            {"condition": {"reduced_rank": True, "custom_encoder": False, "base": ""}},
        ],
        [
            "~ 0 + condition",
            {"condition": {"reduced_rank": True, "custom_encoder": False, "base": ""}},
        ],
        [
            "~ condition + donor",
            {
                "condition": {"reduced_rank": True, "custom_encoder": False, "base": ""},
                "donor": {"reduced_rank": True, "custom_encoder": False, "base": ""},
            },
        ],
        [
            "~ condition * donor",
            {
                "condition": {"reduced_rank": True, "custom_encoder": False, "base": ""},
                "donor": {"reduced_rank": True, "custom_encoder": False, "base": ""},
            },
        ],
        [
            "~ condition + C(condition) + C(condition, contr.treatment(base='B'))",
            {"condition": {"reduced_rank": True, "custom_encoder": False, "base": ""}},
        ],
        [
            "~ condition + continuous + np.log(continuous)",
            {"condition": {"reduced_rank": True, "custom_encoder": False, "base": ""}},
        ],
        [
            "~ condition * donor + continuous",
            {
                "condition": {"reduced_rank": True, "custom_encoder": False, "base": ""},
                "donor": {"reduced_rank": True, "custom_encoder": False, "base": ""},
            },
        ],
    ],
)
def test_custom_materializer(test_adata_minimal, formula, expected_factor_metadata):
    """Test that the custom materializer correctly stores the baseline category."""
    factor_storage, materializer = get_factor_storage_and_materializer()
    materializer(test_adata_minimal.obs).get_model_matrix(formula)
    for factor, expected_metadata in expected_factor_metadata.items():
        actual_metadata = factor_storage[factor]
        for k in expected_metadata:
            assert getattr(actual_metadata, k) == expected_metadata[k]
