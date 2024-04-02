import pandas as pd
import pytest
from formulaic.parser.types import Factor

from multi_condition_comparisions._util.formulaic import get_factor_storage_and_materializer


@pytest.mark.parametrize(
    "formula,reorder_categorical,expected_factor_metadata",
    [
        [
            "~ donor",
            None,
            {"donor": {"reduced_rank": True, "custom_encoder": False, "base": "D0"}},
        ],
        [
            "~ donor",
            {"donor": ["D2", "D1", "D0", "D3"]},
            {"donor": {"reduced_rank": True, "custom_encoder": False, "base": "D2"}},
        ],
        [
            "~ C(donor)",
            None,
            {"C(donor)": {"reduced_rank": True, "custom_encoder": True, "base": "D0"}},
        ],
        [
            "~ C(donor, contr.treatment(base='D2'))",
            None,
            {"C(donor, contr.treatment(base='D2'))": {"reduced_rank": True, "custom_encoder": True, "base": "D2"}},
        ],
        [
            "~ C(donor, contr.sum)",
            None,
            {"C(donor, contr.sum)": {"reduced_rank": True, "custom_encoder": True, "base": "D3"}},
        ],
        [
            "~ C(donor, contr.sum)",
            {"donor": ["D1", "D0", "D3", "D2"]},
            {"C(donor, contr.sum)": {"reduced_rank": True, "custom_encoder": True, "base": "D2"}},
        ],
        [
            "~ condition",
            None,
            {"condition": {"reduced_rank": True, "custom_encoder": False, "base": "A"}},
        ],
        [
            "~ C(condition)",
            None,
            {"C(condition)": {"reduced_rank": True, "custom_encoder": True, "base": "A"}},
        ],
        [
            "~ C(condition, contr.treatment(base='B'))",
            None,
            {"C(condition, contr.treatment(base='B'))": {"reduced_rank": True, "custom_encoder": True, "base": "B"}},
        ],
        [
            "~ C(condition, contr.sum)",
            None,
            {"C(condition, contr.sum)": {"reduced_rank": True, "custom_encoder": True, "base": "B"}},
        ],
        [
            "~ 0 + condition",
            None,
            {"condition": {"reduced_rank": False, "custom_encoder": False, "base": None}},
        ],
        [
            "~ condition + donor",
            None,
            {
                "condition": {"reduced_rank": True, "custom_encoder": False, "base": "A"},
                "donor": {"reduced_rank": True, "custom_encoder": False, "base": "D0"},
            },
        ],
        [
            "~ 0 + condition + donor",
            None,
            {
                "condition": {"reduced_rank": False, "custom_encoder": False, "base": None},
                "donor": {"reduced_rank": True, "custom_encoder": False, "base": "D0"},
            },
        ],
        [
            "~ condition * donor",
            None,
            {
                "condition": {"reduced_rank": True, "custom_encoder": False, "base": "A"},
                "donor": {"reduced_rank": True, "custom_encoder": False, "base": "D0"},
            },
        ],
        [
            "~ condition * C(donor, contr.treatment(base='D2'))",
            None,
            {
                "condition": {"reduced_rank": True, "custom_encoder": False, "base": "A"},
                "C(donor, contr.treatment(base='D2'))": {"reduced_rank": True, "custom_encoder": True, "base": "D2"},
            },
        ],
        [
            "~ condition + C(condition) + C(condition, contr.treatment(base='B'))",
            None,
            {
                "condition": {"reduced_rank": True, "custom_encoder": False, "base": "A"},
                "C(condition)": {"reduced_rank": True, "custom_encoder": True, "base": "A"},
                "C(condition, contr.treatment(base='B'))": {"reduced_rank": True, "custom_encoder": True, "base": "B"},
            },
        ],
        [
            "~ condition + continuous + np.log(continuous)",
            None,
            {
                "condition": {
                    "reduced_rank": True,
                    "custom_encoder": False,
                    "base": "A",
                    "kind": Factor.Kind.CATEGORICAL,
                },
                "continuous": {
                    "reduced_rank": False,
                    "custom_encoder": False,
                    "base": None,
                    "kind": Factor.Kind.NUMERICAL,
                },
                "np.log(continuous)": {
                    "reduced_rank": False,
                    "custom_encoder": False,
                    "base": None,
                    "kind": Factor.Kind.NUMERICAL,
                },
            },
        ],
        [
            "~ condition * donor + continuous",
            None,
            {
                "condition": {"reduced_rank": True, "custom_encoder": False, "base": "A"},
                "donor": {"reduced_rank": True, "custom_encoder": False, "base": "D0"},
                "continuous": {
                    "reduced_rank": False,
                    "custom_encoder": False,
                    "base": None,
                    "kind": Factor.Kind.NUMERICAL,
                },
            },
        ],
    ],
)
def test_custom_materializer(test_adata_minimal, formula, reorder_categorical, expected_factor_metadata):
    """Test that the custom materializer correctly stores the baseline category.

    Parameters
    ----------
    test_adata_minimal
        adata fixture
    formula
        Formula to test
    reorder_categorical
        Create a pandas categorical for a given column with a certain order of categories
    expected_factor_metadata
        dict with expected values for each factor
    """
    if reorder_categorical is not None:
        for col, order in reorder_categorical.items():
            test_adata_minimal.obs[col] = pd.Categorical(test_adata_minimal.obs[col], categories=order)
    factor_storage, materializer = get_factor_storage_and_materializer()
    materializer(test_adata_minimal.obs).get_model_matrix(formula)
    for factor, expected_metadata in expected_factor_metadata.items():
        actual_metadata = factor_storage[factor]
        for k in expected_metadata:
            assert getattr(actual_metadata, k) == expected_metadata[k]
