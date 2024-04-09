from collections.abc import Sequence

import pytest
from pandas.core.api import DataFrame as DataFrame

from multi_condition_comparisions.methods import LinearModelBase


@pytest.fixture
def MockLinearModel():
    class _MockLinearModel(LinearModelBase):
        def _check_counts(self) -> None:
            pass

        def fit(self, **kwargs) -> None:
            pass

        def _test_single_contrast(self, contrast: Sequence[float], **kwargs) -> DataFrame:
            pass

    return _MockLinearModel


@pytest.mark.parametrize(
    "formula,cond_kwargs,expected_contrast",
    [
        ["~ condition", {"condition": "A"}, [1, 0]],
        ["~ condition", {"condition": "B"}, [1, 1]],
        ["~ condition", {"condition": "42"}, ValueError],  # non-existant category
        ["~ 0 + condition", {"condition": "A"}, [1, 0]],
        ["~ 0 + condition", {"condition": "B"}, [0, 1]],
        ["~ donor", {"donor": "D0"}, [1, 0, 0, 0]],
        ["~ C(donor)", {"donor": "D0"}, [1, 0, 0, 0]],
        ["~ C(donor, contr.treatment(base='D2'))", {"donor": "D2"}, [1, 0, 0, 0]],
        ["~ C(donor, contr.treatment(base='D2'))", {"donor": "D0"}, [1, 1, 0, 0]],
        ["~ donor + continuous", {"donor": "D1"}, [1, 1, 0, 0, 0]],
        ["~ donor + np.log1p(continuous)", {"donor": "D1"}, [1, 1, 0, 0, 0]],
        [
            "~ donor + continuous + np.log(continuous)",
            {"donor": "D0"},
            ValueError,
        ],  # current limitation: variables may only be used once per formula
        [
            "~ donor + C(donor)",
            {"donor": "D0"},
            ValueError,
        ],  # current limitation: variables may only be used once per formula
        ["~ C(donor, contr.sum)", {"donor": "D0"}, [1, 1, 0, 0]],
        ["~ C(donor, contr.sum)", {"donor": "D3"}, [1, -1, -1, -1]],
        ["~ condition + donor", {"condition": "A"}, [1, 0, 0, 0, 0]],
        ["~ 0 + condition + donor", {"donor": "D1"}, [0, 0, 1, 0, 0]],
        ["~ condition + donor", {"donor": "D2"}, [1, 0, 0, 1, 0]],
        ["~ condition + donor", {"condition": "B", "donor": "D2"}, [1, 1, 0, 1, 0]],
        ["~ condition * donor", {"condition": "A"}, [1, 0, 0, 0, 0]],
        ["~ condition + donor + condition:donor", {"condition": "A"}, [1, 0, 0, 0, 0]],
        ["~ condition:donor", {"condition": "A"}, [1, 0, 0, 0, 0]],
        # TODO: also include tests for errors that should be caught by .cond
    ],
)
def test_model_cond(test_adata_minimal, MockLinearModel, formula, cond_kwargs, expected_contrast):
    mod = MockLinearModel(test_adata_minimal, formula)
    if isinstance(expected_contrast, type):
        with pytest.raises(expected_contrast):
            mod.cond(**cond_kwargs)
    else:
        actual_contrast = mod.cond(**cond_kwargs)
        assert actual_contrast.tolist() == expected_contrast
        assert actual_contrast.index.tolist() == mod.design.columns.tolist()
