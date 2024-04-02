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
