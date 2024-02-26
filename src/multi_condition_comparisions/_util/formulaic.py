"""Helpers to interact with Formulaic Formulas"""

from collections.abc import Sequence
from typing import Any

from formulaic import ModelSpec
from formulaic.materializers import PandasMaterializer
from formulaic.materializers.types import EvaluatedFactor
from interface_meta import override


class CustomPandasMaterializer(PandasMaterializer):
    """An extension of the PandasMaterializer that records all cateogrical variables and their (base) categories."""

    REGISTER_NAME = "custom_pandas"
    REGISTER_INPUTS = ("pandas.core.frame.DataFrame",)
    REGISTER_OUTPUTS = ("pandas", "numpy", "sparse")

    @override
    def _encode_evaled_factor(
        self, factor: EvaluatedFactor, spec: ModelSpec, drop_rows: Sequence[int], reduced_rank: bool = False
    ) -> dict[str, Any]:
        print(factor)
        print(spec)
        res = super()._encode_evaled_factor(factor, spec, drop_rows, reduced_rank)
        print(res)
        return res
