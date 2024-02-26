"""Helpers to interact with Formulaic Formulas"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from formulaic import ModelSpec
from formulaic.materializers import PandasMaterializer
from formulaic.materializers.types import EvaluatedFactor
from interface_meta import override


@dataclass
class FactorMetadata:
    """Store (relevant) metadata for a factor of a formula."""

    name: str
    reduced_rank: bool
    base_level: str


def get_factor_storage_and_materializer():
    """
    Keep track of categorical factors used in a model spec.

    Generates a custom materializers that reports back certain metadata upon materialization of the model matrix.
    """
    factor_storage: dict[str, FactorMetadata] = {}

    class CustomPandasMaterializer(PandasMaterializer):
        """An extension of the PandasMaterializer that records all cateogrical variables and their (base) categories."""

        REGISTER_NAME = "custom_pandas"
        REGISTER_INPUTS = ("pandas.core.frame.DataFrame",)
        REGISTER_OUTPUTS = ("pandas", "numpy", "sparse")

        factor_metadata_storage = factor_storage

        @override
        def _encode_evaled_factor(
            self, factor: EvaluatedFactor, spec: ModelSpec, drop_rows: Sequence[int], reduced_rank: bool = False
        ) -> dict[str, Any]:
            assert factor.expr not in self.factor_metadata_storage, "Factor already present in metadata storage"
            self.factor_metadata_storage[factor.expr] = FactorMetadata(
                name=factor.expr, reduced_rank=reduced_rank, base_level="TODO"
            )
            print(factor)
            print(spec)
            res = super()._encode_evaled_factor(factor, spec, drop_rows, reduced_rank)
            print(res)
            return res

    return factor_storage, CustomPandasMaterializer
