"""Helpers to interact with Formulaic Formulas"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from formulaic import FactorValues, ModelSpec
from formulaic.materializers import PandasMaterializer
from formulaic.materializers.types import EvaluatedFactor
from formulaic.parser.types import Factor
from interface_meta import override


@dataclass
class FactorMetadata:
    """Store (relevant) metadata for a factor of a formula."""

    name: str
    """The unambiguous factor name as specified in the formula. E.g. `donor`, or `C(donor, contr.treatment(base="A"))`"""

    reduced_rank: bool
    """Whether a column will be dropped because it is redundant"""

    drop_field: str = None
    """The category that is dropped. Note that this may also be populated if `reduced_rank = False`"""

    kind: Factor.Kind = None
    """Type of the factor"""

    categories: Sequence[str] = None
    """The unique categories in this factor"""

    colname_format: str = None
    """A formattable string that can be used to generate the column name in the design matrix, e.g. `{name}[T.{field}]`"""


def get_factor_storage_and_materializer() -> tuple[dict[str, FactorMetadata], type]:
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
            self.factor_metadata_storage[factor.expr] = FactorMetadata(name=factor.expr, reduced_rank=reduced_rank)
            return super()._encode_evaled_factor(factor, spec, drop_rows, reduced_rank)

        @override
        def _flatten_encoded_evaled_factor(self, name: str, values: FactorValues[dict]) -> dict[str, Any]:
            """Function is called at the end, here se still have access to the raw factor values."""
            factor_metadata = self.factor_metadata_storage[name]
            factor_metadata.drop_field = values.__formulaic_metadata__.drop_field
            factor_metadata.categories = values.__formulaic_metadata__.column_names
            factor_metadata.colname_format = values.__formulaic_metadata__.format

            return super()._flatten_encoded_evaled_factor(name, values)

    return factor_storage, CustomPandasMaterializer
