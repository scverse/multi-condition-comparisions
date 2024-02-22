from abc import ABC, abstractmethod
import pandas as pd


class BaseMethod(ABC):
    @abstractmethod
    @staticmethod
    def compare_groups() -> pd.DataFrame: ...


class LinearModelBase(BaseMethod): ...
