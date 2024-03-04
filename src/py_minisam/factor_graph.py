from abc import abstractmethod
from typing import List

import numpy as np


class FactorBase:
    def __init__(
        self,
        dim_residual: int,
        variable_key_list: List[str],
    ) -> None:
        self.dim_residual = dim_residual
        self.variable_key_list = variable_key_list

    @abstractmethod
    def error_func(self, *variables) -> np.ndarray:
        pass


class FactorGraph:
    def __init__(self) -> None:
        self.factors: List[FactorBase] = []

    def add(self, factor: FactorBase):
        if not isinstance(factor, FactorBase):
            raise ValueError
        self.factors.append(factor)
