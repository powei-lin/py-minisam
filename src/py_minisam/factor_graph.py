from dataclasses import dataclass
from typing import List
from functools import partial
import numpy as np
from py_minisam.auto_diff import diff_3point_inplace

@dataclass
class Factor:
    dim_residual: int
    residual_row_start_idx: int
    variable_key_list: List[str]
    residual_func: callable
    jac_func = None
    loss_func = None
    jac_sparsity = None

    def __post_init__(self):
        self.jac_func = partial(
            diff_3point_inplace, self.residual_func, self.variable_col_start_index_list, self.residual_row_start_idx
        )

    def jacobians_inplace(self, jac: np.ndarray, *variables):
        self.jac_func(jac, *variables)

class FactorGraph:
    def __init__(self) -> None:
        self.factors = []