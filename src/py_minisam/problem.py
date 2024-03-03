from dataclasses import dataclass
from functools import partial
from typing import Dict, List

import numpy as np

from py_minisam.auto_diff import diff_3point_inplace


@dataclass
class _ResidualBlock:
    """DO NOT USE THIS CLASS DIRECTLY!"""

    dim_residual: int
    residual_row_start_idx: int
    variable_col_start_index_list: List[int]
    residual_func: callable
    loss_func = None
    jac_func = None
    jac_sparsity = None

    def __post_init__(self):
        self.jac_func = partial(
            diff_3point_inplace, self.residual_func, self.variable_col_start_index_list, self.residual_row_start_idx
        )

    def calulate_jac(self, jac: np.ndarray, *variables):
        self.jac_func(jac, *variables)


class Problem:
    def __init__(self) -> None:
        self._dim_variable = 0
        self._dim_residual = 0
        self.residual_blocks: List[_ResidualBlock] = []
        self.variable_addr_to_col_idx_dict: Dict[int, int] = {}
        self.col_idx_to_variable_dict: Dict[int, np.ndarray] = {}

    def add_residual_block(self, dim_residual, residual_func, *variables):
        if dim_residual <= 0:
            err_msg = f"dim_residual should > 0, got {dim_residual}"
            raise ValueError(err_msg)
        if not callable(residual_func):
            err_msg = "residual_func should be callable"
            raise TypeError(err_msg)

        variable_col_start_index_list = []
        for variable in variables:
            if not isinstance(variable, np.ndarray):
                raise TypeError
            address = variable.__array_interface__["data"][0]
            if address not in self.variable_addr_to_col_idx_dict:
                variable_dim = variable.shape[0]
                self.variable_addr_to_col_idx_dict[address] = self._dim_variable
                self.col_idx_to_variable_dict[self._dim_variable] = variable
                self._dim_variable += variable_dim
            variable_col_start_index_list.append(self.variable_addr_to_col_idx_dict[address])

        residual_block = _ResidualBlock(
            dim_residual=dim_residual,
            residual_row_start_idx=self._dim_residual,
            variable_col_start_index_list=variable_col_start_index_list,
            residual_func=residual_func,
        )
        self.residual_blocks.append(residual_block)
        self._dim_residual += dim_residual

    def combine_variables(self) -> np.ndarray:
        all_variables = np.zeros(self._dim_variable, np.float64)
        for col, variable in self.col_idx_to_variable_dict.items():
            all_variables[col : col + variable.size] = variable
        return all_variables

    def write_back_variables(self, all_variables: np.ndarray):
        for col, variable in self.col_idx_to_variable_dict.items():
            variable[:] = all_variables[col : col + variable.size]
