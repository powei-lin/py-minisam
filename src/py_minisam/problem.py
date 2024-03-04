from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Tuple

import numpy as np

from py_minisam.auto_diff import diff_3point_inplace
from py_minisam.factor_graph import FactorGraph


@dataclass
class ResidualBlock:

    dim_residual: int
    residual_row_start_idx: int
    variable_col_start_index_list: List[int]
    residual_func: callable
    jac_func = None
    loss_func = None
    jac_sparsity = None

    def __post_init__(self):
        self.jac_func = partial(
            diff_3point_inplace, self.residual_func, self.variable_col_start_index_list, self.residual_row_start_idx
        )

    def jacobians_inplace(self, jacobian_matrix: np.ndarray, *variables):
        self.jac_func(jacobian_matrix, *variables)


class Problem:
    def __init__(self) -> None:
        self._dim_variable = 0
        self._dim_residual = 0
        self.residual_blocks: List[ResidualBlock] = []
        self.variable_addr_to_col_idx_dict: Dict[int, int] = {}
        self.col_idx_to_variable_dict: Dict[int, np.ndarray] = {}

    @staticmethod
    def from_factor_graph(factor_graph: FactorGraph, initial_variables: Dict[str, np.ndarray]):
        problem = Problem()
        for factor in factor_graph.factors:
            variables = (initial_variables[k] for k in factor.variable_key_list)
            problem.add_residual_block(factor.dim_residual, factor.error_func, *variables)
        return problem

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

        residual_block = ResidualBlock(
            dim_residual=dim_residual,
            residual_row_start_idx=self._dim_residual,
            variable_col_start_index_list=variable_col_start_index_list,
            residual_func=residual_func,
        )
        self.residual_blocks.append(residual_block)
        self._dim_residual += dim_residual

    def compute_residual_and_jacobian(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        jac = np.zeros((self._dim_residual, self._dim_variable), dtype=np.float64)
        residuals = np.zeros(self._dim_residual, dtype=np.float64)
        for residual_block in self.residual_blocks:
            variables = []
            for col in residual_block.variable_col_start_index_list:
                variables.append(params[col : col + self.col_idx_to_variable_dict[col].size])
            residual = residual_block.residual_func(*variables)
            residuals[residual_block.residual_row_start_idx : residual_block.residual_row_start_idx + residual.size] = (
                residual
            )
            residual_block.jacobians_inplace(jac, *variables)
        return residuals, jac

    def combine_variables(self) -> np.ndarray:
        all_variables = np.zeros(self._dim_variable, np.float64)
        for col, variable in self.col_idx_to_variable_dict.items():
            all_variables[col : col + variable.size] = variable
        return all_variables

    def write_back_variables(self, all_variables: np.ndarray):
        for col, variable in self.col_idx_to_variable_dict.items():
            variable[:] = all_variables[col : col + variable.size]
