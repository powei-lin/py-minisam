from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Dict, List

import numpy as np
import scipy
from solver_concept.auto_diff import diff_3point_inplace


class SolverStatus(Enum):
    Running = auto()
    GradientTooSmall = auto()  # eps > max(J'*f(x))
    RelativeStepSizeTooSmall = auto()  # eps > ||dx|| / ||x||
    ErrorTooSmall = auto()  # eps > ||f(x)||
    HitMaxIterations = auto()


@dataclass
class SolverParameters:
    gradient_threshold: float = 1e-16
    relative_step_threshold: float = 1e-16
    error_threshold: float = 1e-16
    initial_scale_factor: float = 1e-3
    max_iterations: int = 100


@dataclass
class ProblemResult:
    error_magnitude: float = 0.0
    gradient_magnitude: float = 0.0
    num_failed_linear_solves: int = 0
    iterations: int = 0
    status: SolverStatus = SolverStatus.Running


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
    # ----->
    # Jac  |
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


def solve_gn(problem: Problem, max_iteration: int = 250):
    result = ProblemResult()
    params = problem.combine_variables()
    for i in range(max_iteration):
        print(i)
        result.iterations += 1

        jac = np.zeros((problem._dim_residual, problem._dim_variable), dtype=np.float64)
        residuals = np.zeros(problem._dim_residual, dtype=np.float64)
        for residual_block in problem.residual_blocks:
            variables = []
            for col in residual_block.variable_col_start_index_list:
                variables.append(params[col : col + problem.col_idx_to_variable_dict[col].size])
            residual = residual_block.residual_func(*variables)
            residuals[residual_block.residual_row_start_idx : residual_block.residual_row_start_idx + residual.size] = (
                residual
            )
            residual_block.calulate_jac(jac, *variables)
        # gradient = jac.T @ -residual
        # jtj = jac.T @ jac
        # max_gradient = np.amax(np.abs(gradient))
        H = jac.T @ jac
        B = -jac.T @ residuals
        dx = scipy.linalg.solve(H, B)
        print(dx)
        if np.linalg.norm(dx) < 1e-8:
            break
        params += dx
        problem.write_back_variables(params)
    print(np.linalg.norm(residuals))


def solve(problem: Problem, max_iteration: int = 150):
    result = ProblemResult()
    solver_params = SolverParameters()
    v = 2
    u = 0.0

    for i in range(max_iteration):
        result.iterations += 1

        jac = np.zeros((problem._dim_residual, problem._dim_variable), dtype=np.float64)
        residuals = np.zeros(problem._dim_residual, dtype=np.float64)
        for residual_block in problem.residual_blocks:
            variables = []
            for col in residual_block.variable_col_start_index_list:
                variables.append(problem.col_idx_to_variable_dict[col])
            residual = residual_block.residual_func(*variables)
            residuals[residual_block.residual_row_start_idx : residual_block.residual_row_start_idx + residual.size] = (
                residual
            )
            residual_block.calulate_jac(jac, *variables)
        gradient = jac.T @ -residuals
        jtj = jac.T @ jac
        max_gradient = np.amax(np.abs(gradient))
        if max_gradient < solver_params.gradient_threshold:
            print("g small")
            break
        elif np.linalg.norm(residual) < solver_params.error_threshold:
            print("err small")
            break
        if i == 0:
            u = solver_params.initial_scale_factor * np.amax(np.diag(jtj))

        jtj_augmented = jtj.copy()
        np.fill_diagonal(jtj_augmented, np.diag(jtj_augmented) + u)
        dx = np.linalg.solve(jtj_augmented, gradient)
        solution = jtj_augmented @ dx
        if np.amin(np.abs(solution - gradient)) < solver_params.error_threshold:
            params = problem.combine_variables()
            if np.linalg.norm(dx) < solver_params.relative_step_threshold * np.linalg.norm(params):
                result.status = SolverStatus.RelativeStepSizeTooSmall
                break
            param_new = params + dx
            residual_new = np.zeros(problem._dim_residual, np.float64)
            for residual_block in problem.residual_blocks:
                variables = []
                for col in residual_block.variable_col_start_index_list:
                    variables.append(problem.col_idx_to_variable_dict[col])
                residual = residual_block.residual_func(*variables)
                residuals[
                    residual_block.residual_row_start_idx : residual_block.residual_row_start_idx + residual.size
                ] = residual
            r_norm = np.linalg.norm(residual)
            r_norm_new = np.linalg.norm(residual_new)

            rho = (r_norm * r_norm - r_norm_new * r_norm_new) / np.dot(dx, (u * dx + gradient))
            if rho > 0.0:
                # params = param_new
                print("good")
                problem.write_back_variables(param_new)
                tmp = 2.0 * rho - 1.0
                u = u * max(1.0 / 3.0, 1.0 - tmp**3)
                v = 2
                continue
        else:
            result.num_failed_linear_solves += 1
            print(f"fail {solution - gradient}")
        u *= v
        v *= 2

        # print(f"residual shape: {residuals.shape}")
        # print(f"jac shape: {jac.shape}")
