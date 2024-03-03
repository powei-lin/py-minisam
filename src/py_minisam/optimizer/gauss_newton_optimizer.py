import numpy as np

from py_minisam.optimizer.base_optimizer import BaseOptimizer, ProblemResult
from py_minisam.problem import Problem


class GaussNewtonOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        pass

    def optimize(self, problem: Problem, max_iteration: int = 100):
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
                residuals[
                    residual_block.residual_row_start_idx : residual_block.residual_row_start_idx + residual.size
                ] = residual
                residual_block.calulate_jac(jac, *variables)
            hessian = jac.T @ jac
            b = -jac.T @ residuals
            dx = np.linalg.solve(hessian, b)
            # print(dx)
            if np.linalg.norm(dx) < 1e-8:
                break
            params += dx
            problem.write_back_variables(params)
        print(np.linalg.norm(residuals))
