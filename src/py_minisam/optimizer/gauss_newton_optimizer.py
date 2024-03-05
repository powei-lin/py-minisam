import numpy as np
from scipy.sparse.linalg import spsolve

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

            residuals, jac = problem.compute_residual_and_jacobian(params)
            hessian = jac.T @ jac
            b = -jac.T @ residuals
            dx = spsolve(hessian, b)
            if np.linalg.norm(dx) < 1e-16:
                break
            params += dx
            problem.write_back_variables(params)
        print(np.linalg.norm(residuals))
