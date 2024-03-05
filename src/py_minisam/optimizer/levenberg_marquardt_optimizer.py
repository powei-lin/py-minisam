import numpy as np
from scipy.sparse import eye_array
from scipy.sparse.linalg import spsolve

from py_minisam.optimizer.base_optimizer import BaseOptimizer, ProblemResult, SolverParameters, SolverStatus
from py_minisam.problem import Problem


class LevenbergMarquardtOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        pass

    def optimize(self, problem: Problem, max_iteration: int = 100):
        result = ProblemResult()
        solver_params = SolverParameters()
        v = 2
        u = 0.0

        for i in range(max_iteration):
            print(f"{u=}, {v=}")
            result.iterations += 1

            params = problem.combine_variables()
            residuals, jac = problem.compute_residual_and_jacobian(params)
            gradient = jac.T @ -residuals
            jtj = jac.T @ jac
            max_gradient = np.amax(np.abs(gradient))
            if max_gradient < solver_params.gradient_threshold:
                print("g small")
                break
            elif np.linalg.norm(residuals) < solver_params.error_threshold:
                print("err small")
                break
            if i == 0:
                u = solver_params.initial_scale_factor * np.amax(jtj.diagonal())

            jtj_augmented = jtj + eye_array(*jtj.shape) * u
            # jtj_augmented.setdiag(jtj_augmented.diagonal() + u)
            # np.fill_diagonal(jtj_augmented.set, jtj_augmented.diagonal() + u)
            dx = spsolve(jtj_augmented, gradient)
            solution = jtj_augmented @ dx
            if np.amin(np.abs(solution - gradient)) < solver_params.error_threshold:
                if np.linalg.norm(dx) < solver_params.relative_step_threshold * np.linalg.norm(params):
                    result.status = SolverStatus.RelativeStepSizeTooSmall
                    print("rrr")
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
                    print("not good")
            else:
                result.num_failed_linear_solves += 1
                print(f"fail {solution - gradient}")
            u *= v
            v *= 2
