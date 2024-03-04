from functools import partial
from time import perf_counter
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation

from py_minisam.factor_graph import FactorBase, FactorGraph
from py_minisam.optimizer import GaussNewtonOptimizer, LevenbergMarquardtOptimizer
from py_minisam.problem import Problem

np.random.seed(2)

rvec_gt = np.random.random(3)
rmat_gt = Rotation.from_rotvec(rvec_gt).as_matrix()


def cost_function(p3d: np.ndarray, rvec: np.ndarray):
    rotation_matrix = Rotation.from_rotvec(rvec).as_matrix()
    rotated_3d_points = rmat_gt @ p3d
    return ((rotation_matrix @ p3d) - rotated_3d_points).T.flatten()


class CustomFactor(FactorBase):
    def __init__(self, key: str, p3d: np.ndarray) -> None:
        self.p3d = p3d
        super().__init__(p3d.size, [key])

    def error_func(self, rvec) -> np.ndarray:
        rotation_matrix = Rotation.from_rotvec(rvec).as_matrix()
        rotated_3d_points = rmat_gt @ self.p3d
        return ((rotation_matrix @ self.p3d) - rotated_3d_points).T.flatten()


def main():
    rvec_noise = rvec_gt + np.random.random(3) / 10.0

    for solver_class in [GaussNewtonOptimizer, LevenbergMarquardtOptimizer]:
        rvec_noise_init = rvec_noise.copy()
        print("init:", rvec_noise_init)
        problem = Problem()

        p3d0 = np.random.random((3, 2))
        problem.add_residual_block(p3d0.size, partial(cost_function, p3d0), rvec_noise_init)
        p3d1 = np.random.random((3, 3))
        problem.add_residual_block(p3d1.size, partial(cost_function, p3d1), rvec_noise_init)
        print(f"{rvec_noise_init=}")
        solver = solver_class()
        solver.optimize(problem)
        print(f"{rvec_noise_init=}")
        print(f"{rvec_gt=}")

    # factor graph
    factor_graph = FactorGraph()
    p3d0 = np.random.random((3, 2))
    factor_graph.add(CustomFactor("rvec0", p3d0))
    p3d1 = np.random.random((3, 3))
    factor_graph.add(CustomFactor("rvec0", p3d1))
    initial_values = {"rvec0": rvec_noise.copy()}
    print(initial_values)
    gn = GaussNewtonOptimizer()
    gn.optimize_factor_graph(factor_graph, initial_values)
    print(initial_values)


if __name__ == "__main__":
    main()
