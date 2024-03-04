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
    def __init__(self, variable_key_list: List[str], p3d: np.ndarray) -> None:
        self.p3d = p3d
        super().__init__(p3d.size, variable_key_list)

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
    factor_graph.add(CustomFactor(["rvec0"], p3d0))
    p3d1 = np.random.random((3, 3))
    factor_graph.add(CustomFactor(["rvec0"], p3d1))
    initial_values = {"rvec0": rvec_noise.copy()}
    print(initial_values)
    problem2 = Problem.from_factor_graph(factor_graph, initial_values)
    gn = GaussNewtonOptimizer()
    gn.optimize(problem2)
    print(initial_values)

    # for iter in range(10):
    #     print(f"{iter=}")
    #     residual = cost_function(rvec_noise)
    # a = np.random.random((3, 3))
    # a[1:2, 1:2] = np.zeros(1)
    # print(a.size)
    # a.diagonal() += 1
    # print(a)
    # p_num = 100
    # p3ds_gt = (np.random.random((p_num, 3)) * 100 + np.array([0, 0, 1])).T
    # rvec_gt = np.random.random(3)
    # print("gt", rvec_gt)
    # rmat_gt = Rotation.from_rotvec(rvec_gt).as_matrix()
    # p3ds_gt_r = rmat_gt @ p3ds_gt
    # p2ds_gt = p3ds_gt_r[:2, :] / p3ds_gt_r[2:, :]

    # rvec = rvec_gt + np.random.random(3) / 100
    # rmat = Rotation.from_rotvec(rvec).as_matrix()
    # p3ds_r = rmat @ p3ds_gt
    # name_time = []

    # print("init  ", rvec_init)
    # s = perf_counter()
    # # problem.solve(verbose=2)
    # # t = perf_counter() - s
    # name_time.append((n, t))
    # print("result", rvec_init)
    # for n, t in name_time:
    #     print(f"{n}\t{t:.6f}s")


if __name__ == "__main__":
    main()
