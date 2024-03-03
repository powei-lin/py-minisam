from functools import partial
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from py_minisam.optimizer.gauss_newton_optimizer import GaussNewtonOptimizer
from py_minisam.problem import Problem

np.random.seed(2)

rvec_gt = np.random.random(3)
rmat_gt = Rotation.from_rotvec(rvec_gt).as_matrix()


def cost_function(p3d: np.ndarray, rvec: np.ndarray):
    rotation_matrix = Rotation.from_rotvec(rvec).as_matrix()
    rotated_3d_points = rmat_gt @ p3d
    return ((rotation_matrix @ p3d) - rotated_3d_points).T.flatten()


def main():
    rvec_noise = rvec_gt + np.random.random(3) / 10.0
    print("init:", rvec_noise)

    problem = Problem()

    p3d0 = np.random.random((3, 2))
    problem.add_residual_block(p3d0.size, partial(cost_function, p3d0), rvec_noise)
    p3d1 = np.random.random((3, 3))
    problem.add_residual_block(p3d1.size, partial(cost_function, p3d1), rvec_noise)
    print(f"{rvec_noise=}")
    gn = GaussNewtonOptimizer()
    gn.optimize(problem)
    print(f"{rvec_noise=}")
    print(f"{rvec_gt=}")

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
