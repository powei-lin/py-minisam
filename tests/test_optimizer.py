from functools import partial
from time import perf_counter

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from py_minisam.optimizer import GaussNewtonOptimizer, LevenbergMarquardtOptimizer
from py_minisam.problem import Problem


def cost_function(rmat_gt: np.ndarray, p3d: np.ndarray, rvec: np.ndarray):
    rotation_matrix = Rotation.from_rotvec(rvec).as_matrix()
    rotated_3d_points = rmat_gt @ p3d
    return ((rotation_matrix @ p3d) - rotated_3d_points).T.flatten()


@pytest.mark.parametrize("solver_class", [GaussNewtonOptimizer, LevenbergMarquardtOptimizer])
def test_optimizer_solve_rotation(solver_class):
    np.random.seed(2)
    rvec_gt = np.random.random(3)
    rmat_gt = Rotation.from_rotvec(rvec_gt).as_matrix()
    rvec_noise = rvec_gt + np.random.random(3) / 10.0

    rvec_noise_init = rvec_noise.copy()
    problem = Problem()

    p3d0 = np.random.random((3, 2))
    problem.add_residual_block(p3d0.size, partial(cost_function, rmat_gt, p3d0), rvec_noise_init)
    p3d1 = np.random.random((3, 3))
    problem.add_residual_block(p3d1.size, partial(cost_function, rmat_gt, p3d1), rvec_noise_init)
    solver = solver_class()
    solver.optimize(problem)
    assert np.allclose(rvec_gt, rvec_noise_init)
