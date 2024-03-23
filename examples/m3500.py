import math
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from py_minisam.factor_graph import FactorBase, FactorGraph
from py_minisam.optimizer import GaussNewtonOptimizer, LevenbergMarquardtOptimizer


# np.seterr(all='raise')
def rot(x: float) -> np.ndarray:
    c = math.cos(x)
    s = math.sin(x)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def to_theta(m):
    v = m[:2, :2] @ np.array([1, 0])
    return np.arctan2(v[1], v[0])


class SE2:
    def __init__(self, theta: float, x: float, y: float) -> None:
        self.theta = theta
        self.x = x
        self.y = y

    @staticmethod
    def from_matrix(m: np.ndarray):
        return SE2(to_theta(m[:2, :2]), m[0, 2], m[1, 2])

    def flatten(self) -> np.ndarray:
        return np.array([self.theta, self.x, self.y], dtype=np.float64)

    def as_matrix(self) -> np.ndarray:
        """Return self as a 3x3 transformation matrix.

        Returns:
            transformation matrix with shape (3, 3)
        """
        transform_mat = np.eye(3, 3, dtype=np.float64)
        transform_mat[:2, :2] = rot(self.theta)
        transform_mat[:2, 2] = np.array([self.x, self.y])
        return transform_mat

    def inv(self):
        """Return the inverse of self as SE3."""
        theta = -self.theta
        t = rot(theta) @ np.array([-self.x, -self.y])
        return SE2(theta, t[0], t[1])

    def __matmul__(self, other):
        """Matrix multiplication."""
        if isinstance(other, SE2):
            theta = (self.theta + other.theta + np.pi) % (2 * np.pi) - np.pi
            c = math.cos(self.theta)
            s = math.sin(self.theta)
            new_x = c * other.x - s * other.y + self.x
            new_y = s * other.x + c * other.y + self.y
            return SE2(theta, new_x, new_y)
        else:
            err_msg = "other has wrong type"
            raise ValueError(err_msg)


def error_func(t_k0_k1: np.ndarray, se2_k0: np.ndarray, se2_k1: np.ndarray) -> np.ndarray:
    se2_i_0 = SE2(*se2_k0)
    se2_i_1 = SE2(*se2_k1)
    se2_k0_k1 = SE2(*t_k0_k1)
    se2_1_0 = se2_i_1.inv() @ se2_i_0
    diff = (se2_k0_k1 @ se2_1_0).flatten()
    return diff


class BetweenFactor(FactorBase):
    def __init__(self, k0: str, k1: str, t_k0_k1: np.ndarray) -> None:
        self.kk = [k0, k1]
        super().__init__(3, self.kk)
        self.t_k0_k1 = t_k0_k1
        self.se2_k0_k1 = SE2(t_k0_k1[0], t_k0_k1[1], t_k0_k1[2])
        # self.loss = loss_mat

    def error_func(self, se2_k0: np.ndarray, se2_k1: np.ndarray) -> np.ndarray:
        se2_i_0 = SE2(se2_k0[0], se2_k0[1], se2_k0[2])
        se2_i_1 = SE2(se2_k1[0], se2_k1[1], se2_k1[2])
        se2_1_0 = se2_i_1.inv() @ se2_i_0
        diff = (self.se2_k0_k1 @ se2_1_0).flatten()
        return diff


class PriorFactor(FactorBase):
    def __init__(self, k0) -> None:
        super().__init__(3, [k0])

    def error_func(self, se2_k0: np.ndarray) -> np.ndarray:
        return se2_k0 - np.array([0.0, 0.0, 0.0])


def load_g2o(file_path: str):
    init_values = {}
    factor_graph = FactorGraph()
    vertex_num = 4000
    with open(file_path) as ifile:
        for line in ifile.readlines():
            items = line[:-1].split(" ")
            if items[0] == "EDGE_SE2":
                if int(items[1]) > vertex_num or int(items[2]) > vertex_num:
                    continue
                point_id0 = f"x{int(items[1])}"
                point_id1 = f"x{int(items[2])}"
                items_float = [float(i) for i in items[3:]]
                dx = items_float[0]
                dy = items_float[1]
                dtheta = items_float[2]
                dpose = np.array([dtheta, dx, dy])

                # if point_id0 == "x10" and point_id1 == "x11":
                #     print(dpose)
                i11, i12, i13, i22, i23, i33 = items_float[3:]
                matrix_i = np.array([[i11, i12, i13], [i12, i22, i23], [i13, i23, i33]])
                # loss = np.linalg.cholesky(matrix_i)
                factor_graph.add(BetweenFactor(point_id0, point_id1, dpose))
            elif items[0] == "VERTEX_SE2":
                if int(items[1]) > vertex_num:
                    continue
                point_id = f"x{int(items[1])}"
                x = float(items[2])
                y = float(items[3])
                theta = float(items[4])
                # if point_id == "x10" or point_id == "x11":
                #     print(point_id, theta, x, y)

                init_values[point_id] = np.array([theta, x, y])
            else:
                print(items)
                break
    return factor_graph, init_values
    # show_pose(init_values=init_values)
    # print(init_values)


def show_pose(init_values, color):
    data_x = [x[1] for x in init_values.values()]
    data_y = [x[2] for x in init_values.values()]
    plt.scatter(data_x, data_y, s=1, c=color)


def main():
    file_path = "tests/input_M3500_g2o.g2o"
    factor_graph, init_values = load_g2o(file_path)

    factor_graph.add(PriorFactor("x0"))
    solver = GaussNewtonOptimizer()
    # gn = LevenbergMarquardtOptimizer()
    draw = False
    if draw:
        plt.figure(figsize=(8, 8))
        show_pose(init_values, "red")

    start_time = perf_counter()
    solver.optimize_factor_graph(factor_graph, init_values, 8)
    end_time = perf_counter()
    print(f"{solver.__class__.__name__} takes {end_time-start_time:.3f} sec")
    if draw:
        show_pose(init_values, "blue")
        ax = plt.gca()
        ax.set_xlim((-50, 50))
        ax.set_ylim((-80, 20))
        plt.tight_layout()
        plt.show()
    # print("end")
    pass


if __name__ == "__main__":
    main()
