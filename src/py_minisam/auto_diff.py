from typing import List

import numpy as np

EPISILON = 1.48e-8


def diff_3point_inplace(
    func: callable, variable_col_idx_list: List[int], residual_row_start: int, jac: np.ndarray, *variables
):
    """3-point numeric finite difference. f'(x) = (f(x+h) - f(x-h)) / 2h
    :param jac_shape: tuple
        the shape of returned jacobian matrix.
    :param func: callable
        the function to evaluate jacobian matrix.
    :param variables: ndarrays
        At which the jacobian matrix is evaluated.
    :return:
    """
    for col, variable in zip(variable_col_idx_list, variables):
        h = np.maximum(EPISILON * variable, EPISILON)
        for i in range(h.shape[0]):
            variable[i] += h[i]
            f_plus = func(*variables)
            variable[i] -= 2 * h[i]
            f_subs = func(*variables)
            slope = (f_plus - f_subs) / (2 * h[i])
            jac[residual_row_start : residual_row_start + slope.size, col + i] = slope
            variable[i] += h[i]
