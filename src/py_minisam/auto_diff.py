from typing import List, Tuple

import numpy as np

EPISILON = 1.48e-8


def diff_2point(jac_shape, func: callable, *variables) -> np.ndarray:
    """
    2-point numeric finite difference. f'(x) = (f(x+h) - f(x)) / h
    Args:
        jac_shape: (residual_num, variable_num)
    """
    f0 = func(*variables)
    jac = np.zeros(jac_shape, dtype=np.float64)

    jac_col = 0
    for variable in variables:
        h = np.maximum(EPISILON * variable, EPISILON)
        for j in range(h.shape[0]):
            variable[j] += h[j]
            jac[:, jac_col] = (func(*variables) - f0) / h[j]
            variable[j] -= h[j]
            jac_col += 1
    return jac


def diff_3point(jac_shape, func: callable, *variables) -> np.ndarray:
    """
    3-point numeric finite difference. f'(x) = (f(x+h) - f(x-h)) / 2h
    :param jac_shape: tuple
        the shape of returned jacobian matrix.
    :param func: callable
        the function to evaluate jacobian matrix.
    :param variables: ndarrays
        At which the jacobian matrix is evaluated.
    :return:
    """
    jac = np.zeros(jac_shape, dtype=np.float64)

    jac_col = 0
    for variable in variables:
        h = np.maximum(EPISILON * variable, EPISILON)
        for j in range(h.shape[0]):
            variable[j] += h[j]
            f_plus = func(*variables)
            variable[j] -= 2 * h[j]
            f_subs = func(*variables)
            jac[:, jac_col] = (f_plus - f_subs) / (2 * h[j])
            variable[j] += h[j]
            jac_col += 1
    return jac


def diff_3point_inplace(
    func: callable, variable_col_idx_list: List[int], residual_row_start: int, jac: np.ndarray, *variables
):
    """
    3-point numeric finite difference. f'(x) = (f(x+h) - f(x-h)) / 2h
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
