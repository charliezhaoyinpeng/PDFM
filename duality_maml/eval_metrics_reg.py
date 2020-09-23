import torch
import numpy as np


def abs_mean_difference(input_zy):
    """
    :param input_zy: 2D numpy matrix, for each element contains [z, y]
    :return: a scalar representing absolute mean difference regarding binary groups. MD=0 represents fairness.
    """
    a = 0
    a_size = 0
    b = 0
    b_size = 0
    for line in input_zy:
        if line[0] == 0:
            a += line[1]
            a_size += 1
        elif line[0] == 1:
            b += line[1]
            b_size += 1
    return np.abs(a / a_size - b / b_size)


def cal_auc(y_a, a_size, y_b, b_size):
    """
    :param y_a: torch tensor
    :param a_size: scalar
    :param y_b: torch tensor
    :param b_size: scalar
    """
    y_a = y_a.cpu().detach().numpy()
    y_b = y_b.cpu().detach().numpy()

    count_a = 0
    for a in y_a:
        for b in y_b:
            if a >= b:
                count_a += 1
    auc_a = (count_a * 1.0) / (a_size * b_size)
    if auc_a < 0.5:
        return 1 - auc_a
    else:
        return auc_a


def cal_ir(y_a, a_size, y_b, b_size):
    y_a = y_a.cpu().detach().numpy()
    y_b = y_b.cpu().detach().numpy()
    a = np.sum(y_a) * 1.0 / a_size
    b = np.sum(y_b) * 1.0 / b_size

    if np.abs(a / b) >= 1:
        return np.abs(b / a)
    else:
        return np.abs(a / b)
