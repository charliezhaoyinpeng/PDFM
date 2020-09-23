import torch
import math
import numpy as np


def cal_discrimination(input_zy):
    a_values = []
    b_values = []

    for line in input_zy:
        if line[0] == 0:
            a_values.append(line[1])
        elif line[0] == 1:
            b_values.append(line[1])

    if len(a_values) == 0:
        discrimination = sum(b_values) * 1.0 / len(b_values)
    elif len(b_values) == 0:
        discrimination = sum(a_values) * 1.0 / len(a_values)
    else:
        discrimination = sum(a_values) * 1.0 / len(a_values) - sum(b_values) * 1.0 / len(b_values)
    return abs(discrimination)


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


# Locate the most similar neighbors
# example: get_neighbors(yX, X[0], 3)
def get_neighbors(yX, target_row, num_neighbors):
    distances = list()
    for yX_row in yX:
        X_row = yX_row[1:]
        y = yX_row[0]
        dist = euclidean_distance(target_row, X_row)
        distances.append((y, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def cal_consistency(yX, num_neighbors):
    ans = 0
    for yX_row in yX:
        temp = 0
        target_row = yX_row[1:]
        target_y = yX_row[0]
        y_neighbors = get_neighbors(yX, target_row, num_neighbors)
        for y_neighbor in y_neighbors:
            temp += abs(target_y - y_neighbor)
        ans += temp
    return (1 - (ans * 1.0) / (len(yX) * num_neighbors))


def cal_dbc(input_zy):
    length = len(input_zy)
    z_bar = np.mean(input_zy[:, 0])
    dbc = 0
    for zy in input_zy:
        dbc += (zy[0] - z_bar) * zy[1] * 1.0
    return abs(dbc / length)