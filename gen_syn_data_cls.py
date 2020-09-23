import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for plotting stuff
from random import seed, shuffle
from scipy.stats import multivariate_normal  # generating synthetic data


# SEED = 1122334455
# seed(SEED)  # set the random seed so that the random permutations can be reproduced again
# np.random.seed(SEED)

def cal_discrimination(input_zy):
    a_values = []
    b_values = []

    for line in input_zy:
        if line[0] == 0:
            a_values.append(line[1])
        elif line[0] == 1:
            b_values.append(line[1])

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


def generate_synthetic_data(plot_data=False):
    """
        Code for generating the synthetic data.
        We will have two non-sensitive features and one sensitive feature.
        A sensitive feature value of 0.0 means the example is considered to be in protected group (e.g., female) and 1.0 means it's in non-protected group (e.g., male).
    """
    n_samples = 1000  # generate these many data points per class
    disc_factor_range = [2.0, 4.0, 8.0, 16.0]
    disc_factor = math.pi / random.choice(disc_factor_range)  # this variable determines the initial discrimination in the data -- decraese it to generate more discrimination

    def gen_gaussian(mean_in, cov_in, class_label):
        nv = multivariate_normal(mean=mean_in, cov=cov_in)
        X = nv.rvs(n_samples)
        y = np.ones(n_samples, dtype=float) * class_label
        return nv, X, y

    """ Generate the non-sensitive features randomly """
    # We will generate one gaussian cluster for each class
    mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
    mu2, sigma2 = [-2, -2], [[10, 1], [1, 3]]
    nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1)  # positive class
    nv2, X2, y2 = gen_gaussian(mu2, sigma2, 0)  # negative class

    # join the posisitve and negative class clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # shuffle the data
    perm = range(0, n_samples * 2)
    shuffle(perm)
    X = X[perm]
    y = y[perm]

    rotation_mult = np.array([[math.cos(disc_factor), -math.sin(disc_factor)], [math.sin(disc_factor), math.cos(disc_factor)]])
    X_aux = np.dot(X, rotation_mult)

    """ Generate the sensitive feature here """
    x_control = []  # this array holds the sensitive feature value
    for i in range(0, len(X)):
        x = X_aux[i]

        # probability for each cluster that the point belongs to it
        p1 = nv1.pdf(x)
        p2 = nv2.pdf(x)

        # normalize the probabilities from 0 to 1
        s = p1 + p2
        p1 = p1 / s
        p2 = p2 / s

        r = np.random.uniform()  # generate a random number from 0 to 1

        if r < p1:  # the first cluster is the positive class
            x_control.append(1.0)  # 1.0 means its male
        else:
            x_control.append(0.0)  # 0.0 -> female

    x_control = np.array(x_control)

    """ Show the data """
    if plot_data:
        num_to_draw = 200  # we will only draw a small number of points to avoid clutter
        x_draw = X[:num_to_draw]
        y_draw = y[:num_to_draw]
        x_control_draw = x_control[:num_to_draw]

        X_s_0 = x_draw[x_control_draw == 0.0]
        X_s_1 = x_draw[x_control_draw == 1.0]
        y_s_0 = y_draw[x_control_draw == 0.0]
        y_s_1 = y_draw[x_control_draw == 1.0]
        plt.scatter(X_s_0[y_s_0 == 1.0][:, 0], X_s_0[y_s_0 == 1.0][:, 1], color='green', marker='x', s=30, linewidth=1.5, label="Prot. +ve")
        plt.scatter(X_s_0[y_s_0 == -1.0][:, 0], X_s_0[y_s_0 == -1.0][:, 1], color='red', marker='x', s=30, linewidth=1.5, label="Prot. -ve")
        plt.scatter(X_s_1[y_s_1 == 1.0][:, 0], X_s_1[y_s_1 == 1.0][:, 1], color='green', marker='o', facecolors='none', s=30, label="Non-prot. +ve")
        plt.scatter(X_s_1[y_s_1 == -1.0][:, 0], X_s_1[y_s_1 == -1.0][:, 1], color='red', marker='o', facecolors='none', s=30, label="Non-prot. -ve")

        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')  # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.legend(loc=2, fontsize=15)
        plt.xlim((-15, 10))
        plt.ylim((-10, 15))
        plt.savefig("img/data.png")
        plt.show()

    # x_control = {"s1": x_control}  # all the sensitive features are stored in a dictionary
    title = ["z", "y", "x1", "x2"]
    zy = np.column_stack((x_control, y))
    # print("discrimination: ", cal_discrimination(zy))
    # print("DBC: ", cal_dbc(zy))

    yX = np.column_stack((y, X))
    # print("consistency: ", cal_consistency(yX, 20))

    task = np.column_stack((zy, X))
    task = pd.DataFrame(task, columns=title)

    # normalize all attributes in task except z and y with zero mean and unit variance
    task_norm = (task - task.mean()) / task.std()
    task_norm['z'] = task['z']
    task_norm['y'] = task['y']

    return task_norm


def generate_tasks(save, num_tasks):
    for i in range(num_tasks):
        print("generating task: ", i + 1)
        task = generate_synthetic_data()
        task.to_csv(save + '/task' + str(i + 1) + '.csv')


def tasks_evaluation(save, num_tasks, num_neighbors):
    total_dbc = 0
    total_discrimination = 0
    total_consistency = 0
    for task_num in range(1, num_tasks + 1):
        df = pd.read_csv(save + '/task' + str(task_num) + '.csv')
        zy = df[['z', 'y']].values
        yX = df[['y', 'x1', 'x2']].values
        discrimination = cal_discrimination(zy)
        consistency = cal_consistency(yX, num_neighbors)
        dbc = cal_dbc(zy)
        print("task %s: dbc=%s, discrimination=%s, consistency=%s" % (task_num, dbc, discrimination, consistency))
        total_dbc += dbc
        total_discrimination += discrimination
        total_consistency += consistency

    print("#################################################################################################")
    print("Average dbc=%s" % (total_dbc / num_tasks))
    print("Average discrimination=%s" % (total_discrimination / num_tasks))
    print("Average consistency=%s" % (total_consistency / num_tasks))


if __name__ == "__main__":
    # parameters
    ###################################################################
    # tasks generation
    save = r"../../Data/syn_data_cls/test"
    num_tasks = 1000

    ###################################################################

    # generate_tasks(save, num_tasks)

    tasks_evaluation(save, num_tasks, 5)
