from random import *
import numpy as np
import pandas as pd


def gen_syn_reg_task(task_num, save, num_datapoints):
    """
    randomly generate a synthetic task which containing "num_datapoints" data points
    each data points contains z(protected variable), y(target), and seven explanatory variables(x1 to x7)
    the protected variable z is binary, i.e. {0,1}
    according to the nature of z, target y is generate from two different Gaussian distributions with the same std but distinct mean
    """

    # group 1 (group b): unprotected group
    gauss_mean1 = np.random.uniform(0, 10)
    # group 0 (group a): protected group
    gauss_mean0 = gauss_mean1 + np.random.uniform(1, 5)

    task = []
    for i in range(num_datapoints):
        z = randint(0, 1)
        if z == 0:
            y = gauss(gauss_mean0, 1)
            if y < 0:
                y = 2 * gauss_mean0 - y
        else:
            y = gauss(gauss_mean1, 1)
            if y < 0:
                y = 2 * gauss_mean1 - y
        x1 = np.random.uniform(0, 1)
        x2 = np.random.uniform(0, 1)
        x3 = np.random.uniform(0, 1)
        x4 = np.random.uniform(0, 1)
        x5 = np.random.uniform(0, 1)
        x6 = np.random.uniform(0, 1)
        x7 = np.random.uniform(0, 1)
        datapoint = [z, y, x1, x2, x3, x4, x5, x6, x7]
        task.append(datapoint)

    title = ["z", "y", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    task = pd.DataFrame(task, columns=title)

    # normalize all attributes in task except z and y with zero mean and unit variance
    task_norm = (task - task.mean()) / task.std()
    task_norm['z'] = task['z']
    task_norm['y'] = task['y']

    task_norm.to_csv(save + '/task' + str(task_num) + '.csv')


def mean_difference(input_zy):
    """
    :param input_zy: numpy matrix
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


def cal_auc(input_zy):
    count_a = 0
    a_values = []
    b_values = []

    for line in input_zy:
        if line[0] == 0:
            a_values.append(line[1])
        elif line[0] == 1:
            b_values.append(line[1])

    for a in a_values:
        for b in b_values:
            if a >= b:
                count_a += 1
    auc_a = (count_a * 1.0) / (len(a_values) * len(b_values))
    if auc_a < 0.5:
        return 1 - auc_a
    else:
        return auc_a


def cal_ir(input_zy):
    a_values = []
    b_values = []

    for line in input_zy:
        if line[0] == 0:
            a_values.append(line[1])
        elif line[0] == 1:
            b_values.append(line[1])

    a = np.sum(a_values) * 1.0 / len(a_values)
    b = np.sum(b_values) * 1.0 / len(b_values)

    if a / b >= 1:
        return b / a
    else:
        return a / b


if __name__ == "__main__":
    train_save = r"../data/real_data/new_test"
    num_train_tasks = 10000
    total_absmd = 0
    total_auc = 0
    total_ir = 0
    for task_num in range(1, num_train_tasks + 1):
        # if task_num % 100 == 0:
        #     print("Generating==> %s/%s" % (task_num, num_train_tasks))
        # gen_syn_reg_task(task_num, train_save, 1000)
        # print(task_num)
        df = pd.read_csv(train_save + '/task' + str(task_num) + '.csv')
        zy = df[['z', 'y']].values
        task_absmd = mean_difference(zy)
        task_auc = cal_auc(zy)
        task_ir = cal_ir(zy)
        # print("task %s: MD=%s, AUC=%s, IR=%s" % (task_num, task_absmd, task_auc, task_ir))
        total_absmd += task_absmd
        total_auc += task_auc
        total_ir += task_ir
    print("Average abs MD=%s" % (total_absmd / num_train_tasks))
    print("Average AUC=%s" % (total_auc / num_train_tasks))
    print("Average IR=%s" % (total_ir / num_train_tasks))
