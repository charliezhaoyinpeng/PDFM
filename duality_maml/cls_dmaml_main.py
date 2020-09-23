import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import pandas as pd
import pickle
import copy

from os import listdir
from os.path import isfile, join
from cls_dmaml import *
from eval_metrics_cls import *


def mean(a):
    return sum(a).to(dtype=torch.float) / len(a)


def test_single_task(train_model_save_path, task, meta_lamb):
    # load the trained Fair MAML
    net = MAMLModel()
    net.load_state_dict(torch.load(train_model_save_path))

    criterion = nn.BCELoss()

    # load trained parameters
    temp_weights = [w.clone() for w in list(net.parameters())]
    try:
        temp_lambda = copy.deepcopy(meta_lamb)
    except:
        temp_lambda = (meta_lamb).clone()
    a_array = []

    # sample support data for testing
    K_Xy = task.sample(K)
    X = K_Xy[K_Xy.columns[-num_feature:]].copy().values
    y = K_Xy[["y"]].values
    z = K_Xy[["z"]].values
    z_bar = np.mean(z) * np.ones((len(z), 1))

    X = torch.tensor(X, dtype=torch.float).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float).unsqueeze(1)
    ones = torch.tensor(np.ones((len(y), 1)), dtype=torch.float).unsqueeze(1)
    z = torch.tensor(z, dtype=torch.float).unsqueeze(1)
    z_bar = torch.tensor(z_bar, dtype=torch.float).unsqueeze(1)

    for co_update in range(pd_updates):
        for step in range(inner_steps):
            y_hat = net.parameterised(X, temp_weights)
            fair = torch.abs(torch.mean((z - z_bar) * y_hat)) - c
            loss = (-1.0) * torch.mean(y * torch.log(y_hat) + (ones - y) * torch.log(ones - y_hat)) + temp_lambda * fair
            grad = torch.autograd.grad(loss.sum(), temp_weights)
            temp_weights = [w - inner_lr * g for w, g in zip(temp_weights, grad)]
        a_array.append(temp_weights)
        tilde_weight = (*map(mean, zip(*a_array))),
        gk = torch.abs(torch.mean((z - z_bar) * net.parameterised(X, tilde_weight))) - c
        temp_lambda = temp_lambda + dual_ud_lr * gk
        boolean = temp_lambda.item()
        if boolean > 0:
            temp_lambda = temp_lambda
        else:
            temp_lambda = 0


    # sample query data for testing
    K_Xy = task.sample(Kq)
    X = K_Xy[K_Xy.columns[-num_feature:]].copy().values
    y = K_Xy[["y"]].values
    z = K_Xy[["z"]].values
    X_temp = copy.deepcopy(X)
    z_temp = copy.deepcopy(z)
    z_bar = np.mean(z) * np.ones((len(z), 1))

    X = torch.tensor(X, dtype=torch.float).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float).unsqueeze(1)
    z = torch.tensor(z, dtype=torch.float).unsqueeze(1)
    z_bar = torch.tensor(z_bar, dtype=torch.float).unsqueeze(1)

    y_hat = net.parameterised(X, temp_weights)
    fair = torch.abs(torch.mean((z - z_bar) * y_hat))

    y_hat = y_hat.detach().numpy().reshape(len(y_hat), 1)
    y = y.detach().numpy().reshape(len(y), 1)

    input_zy = np.column_stack((z_temp, y_hat))
    yX = np.column_stack((y_hat, X_temp))

    accuracy = accuracy_score(y_hat.round(), y)
    discrimination = cal_discrimination(input_zy)
    consistency = cal_consistency(yX, num_neighbors)

    return [accuracy, fair, discrimination, consistency]


def test_tasks(tasks, meta_lamb):
    meta_accu = 0
    meta_fair = 0
    meta_disc = 0
    meta_consis = 0
    for i in range(tasks_per_meta_batch):
        task_name = random.choice(tasks)
        test_task = pd.read_csv(test_tasks_path + '/' + task_name)
        [t_accu, t_fair, t_disc, t_consis] = test_single_task(train_model_save_path, test_task, meta_lamb)
        meta_accu += t_accu
        meta_fair += t_fair
        meta_disc += t_disc
        meta_consis += t_consis
    avg_accu = meta_accu / tasks_per_meta_batch
    avg_fair = meta_fair / (tasks_per_meta_batch)
    avg_disc = meta_disc / (tasks_per_meta_batch)
    avg_consis = meta_consis / (tasks_per_meta_batch)

    return [avg_accu, avg_fair.item(), avg_disc, avg_consis]


if __name__ == "__main__":
    ####################################################################################################
    # hyperparameters
    inner_lr = 0.01  # learning rate of the inner loop
    dual_ud_lr = 0.01
    meta_lr = 0.001  # learning rate of the outer loop
    meta_dual_lr = 0.01

    lamb = 1
    K = 10  # K-shot per task for support data
    Kq = K * 2  # Kq-shot per task for query data
    c = 0.05  # fairness bound
    inner_steps = 1  # number of gradient steps in the inner loop, normally equals to 1 in MAML
    pd_updates = 10
    tasks_per_meta_batch = 8  # number of tasks sampled from tasks repository
    num_iterations = 4000  # number of iterations of the outer loop

    # other parameters
    num_feature = 98  # number of features of the input data for each task
    num_neighbors = 3
    plot_every = 3
    print_every = 25
    repeat = 5

    # PATHs
    train_tasks_path = r'train'
    train_model_save_path = r'train_model.pth'
    tr_val_meta_losses_faires_save_path = r'tr_val_meta_losses_faires_save.txt'

    test_tasks_path = r'test'
    val_tasks_path = r'val'
    ####################################################################################################

    ####################################################################################################

    for i in range(repeat):
        # -------------------------------------------  Training  ------------------------------------------#
        # load training tasks
        tasks_for_train = [f for f in listdir(train_tasks_path) if isfile(join(train_tasks_path, f))]

        # train maml, output train loss, and plot meta losses
        maml = MAML(MAMLModel(), tasks_for_train, num_feature, lamb, inner_lr, dual_ud_lr, meta_lr, meta_dual_lr, K, Kq, inner_steps, pd_updates, tasks_per_meta_batch, c, plot_every, print_every, val_tasks_path)
        [meta_weights, meta_lamb] = maml.main_loop(num_iterations, train_tasks_path)
        torch.save(maml.model.state_dict(), train_model_save_path)

        # writing losses and faires into a file
        with open(tr_val_meta_losses_faires_save_path, 'wb') as f:
            pickle.dump(maml.train_losses_list, f)
            pickle.dump(maml.train_faires_list, f)
            pickle.dump(maml.val_losses_list, f)
            pickle.dump(maml.val_faires_list, f)
        ####################################################################################################

        # load testing tasks
        tasks_for_test = [f for f in listdir(test_tasks_path) if isfile(join(test_tasks_path, f))]

        [test_accu, test_fair, test_disc, test_consis] = test_tasks(tasks_for_test, meta_lamb)
        print('---------------------------- %s / %s ------------------------------------------' % (i + 1, repeat))
        print('Accuracy for testing data =', test_accu)
        print('DBC for testing data =', test_fair)
        print('Discrimination for testing data =', test_disc)
        print('Consistency for testing data =', test_consis)
        print('-------------------------------------------------------------------------------')
