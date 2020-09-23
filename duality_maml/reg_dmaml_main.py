from __future__ import division
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import pandas as pd
import pickle
from os import listdir
from os.path import isfile, join

from reg_dmaml import *
from eval_metrics_reg import *


def mean(a):
    return sum(a).to(dtype=torch.float) / len(a)


def test_single_task(train_model_save_path, task, meta_lamb):
    # load the trained Fair MAML
    net = MAMLModel()
    net.load_state_dict(torch.load(train_model_save_path))

    criterion = nn.MSELoss()

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

    X_a = 0
    X_b = 0
    a_size, b_size = 0, 0
    for i in range(len(z)):
        if z[i] == 0:
            temp_a = np.array([X[i]])
            if type(X_a) is int:
                X_a = temp_a
            else:
                X_a = np.concatenate((X_a, temp_a))
            a_size += 1

        elif z[i] == 1:
            temp_b = np.array([X[i]])
            if type(X_b) is int:
                X_b = temp_b
            else:
                X_b = np.concatenate((X_b, temp_b))
            b_size += 1

    X = torch.tensor(X, dtype=torch.float).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float).unsqueeze(1)

    try:
        X_a = torch.tensor(X_a, dtype=torch.float).unsqueeze(1)
        X_b = torch.tensor(X_b, dtype=torch.float).unsqueeze(1)
    except:
        pass

    if a_size == 0 or b_size == 0:
        for step in range(inner_steps):
            loss = criterion(net.parameterised(X, temp_weights), y) / K
            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - inner_lr * g for w, g in zip(temp_weights, grad)]
    else:
        for co_update in range(pd_updates):
            for step in range(inner_steps):
                y_hat = net.parameterised(X, temp_weights)
                md = temp_lambda * (torch.abs(
                    torch.sum(net.parameterised(X_a, temp_weights) / a_size) - torch.sum(net.parameterised(X_b, temp_weights) / b_size)) - c)
                loss = torch.mean((y_hat - y) ** 2) + md
                loss = loss / K
                grad = torch.autograd.grad(loss.sum(), temp_weights)
                temp_weights = [w - inner_lr * g for w, g in zip(temp_weights, grad)]
            a_array.append(temp_weights)
            tilde_weight = (*map(mean, zip(*a_array))),

            gk = (torch.abs(torch.sum(net.parameterised(X_a, tilde_weight) / a_size) - torch.sum(net.parameterised(X_b, tilde_weight) / b_size)) - c)
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

    X_a = 0
    X_b = 0
    a_size, b_size = 0, 0
    for i in range(len(z)):
        if z[i] == 0:
            temp_a = np.array([X[i]])
            if type(X_a) is int:
                X_a = temp_a
            else:
                X_a = np.concatenate((X_a, temp_a))
            a_size += 1

        elif z[i] == 1:
            temp_b = np.array([X[i]])
            if type(X_b) is int:
                X_b = temp_b
            else:
                X_b = np.concatenate((X_b, temp_b))
            b_size += 1

    X = torch.tensor(X, dtype=torch.float).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float).unsqueeze(1)

    if a_size == 0 or b_size == 0:
        loss = criterion(net.parameterised(X, temp_weights), y) / Kq
        md = 'skipped'
        auc = 'skipped'
        ir = 'skipped'
    else:
        X_a = torch.tensor(X_a, dtype=torch.float).unsqueeze(1)
        X_b = torch.tensor(X_b, dtype=torch.float).unsqueeze(1)

        y_hat = net.parameterised(X, temp_weights)
        y_a_hat = net.parameterised(X_a, temp_weights)
        y_b_hat = net.parameterised(X_b, temp_weights)

        md = torch.abs(torch.sum(y_a_hat / a_size) - torch.sum(y_b_hat / b_size))
        auc = cal_auc(y_a_hat, a_size, y_b_hat, b_size)
        ir = cal_ir(y_a_hat, a_size, y_b_hat, b_size)
        loss = criterion(y_hat, y)
        loss = loss / Kq

    return [loss, md, auc, ir]


def test_tasks(tasks, meta_lamb):
    meta_loss = 0
    meta_fair = 0
    meta_auc = 0
    meta_ir = 0
    num_deduct_fair = 0
    num_deduct_auc = 0
    num_deduct_ir = 0
    for i in range(tasks_per_meta_batch):
        task_name = random.choice(tasks)
        test_task = pd.read_csv(test_tasks_path + '/' + task_name)
        [t_loss, t_fair, t_auc, t_ir] = test_single_task(train_model_save_path, test_task, meta_lamb)
        meta_loss += t_loss

        if type(t_fair) is not str:
            meta_fair += t_fair
        else:
            num_deduct_fair += 1

        if type(t_auc) is not str:
            meta_auc += t_auc
        else:
            num_deduct_auc += 1
        if type(t_ir) is not str:
            meta_ir += t_ir
        else:
            num_deduct_ir += 1

    avg_MSE = meta_loss / tasks_per_meta_batch
    try:
        avg_fair = meta_fair / (tasks_per_meta_batch - num_deduct_fair)
        ans_fair = avg_fair.item()
    except:
        avg_fair = meta_fair / (tasks_per_meta_batch)
        ans_fair = avg_fair
    try:
        avg_auc = meta_auc / (tasks_per_meta_batch - num_deduct_auc)
    except:
        avg_auc = meta_auc / (tasks_per_meta_batch)
    try:
        avg_ir = meta_ir / (tasks_per_meta_batch - num_deduct_ir)
    except:
        avg_ir = meta_ir / (tasks_per_meta_batch)

    return [avg_MSE.item(), ans_fair, avg_auc, avg_ir]

if __name__ == "__main__":
    ####################################################################################################
    # hyperparameters
    inner_lr = 0.01  # learning rate of the inner loop
    dual_ud_lr = 0.01
    meta_lr = 0.001  # learning rate of the outer loop
    meta_dual_lr = 0.01

    K = 5  # K-shot per task for support data
    Kq = K * 2  # Kq-shot per task for query data
    c = 0.05  # fairness bound
    inner_steps = 1  # number of gradient steps in the inner loop, normally equals to 1 in MAML
    pd_updates = 10
    tasks_per_meta_batch = 8  # number of tasks sampled from tasks repository
    num_iterations = 4000  # number of iterations of the outer loop
    lamb = 0.0001

    # other parameters
    num_feature = 13  # number of features of the input data for each task
    plot_every = 1
    print_every = 25
    repeat = 50

    # PATHs
    train_tasks_path = r'train'
    train_model_save_path = r'train_model.pth'
    tr_val_meta_losses_faires_save_path = r'tr_val_meta_losses_faires_save.txt'

    test_tasks_path = r'test'
    val_tasks_path = r'val'
    ####################################################################################################

    ####################################################################################################

    for i in range(repeat):
        print("===============================================================================================")
        # -------------------------------------------  Training  ------------------------------------------#
        # load training tasks
        tasks_for_train = [f for f in listdir(train_tasks_path) if isfile(join(train_tasks_path, f))]

        # train maml, output train loss, and plot meta losses
        maml = MAML(MAMLModel(), tasks_for_train, num_feature, inner_lr, dual_ud_lr, meta_lr, meta_dual_lr, lamb, K, Kq, inner_steps, pd_updates, tasks_per_meta_batch, c, plot_every, print_every, val_tasks_path)
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

        [test_MSE, test_fair, test_auc, test_ir] = test_tasks(tasks_for_test, meta_lamb)
        print('---------------------------- %s / %s ------------------------------------------' % (i + 1, repeat))
        print('MSE for testing data =', test_MSE)
        print('MD for testing data =', test_fair)
        print('AUC for testing data =', test_auc)
        print('IR for testing data =', test_ir)
