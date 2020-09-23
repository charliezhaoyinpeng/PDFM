from __future__ import division
from memory_profiler import profile

import torch
import torch.nn as nn
import random
import math
import copy, time
import pandas as pd
import numpy as np
from collections import OrderedDict
from os import listdir
from os.path import isfile, join

from eval_metrics_reg import *


class MAMLModel(nn.Module):
    def __init__(self):
        super(MAMLModel, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(13, 40)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(40, 1))
        ]))

    def forward(self, x):
        return self.model(x)

    def parameterised(self, x, weights):
        # like forward, but uses ``weights`` instead of ``model.parameters()``
        # it'd be nice if this could be generated automatically for any nn.Module...
        x = nn.functional.linear(x, weights[0], weights[1])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[2], weights[3])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[4], weights[5])
        return x


class MAML():
    """
    This code implements MAML for supervised few-shot regression learning.
    """

    def __init__(self, model, tasks, num_feature, inner_lr, dual_ud_lr, meta_lr, meta_dual_lr, lamb, K, Kq, inner_steps, pd_updates, tasks_per_meta_batch, c, plot_every, print_every, val_tasks_path):
        """
            tasks: name collection of each task -- list of strings
            task: df -- pandas data frame
        """

        # important objects
        self.tasks = tasks
        self.model = model
        self.num_feature = num_feature
        self.weights = list(model.parameters())  # the maml weights (primal-variable) we will be meta-optimising
        self.lamb = lamb  # the dual-variable we will be meta-optimising
        self.criterion = nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)

        # hyperparameters
        self.inner_lr = inner_lr
        self.dual_ud_lr = dual_ud_lr
        self.meta_lr = meta_lr
        self.meta_dual_lr = meta_dual_lr
        self.K = K
        self.Kq = Kq
        self.inner_steps = inner_steps  # with the current design of MAML, >1 is unlikely to work well
        self.pd_updates = pd_updates
        self.tasks_per_meta_batch = tasks_per_meta_batch
        self.c = c

        # metrics
        self.plot_every = plot_every
        self.print_every = print_every

        self.meta_losses = []
        self.meta_faires = []

        self.train_losses_list = []
        self.train_faires_list = []

        self.val_losses_list = []
        self.val_faires_list = []

        # others
        self.val_tasks_path = val_tasks_path

    def mean(self, a):
        return sum(a).to(dtype=torch.float) / len(a)

    def inner_loop(self, task):
        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]
        try:
            temp_lambda = copy.deepcopy(self.lamb)
        except:
            temp_lambda = (self.lamb).clone()
        a_array = []

        # sample K-shot support data from the task
        K_Xy = task.sample(self.K)
        X = K_Xy[K_Xy.columns[-self.num_feature:]].copy().values
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

        # print(a_size, b_size)
        X = torch.tensor(X, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float).unsqueeze(1)

        try:
            X_a = torch.tensor(X_a, dtype=torch.float).unsqueeze(1)
            X_b = torch.tensor(X_b, dtype=torch.float).unsqueeze(1)
        except:
            pass

        if a_size == 0 or b_size == 0:
            for step in range(self.inner_steps):
                loss = self.criterion(self.model.parameterised(X, temp_weights), y) / self.K
                # compute grad and update inner loop weights
                grad = torch.autograd.grad(loss, temp_weights)
                temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
        else:
            for co_update in range(self.pd_updates):
                # inner_steps: number of steps of gradient, it is greater or equals to one in MAML
                for step in range(self.inner_steps):
                    y_hat = self.model.parameterised(X, temp_weights)
                    l_md = temp_lambda * (torch.abs(torch.sum(self.model.parameterised(X_a, temp_weights) / a_size) - torch.sum(self.model.parameterised(X_b, temp_weights) / b_size)) - self.c)
                    loss = torch.mean((y_hat - y) ** 2) + l_md
                    loss = loss / self.K
                    # compute grad and update inner loop weights
                    grad = torch.autograd.grad(loss.sum(), temp_weights)
                    temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
                a_array.append(temp_weights)
                tilde_weight = (*map(self.mean, zip(*a_array))),

                gk = (torch.abs(torch.sum(self.model.parameterised(X_a, tilde_weight) / a_size) - torch.sum(self.model.parameterised(X_b, tilde_weight) / b_size)) - self.c)
                temp_lambda = temp_lambda + self.dual_ud_lr * gk
                boolean = temp_lambda.item()
                if boolean > 0:
                    temp_lambda = temp_lambda
                else:
                    temp_lambda = 0

        # sample new K-shot query data from the task for meta-update and compute loss
        K_Xy = task.sample(self.Kq)
        X = K_Xy[K_Xy.columns[-self.num_feature:]].copy().values
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
            loss = self.criterion(self.model.parameterised(X, temp_weights), y) / self.Kq
            if a_size == 0:
                X_b = torch.tensor(X_b, dtype=torch.float).unsqueeze(1)
                md = torch.abs(torch.sum(self.model.parameterised(X_b, temp_weights) / b_size))
            elif b_size == 0:
                X_a = torch.tensor(X_a, dtype=torch.float).unsqueeze(1)
                md = torch.abs(torch.sum(self.model.parameterised(X_a, temp_weights) / a_size))
        else:
            X_a = torch.tensor(X_a, dtype=torch.float).unsqueeze(1)
            X_b = torch.tensor(X_b, dtype=torch.float).unsqueeze(1)
            y_hat = self.model.parameterised(X, temp_weights)
            md = torch.abs(
                torch.sum(self.model.parameterised(X_a, temp_weights) / a_size) - torch.sum(self.model.parameterised(X_b, temp_weights) / b_size))
            loss = self.criterion(y_hat, y)
            loss = loss / self.Kq

        return [loss, md]

    def val_single_task(self, task):
        # load trained parameters
        temp_weights = [w.clone() for w in list(self.model.parameters())]
        try:
            temp_lambda = copy.deepcopy(self.lamb)
        except:
            temp_lambda = (self.lamb).clone()
        a_array = []

        # sample support data for testing
        K_Xy = task.sample(self.K)
        X = K_Xy[K_Xy.columns[-self.num_feature:]].copy().values
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
            for step in range(self.inner_steps):
                loss = self.criterion(self.model.parameterised(X, temp_weights), y) / self.K
                grad = torch.autograd.grad(loss, temp_weights)
                temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
        else:
            for co_update in range(self.pd_updates):
                for step in range(self.inner_steps):
                    y_hat = self.model.parameterised(X, temp_weights)
                    md = temp_lambda * (torch.abs(
                        torch.sum(self.model.parameterised(X_a, temp_weights) / a_size) - torch.sum(self.model.parameterised(X_b, temp_weights) / b_size)) - self.c)
                    loss = torch.mean((y_hat - y) ** 2) + md
                    loss = loss / self.K
                    grad = torch.autograd.grad(loss.sum(), temp_weights)
                    temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
                a_array.append(temp_weights)
                tilde_weight = (*map(self.mean, zip(*a_array))),

                gk = (torch.abs(torch.sum(self.model.parameterised(X_a, tilde_weight) / a_size) - torch.sum(self.model.parameterised(X_b, tilde_weight) / b_size)) - self.c)
                temp_lambda = temp_lambda + self.dual_ud_lr * gk
                boolean = temp_lambda.item()
                if boolean < 0:
                    temp_lambda = 0

        # sample query data for testing
        K_Xy = task.sample(self.Kq)
        X = K_Xy[K_Xy.columns[-self.num_feature:]].copy().values
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
            loss = self.criterion(self.model.parameterised(X, temp_weights), y) / self.Kq
            md = 'skipped'
            auc = 'skipped'
            ir = 'skipped'
        else:
            X_a = torch.tensor(X_a, dtype=torch.float).unsqueeze(1)
            X_b = torch.tensor(X_b, dtype=torch.float).unsqueeze(1)

            y_hat = self.model.parameterised(X, temp_weights)
            y_a_hat = self.model.parameterised(X_a, temp_weights)
            y_b_hat = self.model.parameterised(X_b, temp_weights)

            md = torch.abs(torch.sum(y_a_hat / a_size) - torch.sum(y_b_hat / b_size))
            auc = cal_auc(y_a_hat, a_size, y_b_hat, b_size)
            ir = cal_ir(y_a_hat, a_size, y_b_hat, b_size)
            loss = self.criterion(y_hat, y)
            loss = loss / self.Kq

        return [loss, md, auc, ir]

    def val_tasks(self, tasks):
        meta_loss = 0
        meta_fair = 0
        meta_auc = 0
        meta_ir = 0
        num_deduct_fair = 0
        num_deduct_auc = 0
        num_deduct_ir = 0
        for i in range(self.tasks_per_meta_batch):
            task_name = random.choice(tasks)
            val_task = pd.read_csv(self.val_tasks_path + '/' + task_name)
            [t_loss, t_fair, t_auc, t_ir] = self.val_single_task(val_task)
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
        avg_MSE = meta_loss / self.tasks_per_meta_batch
        try:
            avg_fair = meta_fair / (self.tasks_per_meta_batch - num_deduct_fair)
            ans_fair = avg_fair.item()
        except:
            avg_fair = meta_fair / (self.tasks_per_meta_batch)
            ans_fair = avg_fair
        try:
            avg_auc = meta_auc / (self.tasks_per_meta_batch - num_deduct_auc)
        except:
            avg_auc = meta_auc / (self.tasks_per_meta_batch)
        try:
            avg_ir = meta_ir / (self.tasks_per_meta_batch - num_deduct_ir)
        except:
            avg_ir = meta_ir / (self.tasks_per_meta_batch)

        return [avg_MSE.item(), ans_fair, avg_auc, avg_ir]

    # @profile
    def main_loop(self, num_iterations, tasks_path):
        train_loss = 0
        train_fair = 0

        for iteration in range(1, num_iterations + 1):
            torch.cuda.empty_cache()
            # compute meta loss
            meta_loss = 0
            meta_fair = 0

            start_time = time.time()

            for i in range(self.tasks_per_meta_batch):
                task_name = random.choice(self.tasks)
                task = pd.read_csv(tasks_path + '/' + task_name)
                [t_loss, t_fair] = self.inner_loop(task)
                meta_loss += t_loss
                meta_fair += t_fair

            # compute meta gradient of loss with respect to maml weights
            meta_grads = torch.autograd.grad(meta_loss, self.weights)

            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()

            # update meta dual variable
            dual_update = self.lamb + self.meta_dual_lr * (meta_fair - self.tasks_per_meta_batch * self.c)
            if dual_update.item() > 0:
                self.lamb = dual_update
            else:
                self.lamb = 0

            # log metrics
            train_loss += meta_loss.item() / self.tasks_per_meta_batch
            train_fair += meta_fair.item() / self.tasks_per_meta_batch

            self.train_losses_list.append(train_loss)
            self.train_faires_list.append(train_fair)

            tasks_for_val = [f for f in listdir(self.val_tasks_path) if isfile(join(self.val_tasks_path, f))]
            [val_MSE, val_fair, val_auc, val_ir] = self.val_tasks(tasks_for_val)

            self.val_losses_list.append(val_MSE)
            self.val_faires_list.append(val_fair)

            if iteration % self.print_every == 0:
                print("{}/{}. tr_loss: {}; tr_fair: {}; val_loss: {}; val_fair: {} -----> running time: {} seconds.".format(iteration, num_iterations, np.round(train_loss / self.plot_every, 4),
                                                                                           np.round(train_fair / self.plot_every, 4), np.round(val_MSE, 4),
                                                                                           np.round(val_fair, 4), np.round((time.time() - start_time), 4)))
            if iteration % self.plot_every == 0:
                self.meta_losses.append(train_loss / self.plot_every)
                train_loss = 0
                self.meta_faires.append(train_fair / self.plot_every)
                train_fair = 0

        print("output lambda: ", self.lamb)
        return [self.weights, self.lamb]
