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
from sklearn.metrics import accuracy_score

from eval_metrics_cls import *


class MAMLModel(nn.Module):
    def __init__(self):
        super(MAMLModel, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(98, 40)),
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
        return torch.sigmoid(x)


class MAML():
    """
    This code implements MAML for supervised few-shot regression learning.
    """

    def __init__(self, model, tasks, num_feature, lamb, inner_lr, dual_ud_lr, meta_lr, meta_dual_lr, K, Kq, inner_steps, pd_updates, tasks_per_meta_batch, c, plot_every, print_every, val_tasks_path):
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
        self.criterion = nn.BCELoss()
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
        self.meta_accus = []

        self.train_losses_list = []
        self.train_faires_list = []
        self.train_accus_list = []

        self.val_losses_list = []
        self.val_faires_list = []
        self.val_accus_list = []

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
        z_bar = np.mean(z) * np.ones((len(z), 1))

        X = torch.tensor(X, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float).unsqueeze(1)
        ones = torch.tensor(np.ones((len(y), 1)), dtype=torch.float).unsqueeze(1)
        z = torch.tensor(z, dtype=torch.float).unsqueeze(1)
        z_bar = torch.tensor(z_bar, dtype=torch.float).unsqueeze(1)

        for co_update in range(self.pd_updates):
            # inner_steps: number of steps of gradient, it is greater or equals to one in MAML
            for step in range(self.inner_steps):
                y_hat = self.model.parameterised(X, temp_weights)
                fair = torch.abs(torch.mean((z - z_bar) * y_hat)) - self.c
                loss = (-1.0) * torch.mean(y * torch.log(y_hat) + (ones - y) * torch.log(ones - y_hat)) + temp_lambda * fair
                grad = torch.autograd.grad(loss.sum(), temp_weights)
                temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
            a_array.append(temp_weights)

            tilde_weight = (*map(self.mean, zip(*a_array))),

            gk = torch.abs(torch.mean((z - z_bar) * self.model.parameterised(X, tilde_weight))) - self.c
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
        z_bar = np.mean(z) * np.ones((len(z), 1))

        X = torch.tensor(X, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float).unsqueeze(1)
        z = torch.tensor(z, dtype=torch.float).unsqueeze(1)
        z_bar = torch.tensor(z_bar, dtype=torch.float).unsqueeze(1)

        y_hat = self.model.parameterised(X, temp_weights)
        fair = torch.abs(torch.mean((z - z_bar) * y_hat))
        loss = self.criterion(y_hat, y)

        y_hat = y_hat.detach().numpy().reshape(len(y_hat), 1)
        y = y.detach().numpy().reshape(len(y), 1)
        accuracy = accuracy_score(y_hat.round(), y)

        return [loss, fair, accuracy]

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
        z_bar = np.mean(z) * np.ones((len(z), 1))

        X = torch.tensor(X, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float).unsqueeze(1)
        ones = torch.tensor(np.ones((len(y), 1)), dtype=torch.float).unsqueeze(1)
        z = torch.tensor(z, dtype=torch.float).unsqueeze(1)
        z_bar = torch.tensor(z_bar, dtype=torch.float).unsqueeze(1)

        for co_update in range(self.pd_updates):
            for step in range(self.inner_steps):
                y_hat = self.model.parameterised(X, temp_weights)
                fair = torch.abs(torch.mean((z - z_bar) * y_hat)) - self.c
                loss = (-1.0) * torch.mean(y * torch.log(y_hat) + (ones - y) * torch.log(ones - y_hat)) + temp_lambda * fair
                grad = torch.autograd.grad(loss.sum(), temp_weights)
                temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
            a_array.append(temp_weights)
            tilde_weight = (*map(self.mean, zip(*a_array))),
            gk = torch.abs(torch.mean((z - z_bar) * self.model.parameterised(X, tilde_weight))) - self.c
            temp_lambda = temp_lambda + self.dual_ud_lr * gk
            boolean = temp_lambda.item()
            if boolean < 0:
                temp_lambda = 0

        # sample query data for testing
        K_Xy = task.sample(self.Kq)
        X = K_Xy[K_Xy.columns[-self.num_feature:]].copy().values
        y = K_Xy[["y"]].values
        z = K_Xy[["z"]].values
        z_bar = np.mean(z) * np.ones((len(z), 1))

        X = torch.tensor(X, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float).unsqueeze(1)
        z = torch.tensor(z, dtype=torch.float).unsqueeze(1)
        z_bar = torch.tensor(z_bar, dtype=torch.float).unsqueeze(1)

        y_hat = self.model.parameterised(X, temp_weights)
        fair = torch.abs(torch.mean((z - z_bar) * y_hat))
        loss = self.criterion(y_hat, y)

        y_hat = y_hat.detach().numpy().reshape(len(y_hat), 1)
        y = y.detach().numpy().reshape(len(y), 1)
        accuracy = accuracy_score(y_hat.round(), y)

        return [loss, fair, accuracy]

    def val_tasks(self, tasks):
        meta_loss = 0
        meta_fair = 0
        meta_accu = 0
        for i in range(self.tasks_per_meta_batch):
            task_name = random.choice(tasks)
            val_task = pd.read_csv(self.val_tasks_path + '/' + task_name)
            [t_loss, t_fair, t_accu] = self.val_single_task(val_task)
            meta_loss += t_loss
            meta_fair += t_fair
            meta_accu += t_accu
        avg_loss = meta_loss / self.tasks_per_meta_batch
        avg_fair = meta_fair / self.tasks_per_meta_batch
        avg_accu = meta_accu / self.tasks_per_meta_batch
        return [avg_loss.item(), avg_fair.item(), avg_accu]

    def main_loop(self, num_iterations, tasks_path):
        train_loss = 0
        train_fair = 0
        train_accu = 0
        for iteration in range(1, num_iterations + 1):
            # compute meta loss
            meta_loss = 0
            meta_fair = 0
            meta_accu = 0

            start_time = time.time()

            for i in range(self.tasks_per_meta_batch):
                task_name = random.choice(self.tasks)
                task = pd.read_csv(tasks_path + '/' + task_name)
                [t_loss, t_fair, t_accu] = self.inner_loop(task)
                meta_loss += t_loss
                meta_fair += t_fair
                meta_accu += t_accu

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
            train_accu += meta_accu / self.tasks_per_meta_batch

            self.train_losses_list.append(train_loss)
            self.train_faires_list.append(train_fair)
            self.train_accus_list.append(train_accu)

            tasks_for_val = [f for f in listdir(self.val_tasks_path) if isfile(join(self.val_tasks_path, f))]
            [val_loss, val_fair, val_accu] = self.val_tasks(tasks_for_val)

            self.val_losses_list.append(val_loss)
            self.val_faires_list.append(val_fair)
            self.val_accus_list.append(val_accu)

            if iteration % self.print_every == 0:
                print("{}/{}. tr_loss: {}; tr_accu: {}; tr_fair: {}; val_loss: {}; val_accu: {}; val_fair: {} -----> running time: {} seconds.".format(iteration, num_iterations, np.round(train_loss, 4),
                                                                                                                      np.round(train_accu, 4),
                                                                                                                      np.round(train_fair, 4), np.round(val_loss, 4), np.round(val_accu, 4),
                                                                                                                      np.round(val_fair, 4), np.round((time.time() - start_time), 4)))
            if iteration % self.plot_every == 0:
                self.meta_losses.append(train_loss / self.plot_every)
                train_loss = 0
                self.meta_faires.append(train_fair / self.plot_every)
                train_fair = 0
                self.meta_accus.append(train_accu / self.plot_every)
                train_accu = 0

        return [self.weights, self.lamb]
