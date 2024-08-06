import os
import torch
import copy
import numpy as np
import torch.nn as nn
from utils.model_utils import get_dataset_name, METRICS
from utils.model_config import RUNCONFIGS


class Server:
    def __init__(self, args, model, seed):
        # Set up the main attributes
        self.dataset = args.dataset
        self.num_glob_iters = args.num_glob_iters
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.total_train_samples = 0
        self.K = args.K
        self.model = copy.deepcopy(model)
        self.model_name = model
        self.users = []
        self.selected_users = []
        self.num_users = args.num_users
        self.beta = args.beta
        self.lamda = args.lamda
        self.algorithm = args.algorithm
        self.mode = 'partial' if 'partial' in self.algorithm.lower() else 'all'
        self.seed = seed
        self.deviations = {}
        self.metrics = {key: [] for key in METRICS}
        self.timestamp = None
        self.save_path = args.result_path
        self.best_rmse = 10000.0
        self.best_score = 100000.0
        os.system("mkdir {}".format(self.save_path))

    def init_ensemble_configs(self, dataset_all_name='cmapss-biid-u5c126-FD004'):
        #### used for ensemble learning ####
        dataset_name = get_dataset_name(self.dataset)
        self.ensemble_lr = RUNCONFIGS[dataset_name].get('ensemble_lr', 1e-4)
        self.ensemble_batch_size = RUNCONFIGS[dataset_name].get('ensemble_batch_size', 128)
        self.ensemble_epochs = RUNCONFIGS[dataset_name]['ensemble_epochs']
        self.num_pretrain_iters = RUNCONFIGS[dataset_name]['num_pretrain_iters']
        self.temperature = RUNCONFIGS[dataset_name].get('temperature', 1)
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.unique_labels = int((dataset_all_name.split('-')[2]).split('c')[1])
        self.ensemble_alpha = RUNCONFIGS[dataset_name].get('ensemble_alpha', 1)
        self.ensemble_beta = RUNCONFIGS[dataset_name].get('ensemble_beta', 0)
        self.ensemble_eta = RUNCONFIGS[dataset_name].get('ensemble_eta', 1)
        self.weight_decay = RUNCONFIGS[dataset_name].get('weight_decay', 0)
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']
        self.ensemble_train_loss = []
        self.n_teacher_iters = 5
        self.n_student_iters = 1
        print("ensemble_lr: {}".format(self.ensemble_lr))
        print("ensemble_batch_size: {}".format(self.ensemble_batch_size))
        print("unique_labels: {}".format(self.unique_labels))

    def send_parameters(self, mode='all', beta=1, selected=False):
        users = self.users
        if selected:
            assert (self.selected_users is not None and len(self.selected_users) > 0)
            users = self.selected_users
        for user in users:
            if mode == 'all':  # share only subset of parameters
                user.set_parameters(self.model, beta=beta)
            else:  # share all parameters
                user.set_shared_parameters(self.model, mode=mode)

    def add_parameters(self, user, ratio, partial=False):
        if partial:
            for server_param, user_param in zip(self.model.get_shared_parameters(), user.model.get_shared_parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        else:
            for server_param, user_param in zip(self.model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self, partial=False):
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        if partial:
            for param in self.model.get_shared_parameters():
                param.data = torch.zeros_like(param.data)
        else:
            for param in self.model.parameters():
                param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train, partial=partial)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def save_clients_model(self, i, glob_iter):
        model_path = os.path.join("models", self.dataset, 'time_' + str(i + 1))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        j = 0
        for c in self.users:
            j = j + 1
            torch.save(c.model, os.path.join(model_path, "user_" + str(j) + ".pth"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def select_users(self, round, num_users, return_idx=False):
        num_users = min(num_users, len(self.users))
        if return_idx:
            user_idxs = np.random.choice(range(len(self.users)), num_users, replace=False)  # , p=pk)
            return [self.users[i] for i in user_idxs], user_idxs
        else:
            return np.random.choice(self.users, num_users, replace=False)

    def init_loss_fn(self):
        self.loss = nn.NLLLoss()
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()


    def test(self, selected=False):
        num_samples = []
        tot_rmse = []  # RMSE
        tot_score = []  # SCORE
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            RMSE, SCORE, c_loss, ns = c.test()
            tot_rmse.append(RMSE * 1.0)  # RMSE
            tot_score.append(SCORE * 1.0)  # SCORE
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_rmse, tot_score, losses


    def evaluate(self, i, glob_iter, save=True, selected=False):
        # i: train times, glob_iter:train epoch in the i-th time
        test_ids, test_samples, test_rmse, test_score, test_losses = self.test(selected=selected)
        glob_rmse = np.sum(test_rmse) * 1.0 / len(test_samples)
        glob_score = np.sum(test_score) * 1.0 / len(test_samples)
        glob_loss = np.sum([x * y.detach().numpy() for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(
            test_samples)

        if glob_score < self.best_score:
            self.best_score = glob_score
            self.best_rmse = glob_rmse
            self.save_model()
            self.save_clients_model(i, glob_iter)

        if save:
            self.metrics['glob_rmse'].append(glob_rmse)
            self.metrics['glob_score'].append(glob_score)
            self.metrics['glob_loss'].append(glob_loss)
        out_put_ = "Average Global RMSE = {:.4f}, Global SCORE = {:.4f}, Loss = {:.2f}, Best Value [RMSE: {:.4f}, SCORE: {:.4f}].".format(
            glob_rmse, glob_score, glob_loss, self.best_rmse, self.best_score)
        print(out_put_)



        return out_put_
