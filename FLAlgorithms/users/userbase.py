import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import copy
from utils.model_utils import get_dataset_name
from utils.model_config import RUNCONFIGS
from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer
from utils import metric

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, args, id, model, train_data, test_data, use_adam=False):
        self.max_rul = args.max_rul
        self.metrics = metric.MetricList(metric.RMSE(max_rul=args.max_rul), metric.RULscore(max_rul=args.max_rul))
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.lamda = args.lamda
        self.local_epochs = args.local_epochs
        self.algorithm = args.algorithm
        self.K = args.K
        self.dataset = args.dataset
        self.trainloader = DataLoader(train_data, self.batch_size, shuffle=True, drop_last=True)
        self.testloader =  DataLoader(test_data, self.batch_size, drop_last=False)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        dataset_name = get_dataset_name(self.dataset)
        self.unique_labels = int((args.dataset.split('-')[2]).split('c')[1])
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']
        self.local_model = copy.deepcopy(list(self.model.parameters()))

        self.init_loss_fn()
        if use_adam:
            self.optimizer=torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.learning_rate, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=1e-2, amsgrad=False)
        else:
            self.optimizer = pFedIBOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        self.label_counts = {}

    def init_loss_fn(self):
        self.loss_cls = nn.NLLLoss()
        self.loss = nn.MSELoss()
        self.dist_loss = nn.MSELoss()
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def set_parameters(self, model, beta=1):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            if beta == 1:
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()
                local_param.data = beta * new_param.data.clone() + (1-beta) * local_param.data.clone()

    def set_shared_parameters(self, model, mode='decode'):
        # only copy shared parameters to local
        for old_param, new_param in zip(
                self.model.get_parameters_by_keyword(mode),
                model.get_parameters_by_keyword(mode)
        ):
            old_param.data = new_param.data.clone()

    def clone_model_paramenter(self, param, clone_param):
        with torch.no_grad():
            for param, clone_param in zip(param, clone_param):
                clone_param.data = param.data.clone()
        return clone_param
    
    def update_parameters(self, new_params, keyword='all'):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test(self):
        self.metrics.reset()
        self.model.eval()
        losses = 0
        for x, y, y_cls in self.testloaderfull:
            output_reg, output_cls, _, _, _ = self.model(x)
            y = y / self.max_rul
            losses += self.loss(output_reg, y)
            self.metrics.update([output.data.cpu() for output in [output_reg]], y.cpu(),
                                [loss.data.cpu() for loss in [losses]])

        metrics = self.metrics.get_name_value()
        RMSE = metrics[0][0][1]
        SCORE = metrics[1][0][1]
        return RMSE, SCORE, losses, y.shape[0]

    def get_next_train_batch(self, count_labels=True):
        try:
            (X, y, y_cls) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y, y_cls) = next(self.iter_trainloader)
        result = {'X': X, 'y': y, 'y_cls': y_cls}
        if count_labels:
            unique_y, counts=torch.unique(y_cls, return_counts=True)
            unique_y = unique_y.detach().numpy()
            counts = counts.detach().numpy()
            result['labels'] = unique_y
            result['counts'] = counts
        return result

    def get_next_test_batch(self):
        try:
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X, y)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
