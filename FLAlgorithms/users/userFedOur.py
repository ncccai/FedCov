import torch
import torch.nn.functional as F
import numpy as np
from FLAlgorithms.users.userbase import User
from utils.model_utils import criterion_nig

class UserFedOur(User):
    def __init__(self,
                 args, id, model, generative_model,
                 train_data, test_data,
                 available_labels, label_info,
                 use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        self.gen_batch_size = args.gen_batch_size
        self.generative_model = generative_model
        self.available_labels = available_labels
        self.label_info=label_info
        self.max_rul = args.max_rul
        self.risk = args.risk       # uncertainty
        self.uncertainty = args.uncertainty


    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:1 for label in range(self.unique_labels)}

    def train(self, glob_iter, early_stop=100, regularization=True, verbose=False):
        self.clean_up_counts()
        self.model.train()
        self.generative_model.eval()
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS = 0, 0, 0
        for epoch in range(self.local_epochs):
            self.model.train()
            for i in range(self.K):
                self.optimizer.zero_grad()
                samples =self.get_next_train_batch(count_labels=True)
                X, y, y_cls = samples['X'], samples['y'], samples['y_cls']
                self.update_label_counts(samples['labels'], samples['counts'])
                output_reg, output_cls, v, alpha, beta = self.model(X)

                y_normalize = (y / self.max_rul)
                predictive_loss_reg = self.loss(output_reg, y_normalize)
                pre_cls = F.log_softmax(output_cls, dim=1)
                if len(y_cls.shape) > 1:
                    y_cls = torch.squeeze(y_cls, dim=1)
                predictive_loss_cls = self.loss_cls(pre_cls, y_cls)

                # add uncertainty loss
                if self.uncertainty:
                    raw_loss = criterion_nig(output_reg, v, alpha, beta, y_normalize, self.risk)

                #### sample y and generate z
                if regularization and epoch < early_stop:
                    generative_alpha=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                    generative_beta=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_beta)
                    gen_output=self.generative_model(y_cls)['output']
                    output_reg_gen, output_cls_gen = self.model(gen_output, latent=True)
                    target_p = F.softmax(output_cls_gen, dim=1).clone().detach()
                    user_latent_loss = generative_beta * self.ensemble_loss(pre_cls, target_p)

                    sampled_y = np.random.choice(self.available_labels, self.gen_batch_size)
                    sampled_y = torch.tensor(sampled_y)
                    gen_result = self.generative_model(sampled_y)
                    gen_output = gen_result['output']
                    output_reg_gen_cls, output_cls_gen_cls = self.model(gen_output, latent=True)
                    user_output_logp = F.log_softmax(output_cls_gen_cls, dim=1)
                    teacher_loss_cls = generative_alpha * torch.mean(
                        self.generative_model.crossentropy_loss(user_output_logp, sampled_y)
                    )
                    teacher_loss_reg = generative_alpha * torch.mean(
                        self.generative_model.reg_loss(output_reg_gen_cls, y_normalize)
                    )
                    teacher_loss = teacher_loss_cls + teacher_loss_reg

                    # this is to further balance oversampled down-sampled synthetic data
                    gen_ratio = self.gen_batch_size / self.batch_size
                    loss = predictive_loss_reg + predictive_loss_cls + gen_ratio * teacher_loss + user_latent_loss
                    if self.uncertainty:
                         loss = loss + raw_loss
                    TEACHER_LOSS += teacher_loss
                    LATENT_LOSS += user_latent_loss
                else:
                    loss = predictive_loss_reg + predictive_loss_cls
                    if self.uncertainty:
                         loss = loss + raw_loss
                loss.backward()
                self.optimizer.step()
        # local-model <=== self.model
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        self.lr_scheduler.step(glob_iter)
        if regularization and verbose:
            TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.local_epochs * self.K)
            LATENT_LOSS = LATENT_LOSS.detach().numpy() / (self.local_epochs * self.K)
            info = '\nUser Teacher Loss={:.4f}'.format(TEACHER_LOSS)
            info += ', Latent Loss={:.4f}'.format(LATENT_LOSS)
            if self.uncertainty:
                info += ', Uncertainty Loss={:.4f}'.format(raw_loss)
            print(info)

    def adjust_weights(self, samples):
        labels, counts = samples['labels'], samples['counts']
        np_y = samples['y'].detach().numpy()
        n_labels = samples['y'].shape[0]
        weights = np.array([n_labels / count for count in counts])
        weights = len(self.available_labels) * weights / np.sum(weights)
        label_weights = np.ones(self.unique_labels)
        label_weights[labels] = weights
        sample_weights = label_weights[np_y]
        return sample_weights


