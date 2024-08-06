from FLAlgorithms.users.userFedOur import UserFedOur
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model, save_print_results
import torch
import torch.nn.functional as F
import numpy as np
import copy
import time
MIN_SAMPLES_PER_LABEL = 1

class FedOur(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        data = read_data(args.dataset)
        # data contains: clients, groups, train_data, test_data
        clients = data[0]
        total_users = len(clients)
        self.total_test_samples = 0
        self.local = 'local' in self.algorithm.lower()
        self.use_adam = 'adam' in self.algorithm.lower()

        self.early_stop = 20  # stop using generated samples after 20 local epochs
        self.student_model = copy.deepcopy(self.model)
        self.generative_model = create_generative_model(args.dataset, args.algorithm, args.embedding)
        if not args.train:
            print('number of generator parameteres: [{}]'.format(self.generative_model.get_number_of_parameters()))
            print('number of model parameteres: [{}]'.format(self.model.get_number_of_parameters()))
        self.init_ensemble_configs(args.dataset)
        print("label embedding {}".format(self.generative_model.embedding))
        print("ensemeble learning rate: {}".format(self.ensemble_lr))
        print("ensemeble alpha = {}, beta = {}, eta = {}".format(self.ensemble_alpha, self.ensemble_beta, self.ensemble_eta))
        print("generator alpha = {}, beta = {}".format(self.generative_alpha, self.generative_beta))
        self.init_loss_fn()
        self.train_data_loader, self.train_iter, self.available_labels = aggregate_user_data(data, self.ensemble_batch_size)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.ensemble_lr, weight_decay=0.0001)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)

        #### creating users ####
        self.users = []
        for i in range(total_users):
            id, train_data, test_data, label_info = read_user_data(i, data, dataset=args.dataset, count_labels=True)
            self.total_train_samples += len(train_data)
            self.total_test_samples += len(test_data)
            user=UserFedOur(
                args, id, model, self.generative_model,
                train_data, test_data,
                self.available_labels, label_info,
                use_adam=self.use_adam)
            self.users.append(user)
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating server.")

    def train(self, args, i):
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            self.selected_users, self.user_idxs = self.select_users(glob_iter, self.num_users, return_idx=True)
            if not self.local:
                self.send_parameters(mode=self.mode)    # broadcast averaged prediction model
            self.evaluate(i, glob_iter)
            chosen_verbose_user = np.random.randint(0, len(self.users))
            self.timestamp = time.time()
            for user_id, user in zip(self.user_idxs, self.selected_users): # allow selected users to train
                verbose = user_id == chosen_verbose_user
                # perform regularization using generated samples after the first communication round
                user.train(
                    glob_iter,
                    early_stop=self.early_stop,
                    verbose=verbose and glob_iter > 0,
                    regularization=glob_iter > 0)
            curr_timestamp = time.time()
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)

            self.timestamp = time.time()
            self.train_generator(
                self.batch_size,
                epoches=self.ensemble_epochs // self.n_teacher_iters,
                verbose=True
            )
            self.aggregate_parameters()
            curr_timestamp=time.time()
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)

    def train_generator(self, batch_size, epoches=1, verbose=False):
        """
        Learn a generator that find a consensus latent representation z, given a label 'y'.
        :param batch_size:
        :param epoches:
        :param verbose: print loss information.
        :return: Do not return anything.
        """
        self.label_weights, self.qualified_labels = self.get_label_weights()
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0

        def update_generator_(n_iters, student_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS):
            self.generative_model.train()
            student_model.eval()
            for i in range(n_iters):
                self.generative_optimizer.zero_grad()
                y = np.random.choice(self.qualified_labels, batch_size)
                y_normalize = torch.from_numpy(y / len(self.available_labels)).to(torch.float32)
                y_input = torch.LongTensor(y)
                ## feed to generator
                gen_result = self.generative_model(y_input, verbose=True)
                gen_output, eps = gen_result['output'], gen_result['eps']
                ##### get losses ####
                diversity_loss = self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

                ######### get teacher loss ############
                teacher_loss=0
                teacher_logit=0
                for user_idx, user in enumerate(self.selected_users):
                    user.model.eval()
                    weight = self.label_weights[y][:, user_idx].reshape(-1, 1)
                    expand_weight = np.tile(weight, (1, self.unique_labels))
                    user_reg_result_given_gen, user_cls_result_given_gen = user.model(gen_output, latent=True)
                    user_output_logp_ = F.log_softmax(user_cls_result_given_gen, dim=1)
                    teacher_loss_cls = torch.mean( \
                        self.generative_model.crossentropy_loss(user_output_logp_, y_input) * \
                        torch.tensor(weight, dtype=torch.float32))
                    teacher_loss_reg = torch.mean( \
                        self.generative_model.reg_loss(user_reg_result_given_gen, y_normalize) * \
                        torch.tensor(weight, dtype=torch.float32))
                    teacher_loss_ = teacher_loss_cls + teacher_loss_reg

                    teacher_loss += teacher_loss_
                    teacher_logit += user_cls_result_given_gen * torch.tensor(expand_weight, dtype=torch.float32)

                ######### get student loss ############
                student_output_reg, student_output_cls = student_model(gen_output, latent=True)
                student_loss = F.kl_div(F.log_softmax(student_output_cls, dim=1), F.softmax(teacher_logit, dim=1))
                if self.ensemble_beta > 0:
                    loss = self.ensemble_alpha * teacher_loss - self.ensemble_beta * student_loss + self.ensemble_eta * diversity_loss
                else:
                    loss = self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss
                loss.backward()
                self.generative_optimizer.step()
                TEACHER_LOSS += self.ensemble_alpha * teacher_loss
                STUDENT_LOSS += self.ensemble_beta * student_loss
                DIVERSITY_LOSS += self.ensemble_eta * diversity_loss
            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS

        for i in range(epoches):
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS=update_generator_(
                self.n_teacher_iters, self.model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)

        TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        STUDENT_LOSS = STUDENT_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        info="Generator: Teacher Loss= {:.4f}, Teacher Loss= {:.4f}, Diversity Loss = {:.4f} ". \
            format(TEACHER_LOSS, DIVERSITY_LOSS, STUDENT_LOSS)
        if verbose:
            print(info)
        self.generative_lr_scheduler.step()


    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for user in self.selected_users:
                weights.append(user.label_counts[label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append( np.array(weights) / np.sum(weights) )
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels
