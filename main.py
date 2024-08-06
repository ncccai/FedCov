#!/usr/bin/env python
import argparse
import numpy as np
from FLAlgorithms.servers.serverFedOur import FedOur
from utils.model_utils import create_model
import torch
import random

def create_server_n_user(args, i):
    model = create_model(args.dataset)
    server = FedOur(args, model, i)

    return server


def run_job(args, i):
    torch.manual_seed(i)
    print("\n\n         [ Start training iteration {} ]           \n\n".format(i))
    # Generate model
    server = create_server_n_user(args, i)
    if args.train:
        server.train(args, i)
        # server.test(i)

def main(args):
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    args.dataset = 'cmapss-biid-u5c13-FD004'

    args.num_users = int((args.dataset.split('-')[2])[1])
    for i in range(args.times):
        run_job(args, i)

    print("Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cmapss-biid-u5c126-FD004")
    parser.add_argument("--train", type=int, default=1, choices=[0, 1])
    parser.add_argument("--algorithm", type=str, default="FedOur")  # [FedOur, FedGen, FedAvg, FedProx, FedDistill]
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gen_batch_size", type=int, default=32, help='number of samples from generator')
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--personal_learning_rate", type=float, default=0.001, help="Personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--ensemble_lr", type=float, default=1e-4, help="Ensemble learning rate.")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")
    parser.add_argument("--num_glob_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--num_users", type=int, default=5, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=5, help="running times")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")

    parser.add_argument("--max_rul", type=int, default=125, help="max RUL")
    parser.add_argument("--seed", type=int, default=3000, help="randon seed")
    parser.add_argument("--uncertainty", type=int, default=0, help="0:no uncertainty, 1:add uncertainty")
    parser.add_argument('--risk', type=float, default=0.01, help='lambda (in uncertainty module)')

    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Ensemble learing rate       : {}".format(args.ensemble_lr))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_glob_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    # print("Local Model       : {}".format(args.model))
    print("Device            : {}".format(args.device))
    print("=" * 80)
    main(args)
