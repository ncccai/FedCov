import json
import os
import torch
import numpy as np
from FLAlgorithms.trainmodel.models import Net
from torch.utils.data import DataLoader
from FLAlgorithms.trainmodel.generator import Generator
METRICS = ['glob_rmse', 'glob_score', 'glob_loss', 'per_loss', 'user_train_time', 'server_agg_time']

def save_print_results(args, output_list, i):
    save_path = os.path.join('results', args.dataset)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_name = args.algorithm + '-' + str(i) + '.txt'
    if args.uncertainty == 1:
        file_name = args.algorithm + '-uncertainty-' + str(i) + '.txt'
    with open(os.path.join(save_path, file_name), "w") as f:
        # 遍历输出列表，将每个元素写入文件
        for output in output_list:
            f.write(str(output) + "\n")

def get_data_dir(dataset):
    if 'cmapss' in dataset:
        path_prefix = os.path.join('data', 'cmapss', dataset)
        train_data_dir = os.path.join(path_prefix, 'train')
        test_data_dir = os.path.join(path_prefix, 'test')
    else:
        raise ValueError("Dataset not recognized.")
    return train_data_dir, test_data_dir


def read_data(dataset):
    train_data_dir, test_data_dir = get_data_dir(dataset)
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') or f.endswith(".pt")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        if file_path.endswith("json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        elif file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))

        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))
    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json') or f.endswith(".pt")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        if file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        elif file_path.endswith(".json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))
        test_data.update(cdata['user_data'])
    return clients, groups, train_data, test_data

def aggregate_data_(clients, dataset, batch_size):
    combined = []
    unique_labels = []
    for i in range(len(dataset)):
        id = clients[i]
        user_data = dataset[id]
        X, y, y_cls = convert_data(user_data['x'], user_data['y'], user_data['y_cls'])
        combined += [(x, y, y_cls) for x, y, y_cls in zip(X, y, y_cls)]
        unique_y = torch.unique(y_cls)
        unique_y = unique_y.detach().numpy()
        unique_labels += list(unique_y)

    data_loader=DataLoader(combined, batch_size, shuffle=True)
    iter_loader=iter(data_loader)
    return data_loader, iter_loader, unique_labels


def aggregate_user_test_data(data, batch_size):
    clients, loaded_data=data[0], data[3]
    data_loader, _, unique_labels = aggregate_data_(clients, loaded_data, batch_size)
    return data_loader, np.unique(unique_labels)


def aggregate_user_data(data, batch_size):
    clients, loaded_data = data[0], data[2]
    data_loader, data_iter, unique_labels = aggregate_data_(clients, loaded_data, batch_size)
    return data_loader, data_iter, np.unique(unique_labels)


def convert_data(X, y, y_cls):
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X).type(torch.float32)
        y = torch.Tensor(y).type(torch.int64)
        y_cls = torch.Tensor(y_cls).type(torch.int64)
    return X, y, y_cls


def read_user_data(index, data, dataset='', count_labels=False):
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_train, y_train, y_train_cls = convert_data(train_data['x'], train_data['y'], train_data['y_cls'])
    train_data = [(x, y, y_cls) for x, y, y_cls in zip(X_train, y_train, y_train_cls)]
    X_test, y_test, y_test_cls = convert_data(test_data['x'], test_data['y'], test_data['y_cls'])
    test_data = [(x, y, y_cls) for x, y, y_cls in zip(X_test, y_test, y_test_cls)]
    if count_labels:
        label_info = {}
        unique_y, counts=torch.unique(y_train_cls, return_counts=True)
        unique_y=unique_y.detach().numpy()
        counts=counts.detach().numpy()
        label_info['labels']=unique_y
        label_info['counts']=counts
        return id, train_data, test_data, label_info
    return id, train_data, test_data


def get_dataset_name(dataset):
    dataset = dataset.lower()
    if 'cmapss' in dataset:
        passed_dataset='cmapss'
    else:
        raise ValueError('Unsupported dataset {}'.format(dataset))
    return passed_dataset


def create_generative_model(dataset, algorithm='', embedding=False):
    passed_dataset = get_dataset_name(dataset)
    assert any([alg in algorithm for alg in ['FedGen', 'FedOur']])
    return Generator(passed_dataset, dataset, embedding=embedding)


def create_model(dataset):
    passed_dataset = get_dataset_name(dataset)
    model = Net(passed_dataset, dataset)
    return model


# uncertainty
def criterion_nig(u, la, alpha, beta, y, risk):
    om = 2 * beta * (1 + la)
    loss = sum(
        0.5 * torch.log(np.pi / la) - alpha * torch.log(om) + (alpha + 0.5) * torch.log(la * (u - y) ** 2 + om) + torch.lgamma(alpha) - torch.lgamma(alpha+0.5)) / len(u)
    lossr = risk * sum(torch.abs(u - y) * (2 * la + alpha)) / len(u)
    loss = loss + lossr
    return torch.mean(loss)
