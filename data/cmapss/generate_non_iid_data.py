import argparse
import os
import torch
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import MinMaxScaler

COLUMNS_NAMES = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                    's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                    's15', 's16', 's17', 's18', 's19', 's20', 's21']


def gen_sequence(id_df, seq_length, seq_cols):
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values.astype(np.float32)
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 (engine 1) have 192 rows and sequence_length is equal to 15
    # so zip iterate over two following list of numbers (0,177),(14,191)
    # 0 14 -> from row 0 to row 14
    # 1 15 -> from row 1 to row 15
    # 2 16 -> from row 2 to row 16
    # ...
    # 177 191 -> from row 177 to 191
    for start, stop in zip(range(0, num_elements - seq_length + 1), range(seq_length, num_elements + 1)):
        yield data_matrix[start:stop, :]


def gen_labels(id_df, seq_length, label):
    # For example:
    # [[1]
    # [4]
    # [1]
    # [5]
    # [9]
    # ...
    # [200]]
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    label_matrix = []
    for i in range(num_elements-(seq_length-1)):
        label_matrix.append(data_matrix[i+(seq_length-1), :])

    return label_matrix

def gen_test_labels(id_df, seq_length, label):
    # For example:
    # [[1]]
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]

    # For the test labels, only 1 RUL is required per engine which is the last columns on each engine
    return data_matrix[-1,:]

def process(args, train_df, test_df, test_truth):
    # process train data
    train_rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    train_rul.columns = ['id', 'max']
    train_df = train_df.merge(train_rul, on=['id'], how='left')
    train_y = pd.DataFrame(data=[train_df['max'] - train_df['cycle']]).T

    train_df.drop('max', axis=1, inplace=True)
    train_df.drop(['s1', 's5', 's6', 's10', 's16', 's18', 's19'], axis=1, inplace=True)

    if args.setting == 'setting2':
        train_df[args.setting] = train_df[args.setting].abs().round(2)
    else:
        train_df[args.setting] = train_df[args.setting].abs().round(1)
    train_y = train_y.apply(lambda x: [y if y <= args.max_rul else args.max_rul for y in x])
    train_engine_num = train_df['id'].nunique()

    # process test data
    test_rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    test_rul.columns = ['id', 'max']

    test_truth.columns = ['more']
    test_truth['id'] = test_truth.index + 1
    test_truth['max'] = test_rul['max'] + test_truth['more']
    test_truth.drop('more', axis=1, inplace=True)

    test_df = test_df.merge(test_truth, on=['id'], how='left')
    test_y = pd.DataFrame(data=[test_df['max'] - test_df['cycle']]).T

    test_df.drop('max', axis=1, inplace=True)
    test_df.drop(['s1', 's5', 's6', 's10', 's16', 's18', 's19'], axis=1, inplace=True)

    if args.setting == 'setting2':
        test_df[args.setting] = test_df[args.setting].abs().round(2)
    else:
        test_df[args.setting] = test_df[args.setting].abs().round(1)

    test_y = test_y.apply(lambda x: [y if y <= args.max_rul else args.max_rul for y in x])
    test_engine_num = test_df['id'].nunique()

    train_data = train_df.iloc[:, 2:]
    test_data = test_df.iloc[:, 2:]

    train_normalized = pd.DataFrame(columns=train_data.columns[3:])
    test_normalized = pd.DataFrame(columns=test_data.columns[3:])

    scaler = MinMaxScaler()

    grouped_train = train_data.groupby(args.setting)
    grouped_test = test_data.groupby(args.setting)
    for train_idx, train in grouped_train:
        scaled_train = scaler.fit_transform(train.iloc[:, 3:])
        scaled_train_combine = pd.DataFrame(
            data=scaled_train,
            index=train.index,
            columns=train_data.columns[3:])
        train_normalized = pd.concat([train_normalized, scaled_train_combine])

        for test_idx, test in grouped_test:
            if train_idx == test_idx:
                scaled_test = scaler.transform(test.iloc[:, 3:])
                scaled_test_combine = pd.DataFrame(
                    data=scaled_test,
                    index=test.index,
                    columns=test_data.columns[3:])
                test_normalized = pd.concat([test_normalized, scaled_test_combine])

    train_normalized = train_normalized.sort_index()
    test_normalized = test_normalized.sort_index()

    # generate final train data:
    # generate 30 x 14 windows to obtain train_x
    seq_gen = []
    start_index = 0
    for i in range(train_engine_num):
        end_index = start_index + train_rul.loc[i, 'max']
        if end_index - start_index < args.seq_len - 1:
            print('train data less than seq_len!')
        val = list(gen_sequence(train_normalized.iloc[start_index:end_index, :], args.seq_len, train_normalized.columns))
        seq_gen.append(val)
        start_index = end_index
    train_x = list(seq_gen)

    # generate train labels
    seq_gen = []
    start_index = 0
    for i in range(train_engine_num):
        end_index = start_index + train_rul.loc[i, 'max']
        val = list(gen_labels(train_y.iloc[start_index:end_index, :], args.seq_len, train_y.columns))
        seq_gen.append(val)
        start_index = end_index
    train_y = list(seq_gen)

    # generate final test data:
    seq_gen = []
    start_index = 0
    for i in range(test_engine_num):
        end_index = start_index + test_rul.loc[i, 'max']
        # for test matrix, only 1 of n X 15 needed per engine, so the array input start from end index - sequence length
        if end_index - start_index < args.seq_len:
            print('Sensor::test data ({:}) less than seq_len ({:})!'
                  .format(end_index - start_index, args.seq_len))

            print('Sensor::Use first data to pad!')
            num_pad = args.seq_len - (end_index - start_index)
            new_sg = test_normalized.iloc[start_index:end_index, :]
            for idx in range(num_pad):
                new_sg = pd.concat([new_sg.head(1), new_sg], axis=0)

            val = list(gen_sequence(new_sg, args.seq_len, test_normalized.columns))
        else:
            val = list(gen_sequence(test_normalized.iloc[end_index - args.seq_len:end_index, :], args.seq_len,
                                         test_normalized.columns))
        seq_gen.append(val)
        start_index = end_index
    test_x = list(seq_gen)

    # generate test labels
    seq_gen = []
    start_index = 0
    for i in range(test_engine_num):
        end_index = start_index + test_rul.loc[i, 'max']
        val = list(
            [gen_test_labels(test_y.iloc[end_index - args.seq_len:end_index, :], args.seq_len, test_y.columns)])
        seq_gen.append(val)
        start_index = end_index
    test_y = list(seq_gen)

    return train_x, train_y, test_x, test_y

def get_data(args):
    train_data_pt = os.path.join(args.data_root, 'train_' + args.data_set + '.txt')
    assert os.path.exists(train_data_pt), 'data path does not exist: {:}'.format(train_data_pt)

    test_data_pt = os.path.join(args.data_root, 'test_' + args.data_set + '.txt')
    assert os.path.exists(test_data_pt), 'data path does not exist: {:}'.format(test_data_pt)

    test_truth_pt = os.path.join(args.data_root, 'RUL_' + args.data_set + '.txt')
    assert os.path.exists(test_truth_pt), 'data path does not exist: {:}'.format(test_truth_pt)

    train_data_df = pd.read_csv(train_data_pt, sep=" ", header=None)
    train_data_df.drop(train_data_df.columns[[26, 27]], axis=1, inplace=True)
    train_data_df.columns = COLUMNS_NAMES
    train_data_df = train_data_df.sort_values(['id', 'cycle'])

    test_data_df = pd.read_csv(test_data_pt, sep=" ", header=None)
    test_data_df.drop(test_data_df.columns[[26, 27]], axis=1, inplace=True)
    test_data_df.columns = COLUMNS_NAMES
    test_data_df = test_data_df.sort_values(['id', 'cycle'])

    test_truth = pd.read_csv(test_truth_pt, sep=" ", header=None)
    test_truth.drop(test_truth.columns[[1]], axis=1, inplace=True)

    return train_data_df, test_data_df, test_truth


def convert_rul(rul_set, sample_dist):
    """
    rul_set: rul value
    sample_dist: (2/5/10)
    """
    for i in range(len(rul_set)):
        for j in range(len(rul_set[i])):
            rul_dst = rul_set[i][j] // sample_dist
            if sample_dist == 5 and rul_set[i][j] == 125:
                rul_dst = (rul_set[i][j] - 1) // sample_dist
            rul_set[i][j] = rul_dst

    return rul_set


def CMPData(args):
    dataset_train = {'users': [], 'user_data': {}, 'num_samples': []}
    dataset_test = {'users': [], 'user_data': {}, 'num_samples': []}
    user_id = 0
    for data_set in args.dataset_list:
        args.data_set = data_set
        train_data_df, test_data_df, test_truth = get_data(args)
        train_x, train_y, test_x, test_y = process(args, train_data_df, test_data_df, test_truth)
        train_y_cls = copy.deepcopy(train_y)
        test_y_cls = copy.deepcopy(test_y)
        train_y_cls = convert_rul(train_y_cls, args.sample_dist)
        test_y_cls = convert_rul(test_y_cls, args.sample_dist)

        # train data
        grop_x_tmp = []
        grop_y_tmp = []
        grop_y_cls_tmp = []
        for i in range(len(train_x)):
            grop_x_tmp.extend(train_x[i])
            grop_y_tmp.extend(train_y[i])
            grop_y_cls_tmp.extend(train_y_cls[i])

        uname = 'f_{0:03d}'.format(user_id + 1)
        dataset_train['users'].append(uname)
        dataset_train['user_data'][uname] = {
            'x': torch.tensor(grop_x_tmp, dtype=torch.float32),
            'y': torch.tensor(grop_y_tmp, dtype=torch.int64),
            'y_cls': torch.tensor(grop_y_cls_tmp, dtype=torch.int64)}
        dataset_train['num_samples'].append(len(grop_y_tmp))

        # test data
        grop_x_tmp = []
        grop_y_tmp = []
        grop_y_cls_tmp = []
        for i in range(len(test_x)):
            grop_x_tmp.extend(test_x[i])
            grop_y_tmp.extend(test_y[i])
            grop_y_cls_tmp.extend(test_y_cls[i])
        uname = 'f_{0:03d}'.format(user_id + 1)
        dataset_test['users'].append(uname)
        dataset_test['user_data'][uname] = {
            'x': torch.tensor(grop_x_tmp, dtype=torch.float32),
            'y': torch.tensor(grop_y_tmp, dtype=torch.int64),
            'y_cls': torch.tensor(grop_y_cls_tmp, dtype=torch.int64)}
        dataset_test['num_samples'].append(len(grop_y_tmp))

        user_id = user_id + 1

    # Setup directory for train/test data
    path_prefix = f'cmapss-noniid-u{args.n_user}c{args.n_class}-{args.data_set}'
    data_path_train = f'./{path_prefix}/train'
    if not os.path.exists(data_path_train):
        os.makedirs(data_path_train)
    data_path_train = os.path.join(data_path_train, "train." + args.format)

    data_path_test = f'./{path_prefix}/test'
    if not os.path.exists(data_path_test):
        os.makedirs(data_path_test)
    data_path_test = os.path.join(data_path_test, "test." + args.format)

    if args.format == "pt":
        with open(data_path_train, 'wb') as outfile_train:
            print(f"Dumping train data => {data_path_train}")
            torch.save(dataset_train, outfile_train)
        with open(data_path_test, 'wb') as outfile_test:
            print(f"Dumping test data => {data_path_test}")
            torch.save(dataset_test, outfile_test)

    print("Successful generate niid data!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", "-f", type=str, default="pt", help="Format of saving: pt (torch.save), json",
                        choices=["pt", "json"])
    parser.add_argument("--sampling_ratio", type=float, default=1, help="Ratio for sampling training samples.")
    parser.add_argument("--unknown_test", type=int, default=0, help="Whether allow test label unseen for each user.")
    parser.add_argument("--alpha", type=float, default=0.1, help="alpha in Dirichelt distribution (smaller means larger heterogeneity)")
    parser.add_argument("--data_root", type=str, default='./', help="dataset root dir")
    parser.add_argument("--dataset_list", type=list, default=['FD001', 'FD002', 'FD003', 'FD004'], help="dataset name")
    parser.add_argument("--max_rul", type=int, default=125, help="max RUL")
    parser.add_argument("--seq_len", type=int, default=30, help="sequence length of each sample")
    parser.add_argument("--n_user", type=int, default=4, help="number of local clients, should be muitiple of 10.")
    parser.add_argument("--setting", type=str, default='setting1', help="chose setting to normalize data, choice:['setting1', 'setting2', 'setting3']")
    parser.add_argument("--n_class", type=int, default=10, help="number of classification labels")
    parser.add_argument("--sample_dist", type=int, default=1, help="sample distance ( 1/2/5/10 )")

    args = parser.parse_args()

    # different sample distance has different class number
    if args.sample_dist == 2:
        args.n_class = 63
    elif args.sample_dist == 5:
        args.n_class = 25
    elif args.sample_dist == 10:
        args.n_class = 13
    elif args.sample_dist == 1:
        args.n_class = 126
    else:
        print("Wrong sample distance!")

    print()
    print("Number of users: {}".format(args.n_user))
    print("Number of classes: {}".format(args.n_class))
    print("Ratio for Sampling Training Data: {}".format(args.sampling_ratio))

    CMPData(args)

if __name__ == "__main__":
    main()
