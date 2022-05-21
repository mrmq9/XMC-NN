# coding=utf-8
# @Author  : Mohammadreza Qaraei
# @Email   : mohammadreza.mohammadniaqaraei@aalto.fi


import os
import json
import argparse
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from utils import get_cache_data_label
from utils import DatasetInBatchSampling, DatasetUniformSampling, \
                  DatasetFull
from torch.utils.data import DataLoader, SequentialSampler, \
                             BatchSampler, RandomSampler
from runner import Runner


def main(args):
    train_x, train_y = get_cache_data_label(args)
    test_x, test_y = get_cache_data_label(args, train=False)

    mlb = MultiLabelBinarizer(sparse_output=True)
    total_y = train_y + test_y
    mlb.fit(total_y)
    train_y = mlb.transform(train_y)
    test_y = mlb.transform(test_y)

    num_labels, dim_data = train_y.shape[1], train_x.shape[1]

    freq = np.array(np.sum(train_y, axis=0))[0]
    c = (np.log(train_y.shape[0]) - 1) * np.power(args.b+1, args.a)
    inv_prop = 1 + c * np.power(freq + args.b, -args.a)

    model_path = os.path.join(args.model_dir, args.dataset)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, 'model')

    test_dataset = DatasetFull(test_x, test_y, return_label=True, labels_one_hot=False)

    sampler = SequentialSampler(test_dataset)
    sampler = BatchSampler(sampler, batch_size=args.batch_test, drop_last=False)
    test_loader = DataLoader(test_dataset, sampler=sampler)

    if args.sampling_type == 'in-batch':
        train_dataset = DatasetInBatchSampling(train_x, train_y)
    elif args.sampling_type == 'uniform':
        train_dataset = DatasetUniformSampling(
            train_x, train_y, num_neg=args.num_neg)
    elif args.sampling_type == 'full':
        train_dataset = DatasetFull(train_x, train_y)
    else:
        raise NotImplementedError('Unknow negative sampling')

    sampler = RandomSampler(train_dataset)
    sampler = BatchSampler(sampler, batch_size=args.batch_train, drop_last=False)
    train_loader = DataLoader(train_dataset, sampler=sampler)

    runner = Runner(num_labels, dim_data, args.dim_hidden,
                    args.lr, args.weight_decay, args.drop_val,
                    model_path, args.sampling_type)

    runner.train(train_loader, test_loader, inv_prop,
                 args.epoch, args.log_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--dataset_dir', default='datasets', type=str,
                        help='directory to training and test data')
    parser.add_argument('--cache_dir', default='datasets', type=str,
                        help='directory to cache data and labels')
    parser.add_argument('--dataset', default='Eurlex', type=str,
                        help='name of the dataset')

    # model
    parser.add_argument('--model_dir', default='models/', type=str,
                        help='directory for saving the model')
    parser.add_argument('--batch_train', default=256, type=int,
                        help='batch size for training the model')
    parser.add_argument('--batch_test', default=1000, type=int,
                        help='batch size for evaluating the model')
    parser.add_argument('--num_neg', default=50, type=int,
                        help='number of negative labels for uniform negative sampling')
    parser.add_argument('--epoch', default=50, type=int,
                        help='number of epochs for training the model')
    parser.add_argument('--log_step', default=50, type=int,
                        help='logging step at each epoch')
    parser.add_argument('--lr', default=2e-2, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=0.0, type=float,
                        help='weight decay')
    parser.add_argument('--drop_val', default=0.2, type=float,
                        help='probability for drop out')
    parser.add_argument('--dim_hidden', default=1024, type=int,
                        help='dimension of the hidden layer')
    parser.add_argument('--sampling_type', choices=['full', 'in-batch', 'uniform'],
                        default='in-batch', type=str, help='negative sampling type')
    parser.add_argument('--a', default=0.55, type=float,
                        help='"a" hyperparameter for computing the inverse propensities')
    parser.add_argument('--b', default=1.5, type=float,
                        help='"b" hyperparameter for computing the inverse propensities')

    args = parser.parse_args()

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=4))

    main(args)
