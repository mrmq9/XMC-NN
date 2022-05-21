# coding=utf-8
# @Author  : Mohammadreza Qaraei
# @Email   : mohammadreza.mohammadniaqaraei@aalto.fi


import os
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import torch



def load_data_label(data_path):
    with open(data_path, 'r') as f:
        data = f.read().split('\n')
    data = data[:-1]

    heading = data[0]
    num_data, num_features, num_labels = list(map(int, heading.split(' ')))

    labels = []
    row, col, features_val = [], [], []
    row_idx = 0
    for sample in data[1:]:
        sample_labels = sample.split(' ', 1)[0]
        if sample_labels == '':
            sample_labels = []
            # num_data -= 1
            # continue
        else:
            sample_labels = list(map(int, sample_labels.split(',')))
        labels.append(sample_labels)

        features = sample.split(' ', 1)[1]
        features = features.split(' ')
        for f in features:
            f_key, f_val = f.split(':')
            f_key, f_val = int(f_key), float(f_val)
            row.append(row_idx)
            col.append(f_key)
            features_val.append(f_val)
        row_idx += 1
    row, col, features_val = np.array(row), np.array(col), np.array(features_val, dtype=np.float32)
    data = csr_matrix((features_val, (row, col)), shape=(num_data, num_features))

    # normalize features
    data = normalize(data, norm='l2', axis=1)
    
    return data, labels



def get_cache_data_label(args, train=True):
    cache_path = os.path.join(args.cache_dir, args.dataset, 'cached_{}_{}'.format(
                              'train' if train else 'test', args.dataset))
    if os.path.exists(cache_path):
        print('Loading cached {} data'.format('training' if train else 'test'))
        data, labels = torch.load(cache_path)
    else:
        print('Cached {} data not found. Loading data'.format('training' if train else 'test'))
        data_path = os.path.join(args.dataset_dir, args.dataset, 'train.txt' if train else 'test.txt')
        data, labels = load_data_label(data_path)
        torch.save([data, labels], cache_path)
    
    return data, labels
