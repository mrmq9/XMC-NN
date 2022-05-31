# coding=utf-8
# @Author  : Mohammadreza Qaraei
# @Email   : mohammadreza.mohammadniaqaraei@aalto.fi


from torch.utils.data import Dataset
import numpy as np


class DatasetFull(Dataset):
    """

    """
    def __init__(self, data, labels, return_label=True, labels_one_hot=True):
        self.data = data
        self.labels = labels
        self.return_label = return_label
        self.labels_one_hot = labels_one_hot
    
    def _get_docs(self, indices):
        data = []
        for idx in indices:
            data.append(self.data[idx].toarray().squeeze(0))
        return np.array(data)

    def _get_labels(self, indices):
        lbl = []
        for sample_idx in indices:
            lbl_sample = self.labels[sample_idx].indices
            lbl.append(lbl_sample)
        return lbl
    
    def _get_labels_one_hot(self, indices):
        lbl = []
        for sample_idx in indices:
            lbl_sample = self.labels[sample_idx].toarray().squeeze(0).astype(float)
            lbl.append(lbl_sample)
        return np.array(lbl)
        
    def __getitem__(self, indices):
        data = self._get_docs(indices)
        if self.return_label:
            if self.labels_one_hot:
                lbl = self._get_labels_one_hot(indices)
            else:
                lbl = self._get_labels(indices)
            return data, lbl
        return data

    def __len__(self):
        return self.data.shape[0]



class DatasetInBatchSampling(DatasetFull):

    """
        Training data (samples and corresponding pos/neg labels) with in-batch
        negative sampling. Negative labels for each sample are the positive labels
        of the other samples in the minibatch.
    """

    def __init__(self, data, labels):
        super().__init__(data, labels)

    def _get_in_batch_labels(self, indices):
        all_batch_labels = []
        lbl_col, lbl_row = [], []
        
        for idx, sample_idx in enumerate(indices):
            pos_labels_sample = self.labels[sample_idx].indices
            for label in pos_labels_sample:
                if label in all_batch_labels:
                    lbl_col.append(all_batch_labels.index(label))
                else:
                    all_batch_labels.append(label)
                    lbl_col.append(len(all_batch_labels)-1)
                lbl_row.append(idx)

        batch_size, num_pos_labels = len(indices), len(all_batch_labels)
        lbl = np.zeros((batch_size, num_pos_labels))
        lbl[lbl_row, lbl_col] = 1.0
        return lbl, np.array(all_batch_labels)
        


    def __getitem__(self, indices):
    
        data = self._get_docs(indices)
        
        lbl_in_batch, all_batch_labels = self._get_in_batch_labels(indices)

        return data, lbl_in_batch, all_batch_labels
    



class DatasetUniformSampling(DatasetFull):
    
    """
        Training data (samples and corresponding pos/neg labels) with uniform 
        negative sampling.
    """

    def __init__(self, data, labels, num_neg):
        super().__init__(data, labels)
        self.num_neg = num_neg

    def _get_uniform_labels(self, indices):
        num_total_labels = self.labels.shape[1]
        batch_size = len(indices)
        all_labels = np.arange(num_total_labels)
        lbl_one_hot = np.zeros((batch_size, self.num_neg))
        lbl_indices = np.zeros((batch_size, self.num_neg), dtype=int)

        for idx, sample_idx in enumerate(indices):
            pos_labels_sample = self.labels[sample_idx].indices
            possible_negs_ind = np.where(~np.in1d(all_labels, pos_labels_sample))[0]
            neg_labels_sample = np.random.choice(all_labels[possible_negs_ind],
                                                 self.num_neg-len(pos_labels_sample), replace=False)
            lbl_indices[idx] = np.concatenate((pos_labels_sample, neg_labels_sample))
            lbl_one_hot[idx, :len(pos_labels_sample)] = 1.0

        return lbl_one_hot, lbl_indices
        


    def __getitem__(self, indices):
    
        data = self._get_docs(indices)
        
        lbl_one_hot, lbl_indices = self._get_uniform_labels(indices)

        return data, lbl_one_hot, lbl_indices
