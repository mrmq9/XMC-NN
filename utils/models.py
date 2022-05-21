import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_labels, dim_data, dim_hidden, drop_val):
        super(Model, self).__init__()
        self.drop1 = nn.Dropout(p=drop_val)
        self.fc1 = nn.Linear(dim_data, dim_hidden)
        self.last_fc = nn.Embedding(num_labels, dim_hidden)
        nn.init.xavier_uniform_(self.last_fc.weight)

    def forward(self, data, labels, uniform=False):

        data = self.drop1(data)
        data = self.fc1(data)
        data = F.relu(data)
        weights = self.last_fc(labels)
        if uniform:
            scores = torch.bmm(weights, data.unsqueeze(-1)).squeeze(-1)
        else:
            scores = torch.mm(data, weights.T)
        
        return scores