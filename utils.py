import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class FeatureSet(Dataset):
    def __init__(self, features, labels):
        self.data = features
        self.local_lb = labels
        self.num_class = int(np.max(self.local_lb)) + 1
        assert int(np.min(self.local_lb)) == 0
        assert self.data.shape[0] == self.local_lb.shape[0]

    def __len__(self):
        return len(self.local_lb)

    def __getitem__(self, i):
        image, loc_lb = self.data[i], self.local_lb[i]
        return image, loc_lb


class Batch_MetaSampler():
    '''
    sample a batch of few-shot learning episodes of shape (bs * n_cls * (n_shot+n_query))
    '''
    def __init__(self, label, n_batch, n_cls, n_shot, n_query, ep_per_batch=1):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_shot + n_query
        self.ep_per_batch = ep_per_batch

        label = np.array(label)
        self.catlocs = []
        for c in range(max(label) + 1):
            self.catlocs.append(np.argwhere(label == c).reshape(-1))

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                classes = np.random.choice(len(self.catlocs), self.n_cls, replace=False)
                for c in classes:
                    l = np.random.choice(self.catlocs[c], self.n_per, replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch) # bs * n_cls * n_per
            yield batch.view(-1)


def make_nk_label(n, k, ep_per_batch=1, base_way=0):
    '''
    n-way, k-shot labels
    return (ep_per_batch, n_way, k_shot)
    '''
    label = torch.arange(n).unsqueeze(1).expand(n, k) + base_way
    label = label.unsqueeze(0).expand(ep_per_batch, -1, -1)
    return label.contiguous()


def normalize_l2(x, axis=1):
    '''x.shape = (num_samples, feat_dim)'''
    x_norm = np.linalg.norm(x, axis=axis, keepdims=True)
    x = x / (x_norm + 1e-8)
    return x