import argparse
# noinspection PyUnresolvedReferences
import os
import tqdm
# noinspection PyUnresolvedReferences
import math
import pickle
# noinspection PyUnresolvedReferences
import logging
# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
import torch
# noinspection PyUnresolvedReferences
import torch.nn as nn
# noinspection PyUnresolvedReferences
import torch.nn.functional as F
# noinspection PyUnresolvedReferences
import torchvision.transforms as transforms
# noinspection PyUnresolvedReferences
from torch.autograd import Variable
from torch.utils.data import DataLoader
# noinspection PyUnresolvedReferences
from torch.utils.tensorboard import SummaryWriter 
from sklearn.linear_model import LogisticRegression

from utils import *
use_gpu = torch.cuda.is_available()


def distribution_calibration(query, base_means, base_cov, k, alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0) + alpha
    return calibrated_mean, calibrated_cov


# Re-implement ICLR2021 paper "FREE LUNCH FOR FEW-SHOT LEARNING: DISTRIBUTION CALIBRATION"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='miniImagenet')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_shot', type=int, default=5)
    parser.add_argument('--n_query', type=int, default=15)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--n_episode', type=int, default=2000)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--task_per_batch_test', type=int, default=1)
    parser.add_argument('--requries_normalize', action='store_true')
    parser.add_argument('--requries_finetune', action='store_true')
    parser.add_argument('--requries_repeat', action='store_true')
    parser.add_argument('--requries_hallu', action='store_true')
    parser.add_argument('--n_hallu', type=int, default=750)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    assert args.task_per_batch_test == 1
    # build novel_test dataloader
    novel_features_path = 'filelists/%s/novel_features.plk'%(args.dataset)
    l_tmp = -1
    novel_labels = []
    novel_features = []
    with open(novel_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            l_tmp += 1
            feature = np.array(data[key])   # (num_samples, feat_dim)
            novel_labels += [l_tmp] * feature.shape[0]
            novel_features.append(feature)
        novel_labels = np.array(novel_labels)
        novel_features = np.concatenate(novel_features, axis=0)
    noveltest_set = FeatureSet(novel_features, novel_labels) 
    noveltest_sampler = Batch_MetaSampler(noveltest_set.local_lb, args.n_episode//args.task_per_batch_test,
                                          args.n_way, args.n_shot, args.n_query, args.task_per_batch_test) 
    noveltest_loader = DataLoader(dataset=noveltest_set, batch_sampler=noveltest_sampler, 
                                  num_workers=args.n_workers, pin_memory=True) 

    # ---- data loading
    dataset     = args.dataset
    feat_dim    = feature.shape[-1]
    n_shot      = args.n_shot
    n_ways      = args.n_way
    n_queries   = args.n_query
    n_runs      = args.n_episode
    n_lsamples  = n_ways * n_shot                
    
    # ---- Base class statistics
    base_means = []
    base_cov = []
    base_features_path = 'filelists/%s/base_features.plk'%(args.dataset)
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])   # (num_samples, feat_dim)
            mean = np.mean(feature, axis=0) # (feat_dim, )
            cov = np.cov(feature.T)         # (feat_dim, feat_dim)
            base_means.append(mean)
            base_cov.append(cov)

    # ---- classification for each task
    if args.deterministic: 
        np.random.seed(args.seed)
    iter_num = len(noveltest_loader)
    noveltest_loader = iter(noveltest_loader)
    tqdm_gen = tqdm.tqdm(range(iter_num))
    acc_list = []
    for _ in tqdm_gen:
        episode_data, _ = next(noveltest_loader)                                # (n_way*(n_shot+n_query), feat_dim)
        episode_data  = episode_data.view(n_ways, n_shot+n_queries, feat_dim)   # (n_way, n_shot+n_query, feat_dim)
        support_data  = episode_data[:, :n_shot].reshape(-1, feat_dim).numpy()  # (n_way*n_shot, feat_dim)
        query_data    = episode_data[:, n_shot:].reshape(-1, feat_dim).numpy()  # (n_way*n_query, feat_dim)
        support_label = make_nk_label(n_ways, n_shot, 1, 0).view(-1).numpy()    # (n_way*n_shot)
        query_label   = make_nk_label(n_ways, n_queries, 1, 0).view(-1).numpy() # (n_way*n_query)
        if args.requries_finetune:
            num_sampled = int(args.n_hallu // n_shot)
            # ---- Tukey's transform
            beta = args.beta
            support_data = np.power(support_data[:, ] ,beta)
            query_data = np.power(query_data[:, ] ,beta)
            if args.requries_hallu:
                # ---- distribution calibration and feature sampling
                sampled_data = []
                sampled_label = []
                for i in range(n_lsamples):
                    mean, cov = distribution_calibration(support_data[i], base_means, base_cov, k=2)
                    sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
                    sampled_label.extend([support_label[i]]*num_sampled)
                sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled, -1)
                X_aug = np.concatenate([support_data, sampled_data])
                Y_aug = np.concatenate([support_label, sampled_label])
            else:
                if args.requries_repeat:
                    X_aug = np.tile(support_data, (num_sampled, 1))
                    Y_aug = np.tile(support_label, (num_sampled))
                else:
                    X_aug = support_data
                    Y_aug = support_label
            if args.requries_normalize:
                X_aug = normalize_l2(X_aug)
                query_data = normalize_l2(query_data)
            # ---- train classifier
            classifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug)
            predicts = classifier.predict(query_data)
            acc = np.mean(predicts == query_label)
        else:
            raise Exception()
        acc_list.append(acc)
        tqdm_gen.set_description('acc=%.4f%%'%(acc*100))
    print('%s %d-way %d-shot  ACC: %.4f%%'%(dataset, n_ways, n_shot, float(np.mean(acc_list)*100)))