import numpy as np
import torch
import scipy.io
from util import Lt
import argparse

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--m', type=int, default=10, help="m-dimension subspace")
    parser.add_argument('--nanpta1', type=float, default=1, help="hyperparameter")
    parser.add_argument('--nanpta2', type=float, default=1, help="hyperparameter")
    parser.add_argument('--nanpta3', type=float, default=1, help="hyperparameter")
    parser.add_argument('--nanpta4', type=float, default=1, help="hyperparameter")
    parser.add_argument('--lr', type=float, default=0.2, help="learning rate")
    parser.add_argument('--num_epochs', type=int, default=100, help="number of epochs")
    parser.add_argument('--tao', type=float, default=0.5, help="an fixxed hyperparameter")
    parser.add_argument('--M', type=int, default=10, help="number of classes")
    parser.add_argument('--k', type=int, default=5, help="the k nearest samples")
    parser.add_argument('--num_batches', type=int, default=10, help="number of batches")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    m = args.m
    nanpta1 = args.nanpta1
    nanpta2 = args.nanpta2
    nanpta3 = args.nanpta3
    nanpta4 = args.nanpta4

    data = scipy.io.loadmat('example.mat')
    X = data['data']
    X= X.astype(np.float32)

    min_val = np.min(X)
    max_val = np.max(X)
    X = (X - min_val) / (max_val - min_val)
    n, d = X.shape
    X = torch.tensor(X, device=DEVICE)
    W_t = torch.randn(d, m, device=DEVICE, requires_grad=True)

    optimizer = torch.optim.Adam([W_t], lr=args.lr)
    num_epochs =args.num_epochs
    tao = args.tao
    M = args.M
    k=args.k
    num_batches = args.num_batches
    batch_size = n // num_batches

    X_batches = torch.split(X, batch_size, dim=0)

    total_loss = 0
    i = 0
    for batch in range(num_batches):
        batch_loss = 0
        i = i+1
        for epoch in range(num_epochs):
            X_t = X_batches[batch]
            
            W_t_1 = W_t
            loss = Lt(X_t, W_t, W_t_1, nanpta1, nanpta2, nanpta3,nanpta4,tao,M,k)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            batch_loss = loss.item()

    norms = torch.norm(W_t, dim=1)
    sorted_indices = torch.argsort(norms, descending=True)