# Loading libraries
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

"""
This portion of the code sets the global parameters of the code and is not expected to change
"""
std_scaler = preprocessing.StandardScaler()

# Preparing training data with Z-Norm
load = sio.loadmat('D:\Yew Wah\Downloads\spamData.mat')
train = load['Xtrain']
scaler = std_scaler.fit(train)
train = scaler.transform(train)

# Adding term to include bias
train = np.hstack((train, np.ones((train.shape[0], 1))))

# Initializing the weights
wt_b = np.zeros((train.shape[1]))

# Prepare Test Data
test = load['Xtest']
test = scaler.transform(test)

# Load ground truths
grd_truth_train = load['ytrain'].flatten()


def sigmoid(weights, train):
    x = weights.dot(train.T)
    # import pdb;pdb.set_trace()
    return 1 / (1 + np.nan_to_num(np.exp(-x)))


def diff_first(X, prob, labels, reg, weights):
    # flatten labels first
    first_der = X.T.dot(prob - labels)
    w_new = np.hstack((weights[:-1], np.array(0)))
    return first_der + reg * w_new


def hessian(X, prob, reg):
    # Compute S matrix
    S = prob * (1 - prob)
    S_diag = np.diag(S)
    reg_mat = np.eye(X.shape[1]) * reg
    reg_mat[:, -1] = 0
    return X.T.dot(S_diag).dot(X) + reg_mat


def newton_update(weights, first_derivative, hessian_matrix):
    inv = np.linalg.inv(hessian_matrix)
    updates = inv.dot(-first_derivative)
    return weights + updates


max_iters = 500


def check_convergence(beta_old, beta_new, tol, iters):
    coef_change = np.abs(beta_old - beta_new)
    return not (np.any(coef_change > tol) and iters < max_iters)


weights_old = np.ones(wt_b.shape)
iter = 0
while not check_convergence(weights_old, wt_b, 1e-5, iter):
    weights_old = wt_b
    pred = sigmoid(wt_b, train)
    first_der = diff_first(train, pred, grd_truth_train, 1.0, wt_b)
    hessian_matrix = hessian(train, pred, 1.0)
    wt_b = newton_update(wt_b, first_der, hessian_matrix)
    iter += 1
