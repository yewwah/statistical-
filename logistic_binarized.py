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

# Preparing training data with binarization
load = sio.loadmat('spamData.mat')
train = load['Xtrain']
df = pd.DataFrame(train)
df_binarized = df.applymap(lambda x: 1 if x > 0 else 0)
train = df_binarized.values

# Adding term to include bias
train = np.hstack((train, np.ones((train.shape[0], 1))))

# Initializing the weights
wt_b = np.zeros((train.shape[1]))

# Prepare Test Data
test = load['Xtest']
df = pd.DataFrame(test)
df_binarized_test = df.applymap(lambda x: 1 if x > 0 else 0)
test = df_binarized_test.values
test = np.hstack((test, np.ones((test.shape[0], 1))))

# Load ground truths
grd_truth_train = load['ytrain'].flatten()
grd_truth_test = load['ytest'].flatten()


def sigmoid(weights, train):
    x = weights.dot(train.T)
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

def compute_accuracy(grd_truth, predictions):
    # Assign class 0 or class 1
    pred = [0 if x < 0.5 else 1 for x in predictions]
    pred = zip(grd_truth, pred)
    acc = [1 if int(x[0]) == x[1] else 0 for x in pred]
    return sum(acc)/float(len(grd_truth))


weights_old = np.ones(wt_b.shape)
iter = 0
while not check_convergence(weights_old, wt_b, 1e-5, iter):
    weights_old = wt_b
    pred = sigmoid(wt_b, train)
    first_der = diff_first(train, pred, grd_truth_train, 1.0, wt_b)
    hessian_matrix = hessian(train, pred, 1.0)
    wt_b = newton_update(wt_b, first_der, hessian_matrix)
    iter += 1

# Prepare regularization
regularizers = []
lam = 1
while lam < 100:
    if lam < 10:
        regularizers.append(lam)
        lam += 1
    else:
        regularizers.append(lam+5)
        lam += 5

train_errors = []
test_errors = []
for i in regularizers:
    print i
    weights_old = np.ones(wt_b.shape)
    wt_b = np.zeros((train.shape[1]))
    iter = 0
    while not check_convergence(weights_old, wt_b, 1e-5, iter):
        weights_old = wt_b
        pred = sigmoid(wt_b, train)
        first_der = diff_first(train, pred, grd_truth_train, i, wt_b)
        hessian_matrix = hessian(train, pred, i)
        wt_b = newton_update(wt_b, first_der, hessian_matrix)
        iter += 1

    detection = sigmoid(wt_b, train)
    train_errors.append(1.0 - compute_accuracy(grd_truth_train, detection))

    detection = sigmoid(wt_b, test)
    test_errors.append(1.0 - compute_accuracy(grd_truth_test, detection))

print train_errors
x = regularizers
y_test = test_errors
y_train = train_errors
fig, ax = plt.subplots()
ax.plot(x, y_train, 'g', label='Train Error')
ax.plot(x, y_test, 'r', label='Test Error')
leg = ax.legend()

plt.title = 'Lambda against Error'
plt.savefig('logistic_binarized.png')
plt.show()
