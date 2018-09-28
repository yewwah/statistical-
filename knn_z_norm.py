# Loading libraries
import numpy as np
import scipy.io as sio
import pandas as pd

import matplotlib.pyplot as plt
import scipy
from sklearn import preprocessing
import operator

# Loading preprocessors

std_scaler = preprocessing.StandardScaler()

# Preparing training data with Z-Norm
load = sio.loadmat('spamData.mat')
train = load['Xtrain']
scaler = std_scaler.fit(train)
train = scaler.transform(train)

train_labels = load['ytrain']
train_final = np.hstack((train, train_labels))

df_norm = pd.DataFrame(train_final)

# Preparation of training data
# Remove ground truth from training data
df_train = df_norm.drop(df_norm.columns[len(df_norm.columns) - 1], axis=1)

# Prepare Test Data
test = load['Xtest']
test = scaler.transform(test)

df_test = pd.DataFrame(test)

# Load ground truths
grd_truth_train = load['ytrain'].flatten()
grd_truth_test = load['ytest'].flatten()



def calc_proximity(training_data, test_data):
    proximity = []
    for idx, test_vec in enumerate(test_data):
        test_vec_res = []
        for train_vec in training_data:
            # Check L2 distance
            test_vec_res.append(scipy.spatial.distance.euclidean(train_vec, test_vec))
        proximity.append(test_vec_res)
    # Returns a list of list
    return np.array(proximity)


def identify_k_nn(k, proxim, dim, grd_truth):
    final = []
    for idx in proxim:
        res = dict(zip(dim, idx))
        sorted_d = sorted(res.items(), key=operator.itemgetter(1))[:k]
        arr = np.array(sorted_d)
        final.append(classify_neighbours(k, arr[:, 0], grd_truth))
    return final

def classify_neighbours(k, idxes, labels):
    total = 0
    for idx in idxes:
        total += labels[int(idx)]

    if total > k / 2:
        return 1
    return 0


def compute_accuracy(grd_truth, pred):
    # Assign class 0 or class 1
    pred = zip(grd_truth, pred)
    acc = [1 if int(x[0]) == x[1] else 0 for x in pred]
    return sum(acc)/float(len(grd_truth))

lam = 1
lst_of_k = []
while lam < 100:
    if lam < 10:
        lst_of_k.append(lam)
        lam += 1
    else:
        lst_of_k.append(lam+5)
        lam += 5

train_errors = []
prox = calc_proximity(df_train.values, df_train.values)
dimension = range(prox.shape[1])
for i in lst_of_k:
    res_train = identify_k_nn(i, prox, dimension, grd_truth_train)
    train_errors.append(1.0 - compute_accuracy(grd_truth_train, res_train))

test_errors = []
proximity = calc_proximity(df_train.values, df_test.values)
dimension = range(proximity.shape[1])
for i in lst_of_k:
    res_test = identify_k_nn(i, proximity, dimension, grd_truth_train)
    test_errors.append(1.0 - compute_accuracy(grd_truth_test, res_test))


x = lst_of_k
y_test = test_errors
y_train = train_errors
fig, ax = plt.subplots()
ax.plot(x, y_train, 'g', label='Train Error')
ax.plot(x, y_test, 'r', label='Test Error')
leg = ax.legend()

plt.title = 'K against Error'
plt.savefig('knn_z_norm.png')
plt.show()
