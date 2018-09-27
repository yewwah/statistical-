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
load = sio.loadmat('D:\Yew Wah\Downloads\spamData.mat')
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
grd_truth = load['ytrain'].flatten().tolist()


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


def identify_k_nn(k, prox, dim, final):
    for i in prox:
        res = dict(zip(dim, i))
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


prox = calc_proximity(df_train.values, df_test.values)
dim = range(prox.shape[1])
final = []
fin = identify_k_nn(3, prox, dim, final)
print fin