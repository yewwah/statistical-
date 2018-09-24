import scipy.io as sio
import pandas as pd
import numpy as np
from sklearn import preprocessing
from math import e, sqrt, pi
import matplotlib.pyplot as plt


"""
This portion of the code sets the global parameters of the code and is not expected to change
"""
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
grd_truth = load['ytest'].flatten().tolist()

# Calculating priors
p_y0 = len(train_labels[train_labels == 0])/ 3056.0
p_y1 = 1 - p_y0

# Setting the Log Priors
log_p_y0 = np.log(p_y0)
log_p_y1 = np.log(p_y1)


def gaussian(x, mean, var):
    return 1/(sqrt(2*pi)*var)*e**(-0.5*(float(x-mean)/var)**2)


def compute_training_probs(df):
    means = []
    variances = []

    for feat in df.columns:
        # Calculate mean and var
        mean = np.mean(df[feat])
        var = np.var(df[feat])
        means.append(mean)
        variances.append(var)
    return means, variances


def compute_prob_for_class(df, mean, var, label_prior):
    classes = []
    for idx, row in df.iterrows():
        prob = 0
        for col_index in range(len(row)):
            prob += np.log(np.nan_to_num(gaussian(row[col_index], mean[col_index], var[col_index])) + 1e-9)
        classes.append(prob + label_prior)
    return classes


def compute_accuracy(grd_truth, y_pred_0, y_pred_1):
    # Assign class 0 or class 1
    final_results = zip(y_pred_0, y_pred_1)
    final_results = [0 if x[0] > x[1] else 1 for x in final_results]


    pred = zip(grd_truth, final_results)
    acc = [1 if int(x[0]) == x[1] else 0 for x in pred]
    return sum(acc)/float(len(grd_truth))

"""
Compute prior parameters for mean and var
"""
print '---------------Computing the mean and var--------------'
# Compute the probs of x given y
df_y0 = df_norm[df_norm[57] == 0].iloc[:, :57]
m_y0, var_y0 = compute_training_probs(df_y0)

df_y1 = df_norm[df_norm[57] == 1].iloc[:, :57]
m_y1, var_y1 = compute_training_probs(df_y1)

"""
Compute training error
"""
grd_truth_train = df_norm[57]

# Compute probabilites for test data being class 0
y_pred_0 = compute_prob_for_class(df_train, m_y0, var_y0, log_p_y0)

# Compute probabilites for test data being class 1
y_pred_1 = compute_prob_for_class(df_train, m_y1, var_y1, log_p_y1)

training_error = 1.0 - compute_accuracy(grd_truth_train, y_pred_0, y_pred_1)
print 'Training Error : {0}'.format(training_error)

"""
Compute test error
"""

# Compute probabilites for test data being class 0
y_pred_0 = compute_prob_for_class(df_test, m_y0, var_y0, log_p_y0)

# Compute probabilites for test data being class 1
y_pred_1 = compute_prob_for_class(df_test, m_y1, var_y1, log_p_y1)

test_error = 1.0 - compute_accuracy(grd_truth, y_pred_0, y_pred_1)
print 'Test Error : {0}'.format(test_error)


