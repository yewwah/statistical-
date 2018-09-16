# Loading libraries
import scipy.io as sio
import pandas as pd
import numpy as np

"""
This portion of the code sets the global parameters of the coe and is not expected to change
"""

# Preparing training data
load = sio.loadmat('D:\Yew Wah\Downloads\spamData.mat')
train = load['Xtrain']
train_labels = load['ytrain']
train_final = np.hstack((train, train_labels))

# Convert to df to binarize
df = pd.DataFrame(train_final)
df_binarized = df.applymap(lambda x: 1 if x > 0 else 0)

# Prepare Test Data
test = load['Xtest']
df_test = pd.DataFrame(test)

# Binarizing test data
df_test = df_test.applymap(lambda x: 1 if x > 0 else 0)

# Load ground truths
grd_truth = load['ytest'].flatten().tolist()

# Calculating priors
p_y0 = len(train_labels[train_labels == 0])/ 3056.0
p_y1 = 1 - p_y0

# Setting the Log Priors
log_p_y0 = np.log(p_y0)
log_p_y1 = np.log(p_y1)


def compute_training_probs(df, alpha, beta, eps=1e-9):
    p_x_y_probs = []

    N = len(df)

    # Calculating the probabilities with respect to each feature
    for feat in df.columns:
        p_x_y = (len(df[df[feat] == 0]) + alpha) / (float(N) + alpha + beta)
        p_x_y_probs.append(p_x_y)
    return p_x_y_probs


def compute_prob_for_class(df_test, feature_type, label):
    classes = []
    for idx, row in df_test.iterrows():
        prob = 0
        for col_index in range(len(row)):
            if row[col_index] == 0:
                prob += np.log(feature_type[col_index])
            else:
                prob += np.log(1 - feature_type[col_index])
        classes.append(prob + label)
    return classes


def compute_accuracy(grd_truth, y_pred_0, y_pred_1):
    # Assign class 0 or class 1
    final_results = zip(y_pred_0, y_pred_1)
    final_results = [0 if x[0] >= x[1] else 1 for x in final_results]

    pred = zip(grd_truth, final_results)
    acc = [1 if int(x[0]) == x[1] else 0 for x in pred]
    return sum(acc)/float(len(grd_truth))

alphas = [1]

for alpha in alphas:
    beta = alpha
    # Compute the probs of x given y
    df_y0 = df_binarized[df_binarized[57] == 0]
    x_y0 = compute_training_probs(df_y0, alpha, beta)

    df_y1 = df_binarized[df_binarized[57] == 1]
    x_y1 = compute_training_probs(df_y1, alpha, beta)

    # Compute probabilites for test data being class 0
    y_pred_0 = compute_prob_for_class(df_test, x_y0, log_p_y0)

    # Compute probabilites for test data being class 1
    y_pred_1 = compute_prob_for_class(df_test, x_y1, log_p_y1)

    print compute_accuracy(grd_truth, y_pred_0, y_pred_1)