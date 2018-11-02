import keras
import numpy as np
from keras import datasets
from numpy.linalg import eig
import matplotlib.pyplot as plt
from numpy import cov
from keras.datasets import mnist




def reduce_dimensions_pca(x_train, y_train):
    x_train = x_train.reshape(60000, 28*28)
    mean = np.mean(x_train, axis=0)
    centered = x_train - mean

    cov_mat = cov(centered.T)
    eig_val, eig_vec = eig(cov_mat)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(eig_val))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    return eig_pairs

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()