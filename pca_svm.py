import numpy as np
from numpy.linalg import eig
from numpy import cov
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


def reduce_dimensions_pca(data):
    """

    :param data: data to reduce dimension
    :return: data with reduced dimensions
    """

    mean = np.mean(data, axis=0)
    centered = data - mean

    cov_mat = cov(centered.T)
    eig_val, eig_vec = eig(cov_mat)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(eig_val))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Checking the eig values
    eig_val_sorted = np.array([x[0] for x in eig_pairs])

    # Ordering the eig vectors into columns
    eig_vec_sorted = np.array([x[1] for x in eig_pairs]).T
    return mean, eig_val_sorted, eig_vec_sorted


if __name__ == "__main__":
    # Loading the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 28 * 28)
    x_test = x_test.reshape((10000, 28 * 28))
    x_train = x_train / 255
    x_test = x_test / 255

    # getting the mean, eig_val and eig_vecs
    mean, eig_val, eig_vec = reduce_dimensions_pca(x_train)

    # Linear Kernel
    print("Running Linear Kernels")
    dimensions = [40, 80, 200]
    c_param = [0.01, 0.1, 1, 10]
    for dim in dimensions:
        for param in c_param:
            clf = LinearSVC(random_state=0, tol=1e-5, C=param)
            x_train_new = x_train - mean
            reduced = x_train_new.dot(eig_vec[:, :dim])
            clf.fit(reduced, y_train)
            x_test_new = x_test - mean
            x_test_proj = x_test_new.dot(eig_vec[:, :dim])
            pred = clf.predict(x_test_proj)
            print('c_param : {0} , dimensions {1}, acc {2}'.format(param, dim, accuracy_score(y_test, pred)))


    # Rbf kernel
    print("Running Rbf Kernels")
    dimensions = [40, 80, 200]
    c_param = [0.01 , 0.1 , 1, 10]
    for dim in dimensions:
        for param in c_param:
            clf = SVC(random_state=0, tol=1e-5, C=param, gamma = 0.1)
            x_train_new = x_train - mean
            reduced = x_train_new.dot(eig_vec[:,:dim])
            clf.fit(reduced, y_train)
            x_test_new = x_test - mean
            x_test_proj = x_test_new.dot(eig_vec[:, :dim])
            pred = clf.predict(x_test_proj)
            print('c_param : {0} , dimensions {1}, acc {2}'.format(param, dim, accuracy_score(y_test, pred)))


    # Best results
    # Best results
    dimensions = [40, 80, 200]
    c_param = [5]
    for dim in dimensions:
        for param in c_param:
            clf = SVC(random_state=0, tol=1e-5, C=param, gamma=0.05)
            x_train_new = x_train - mean
            reduced = x_train_new.dot(eig_vec[:, :dim])
            clf.fit(reduced, y_train)
            x_test_new = x_test - mean
            x_test_proj = x_test_new.dot(eig_vec[:, :dim])
            pred = clf.predict(x_test_proj)
            print('c_param : {0} , dimensions {1}, acc {2}'.format(param, dim, accuracy_score(y_test, pred)))