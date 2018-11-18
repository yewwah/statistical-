import numpy as np
from numpy.linalg import eig
from numpy import cov
from numpy.linalg import pinv, inv
from sklearn.neighbors import KNeighborsClassifier
from keras.datasets import mnist
from sklearn.metrics import accuracy_score


def reduce_dimensions_lda(data, labels):
    """

    :param data: data to reduce dimension
    :return: data with reduced dimensions
    """

    mean_vectors = []
    for cl in range(10):
        mean_vectors.append(np.mean(data[labels == cl], axis=0))

    scatter_mat = np.zeros((784, 784))
    for cl, mv in zip(range(10), mean_vectors):
        class_sc_mat = np.zeros((784, 784))  # scatter matrix for every class
        for row in x_train[y_train == cl]:
            row, mv = row.reshape(784, 1), mv.reshape(784, 1)  # make column vectors
            class_sc_mat += (row - mv).dot((row - mv).T)
        scatter_mat += class_sc_mat  # sum class scatter matrices
    overall_mean = np.mean(x_train, axis=0)

    scatter_btw = np.zeros((784, 784))
    for i, mean_vec in enumerate(mean_vectors):
        n = x_train[y_train == i + 1, :].shape[0]
        mean_vec = mean_vec.reshape(784, 1)  # make column vector
        overall_mean = overall_mean.reshape(784, 1)  # make column vector
        scatter_btw += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    eig_vals, eig_vecs = np.linalg.eig(pinv(scatter_mat).dot(scatter_btw))

    # creating the eig val pairs
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    eig_val_sorted = np.array([x[0] for x in eig_pairs])
    eig_vec_sorted = np.array([x[1] for x in eig_pairs]).T
    overall_mean = overall_mean.reshape(784,)
    return overall_mean, eig_val_sorted, eig_vec_sorted


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 28 * 28)
    x_test = x_test.reshape((10000, 28 * 28))
    x_train = x_train / 255
    x_test = x_test / 255

    # getting the mean, eig_val and eig_vecs
    overall_mean, eig_val, eig_vec = reduce_dimensions_lda(x_train, y_train)
    dimensions = [2, 3, 9]
    clf = KNeighborsClassifier(n_neighbors=3)
    for dim in dimensions:
        x_train_new = x_train - overall_mean
        reduced = x_train_new.dot(eig_vec[:, :dim])
        clf.fit(reduced, y_train)
        x_test_new = x_test - overall_mean
        x_test_proj = x_test_new.dot(eig_vec[:, :dim])
        pred = clf.predict(x_test_proj)
        print('dimensions {0}, acc {1}'.format(dim, accuracy_score(y_test, pred)))

