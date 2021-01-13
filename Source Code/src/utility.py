import heapq
import pickle

import numpy as np

USERS = 6040
MOVIES = 3952
TEST_RATIO = 0.4
COLLAB_K = 5
SVD_K = 1000
RANK = 3663  # Calculated using np.linalg.matrix_rank(utility_matrix)


def svd_decomposition(arg1, arg2, arg3, matrix, r):
    """Perform SVD decomposition of a given matrix

    Parameters
    ----------
    arg1 : str
        Name of the U matrix with which it should have been saved/loaded
    arg2 : str
        Name of the Sigma matrix with which it should have been saved/loaded
    arg3 : str
        Name of the V matrix with which it should have been saved/loaded
    matrix: numpy matrix
        Matrix of which SVD decomposition has been performed
    r: int
        Number of singular values to be retained during decomposition

    Returns
    -------
    tuple
        A tuple containing U, Sigma and Vt matrix
    """
    try:
        with open(str("files/" + arg1), 'rb') as f:
            u = pickle.load(f)
        with open(str("files/" + arg2), 'rb') as f:
            d = pickle.load(f)
        with open(str("files/" + arg3), 'rb') as f:
            vt = pickle.load(f)
    except FileNotFoundError:
        u, d, vt = np.linalg.svd(matrix)
        u = u[:, 0:r]
        d = np.diag(d[0:r])
        vt = vt[0:r]

        # Below method can be used for calculating SVD from scratch
        # d, u = np.linalg.eig(np.dot(matrix, matrix.T))
        # u, d = u.real, d.real
        # idx = d.argsort()[::-1][0:r]
        # d = np.diag(np.sqrt(d[idx]))
        # u = u[:, idx]
        # d2, v = np.linalg.eig(np.dot(matrix.T, matrix))
        # v, d2 = v.real, d2.real
        # idx = d2.argsort()[::-1][0:r]
        # v = v[:, idx]
        #
        # def change_vector_sign(col):
        #     return -col if col[0] < 0 else col
        #
        # u = np.apply_along_axis(change_vector_sign, 0, u)
        # vt = np.apply_along_axis(change_vector_sign, 1, v.T)

        with open(str("files/" + arg1), 'wb+') as f:
            pickle.dump(u, f)
        with open(str("files/" + arg2), 'wb+') as f:
            pickle.dump(d, f)
        with open(str("files/" + arg3), 'wb+') as f:
            pickle.dump(vt, f)

    return u, d, vt


def find_filter(d):
    """Finds number of singular values for which 90% energy can be retained

    Parameters
    ----------
    d : numpy matrix
        An diagonal matrix containing singular values

    Returns
    -------
    int
        Number of singular values for which 90% energy has been retained
    """
    singular_values = np.diag(d) ** 2
    total = np.sum(singular_values)
    curr_sum = 0
    count = 0
    for i in range(len(singular_values)):
        curr_sum += singular_values[i]
        if curr_sum / total > 0.9:
            break
        count += 1
    return count


def final_results(actual, predicted):
    """Finds rmse, precision of top k, and spearman correlation

    Parameters
    ----------
    actual : numpy array
        Actual ratings of the user
    predicted : numpy array
        Predicted ratings by the recommender system

    Returns
    -------
    tuple
        A tuple containing rmse, precision of top k and spearman correlation
    """
    n = len(actual)
    diff = np.sum((actual - predicted) ** 2)
    rmse = np.sqrt(diff / n)

    spearman = 1 - ((6 * diff) / (n ** 3 - n))

    PRECISION_K = int(n / 5)
    top_actual = set(heapq.nlargest(PRECISION_K, range(n), actual.__getitem__))
    top_predicted = set(heapq.nlargest(PRECISION_K, range(n), predicted.__getitem__))
    precision = len(top_actual.intersection(top_predicted)) / PRECISION_K

    return rmse, precision, spearman
