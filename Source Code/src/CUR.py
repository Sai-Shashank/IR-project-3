import pickle
import time

import numpy as np

from src.utility import find_filter
from src.utility import final_results, svd_decomposition, SVD_K


def cur_decomposition(matrix, sample_r):
    """Perform SVD decomposition of a given matrix

    Parameters
    ----------
    matrix : numpy matrix
        A matrix of which CUR decomposition needs to be done
    sample_r : int
        Number of singular values that needs to be retained during decomposition

    Returns
    -------
    tuple
        A tuple containing C, U and R matrix
    """
    try:
        with open("files/cur_c", 'rb') as f:
            c = pickle.load(f)
        with open("files/cur_u", 'rb') as f:
            u = pickle.load(f)
        with open("files/cur_r", 'rb') as f:
            r = pickle.load(f)
    except FileNotFoundError:
        total = np.sum(matrix ** 2)
        rows, cols = matrix.shape
        col_values = [np.sum(col ** 2) / total for col in matrix.T]
        row_values = [np.sum(row ** 2) / total for row in matrix]
        random_cols = np.random.choice(range(0, cols), sample_r, p=col_values)
        random_rows = np.random.choice(range(0, rows), sample_r, p=row_values)
        c = matrix[:, random_cols]
        r = matrix[random_rows]
        w = c[random_rows]
        for i in range(sample_r):
            norm = np.sqrt(sample_r * col_values[random_cols[i]])
            if norm > 0:
                c[:, i] /= norm
        for i in range(sample_r):
            norm = np.sqrt(sample_r * row_values[random_rows[i]])
            if norm > 0:
                r[i] /= norm

        x, z, yt = svd_decomposition("cur_u_u", "cur_u_d", "cur_u_vt", w, sample_r)
        u = np.dot(np.dot(yt.T, np.linalg.pinv(z) ** 2), x.T)
        with open("files/cur_c", 'wb+') as f:
            pickle.dump(c, f)
        with open("files/cur_u", 'wb+') as f:
            pickle.dump(u, f)
        with open("files/cur_r", 'wb+') as f:
            pickle.dump(r, f)

    return c, u, r


def get_cur_results(matrix):
    """Apply appropriate operations on a given matrix to use CUR technique to predict ratings

    Parameters
    ----------
    matrix : numpy matrix
        Original matrix on which CUR technique needs to be applied

    Returns
    -------
    tuple
        A tuple containing all performance measures of the CUR technique with and without 90% energy retention
    """
    c, u, r = cur_decomposition(matrix, int(SVD_K * np.log10(SVD_K)))
    print("CUR Decomposition Done!")

    start = time.time()
    reconstructed_matrix = np.dot(np.dot(c, u), r)
    mask = (matrix > 0)
    actual = matrix[mask]
    predicted = reconstructed_matrix[mask]
    result_one = final_results(actual, predicted)
    print("Time taken for CUR = " + str(time.time() - start) + " seconds")

    start = time.time()
    energy_filter = find_filter(u)
    c = c[:, 0:energy_filter]
    u = u[0:energy_filter, 0:energy_filter]
    r = r[0:energy_filter]
    reconstructed_matrix = np.dot(np.dot(c, u), r)
    predicted = reconstructed_matrix[mask]
    result_second = final_results(actual, predicted)
    print("Time taken for CUR with 90% energy = " + str(time.time() - start) + " seconds")

    return result_one, result_second
