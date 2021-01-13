import time

import numpy as np

from src.utility import svd_decomposition, final_results, SVD_K, find_filter


def get_svd_results(matrix):
    """Apply appropriate operations on a given matrix to use SVD technique to predict ratings

    Parameters
    ----------
    matrix : numpy matrix
        Original matrix on which SVD technique needs to be applied

    Returns
    -------
    tuple
        A tuple containing all performance measures of the SVD technique with and without 90% energy retention
    """
    u, d, vt = svd_decomposition("svd_u", "svd_d", "svd_v", matrix, SVD_K)
    print("SVD Decomposition Done!")

    start = time.time()
    reconstructed_matrix = np.dot(np.dot(u, d), vt)
    mask = (matrix > 0)
    actual = matrix[mask]
    predicted = reconstructed_matrix[mask]
    result_one = final_results(actual, predicted)
    print("Time taken for SVD = " + str(time.time() - start) + " seconds")

    start = time.time()
    energy_filter = find_filter(d)
    u = u[:, 0:energy_filter]
    d = d[0:energy_filter, 0:energy_filter]
    vt = vt[0:energy_filter]
    reconstructed_matrix = np.dot(np.dot(u, d), vt)
    predicted = reconstructed_matrix[mask]
    result_second = final_results(actual, predicted)
    print("Time taken for SVD with 90% energy = " + str(time.time() - start) + " seconds")

    return result_one, result_second
