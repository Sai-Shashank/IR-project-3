import numpy as np

from src.CUR import get_cur_results
from src.SVD import get_svd_results
from src.collab import get_collab_results
from src.utility import USERS, MOVIES


def get_utility_matrix():
    """Generates an utility matrix from the ratings txt file

    Parameters
    ----------

    Returns
    -------
    numpy matrix
        An (User x Movie) sparse matrix containing all the ratings
    """
    matrix = np.zeros((USERS, MOVIES))
    with open("dataset/ratings.dat", 'r') as ratings:
        for rating in ratings:
            comp = rating.split('::')
            matrix[int(comp[0]) - 1][int(comp[1]) - 1] = int(comp[2])
    return matrix


if __name__ == '__main__':
    original_matrix = get_utility_matrix()  # users * movies

    print("----------Collaborative Filtering:-------------------------------------------------------------------------")
    first, second = get_collab_results(original_matrix.T)
    print("\nRMSE = " + str(first[0]) + ", Precision = " + str(first[1]) + ", Spearman Correlation= " + str(first[2]))
    print("\nCollaborative Filtering with Global Baseline Approach:")
    print("RMSE = " + str(second[0]) + ", Precision = " + str(second[1]) + ", Spearman Correlation= " + str(second[2]))

    print("\n\n----------SVD:-----------------------------------------------------------------------------------------")
    first, second = get_svd_results(original_matrix)
    print("\nRMSE = " + str(first[0]) + ", Precision = " + str(first[1]) + ", Spearman Correlation= " + str(first[2]))
    print("\nSVD with 90% energy:")
    print("RMSE = " + str(second[0]) + ", Precision = " + str(second[1]) + ", Spearman Correlation= " + str(second[2]))

    print("\n\n----------CUR:-----------------------------------------------------------------------------------------")
    first, second = get_cur_results(original_matrix)
    print("\nRMSE = " + str(first[0]) + ", Precision = " + str(first[1]) + ", Spearman Correlation= " + str(first[2]))
    print("\nCUR with 90% energy:")
    print("RMSE = " + str(second[0]) + ", Precision = " + str(second[1]) + ", Spearman Correlation= " + str(second[2]))
