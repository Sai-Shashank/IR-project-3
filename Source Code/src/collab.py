import time

import numpy as np
import pickle
import heapq

from src.utility import MOVIES, USERS, COLLAB_K, TEST_RATIO, final_results


def gen_sim_score(train_matrix):
    """Generates similarity score of each movie pair

    Parameters
    ----------
    train_matrix : numpy matrix
        Original training dataset matrix

    Returns
    -------
    numpy matrix
        Matrix containing similarity scores
    """
    scores = np.zeros((MOVIES, MOVIES), dtype=float)
    for i in range(MOVIES - 1):
        print("Movie", i)
        movie1 = train_matrix[i]
        for j in range(i + 1, MOVIES):
            movie2 = train_matrix[j]
            mask = (movie1 > 0) & (movie2 > 0)
            x = movie1[mask]
            if len(x) == 0:
                continue
            y = movie2[mask]
            x -= np.average(x)
            y -= np.average(y)
            norm = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
            if norm > 0:
                scores[i][j] = scores[j][i] = np.dot(x, y) / norm
    return scores


def get_prediction(train_matrix, x, y, sim_score, b):
    """Perform operations to find rating prediction for a giver user and a movie

    Parameters
    ----------
    train_matrix : numpy matrix
        Original training dataset matrix
    x: int
        Movie for which prediction needs to be done
    y: int
        User for which prediction needs to be done
    sim_score: numpy matrix
        Similarity scores of each and every movie pairs
    b: numpy matrix
        Matrix containing baseline estimates for each user and movie pair

    Returns
    -------
    tuple
        A tuple with prediction ratings with and without global baseline approach
    """
    similarity = heapq.nlargest(COLLAB_K, [(sim_score[x][i], i) for i in range(MOVIES) if train_matrix[i][y] > 0])
    total_sum = sum(sim for sim, _ in similarity)
    if total_sum == 0:
        return 0, 0
    r1 = sum(sim * train_matrix[ind][y] for sim, ind in similarity) / total_sum
    r2 = b[x][y] + sum(sim * (train_matrix[ind][y] - b[ind][y]) for sim, ind in similarity) / total_sum
    return r1, r2


def get_baseline_estimates(train_matrix):
    """Calculates baseline estimates for each user and movie pair

    Parameters
    ----------
    train_matrix : numpy matrix
        Training dataset matrix  from which info (like various means) is needed for calculating baseline estimates

    Returns
    -------
    numpy matrix
        Matrix containing baseline estimates for each user and movie pair
    """
    mean = np.average(train_matrix[train_matrix > 0])
    u = [user[user > 0] for user in train_matrix.T]
    m = [movie[movie > 0] for movie in train_matrix]
    user_mean = [np.average(temp) if len(temp) > 0 else 0 for temp in u]
    movie_mean = [np.average(temp) if len(temp) > 0 else 0 for temp in m]
    b = np.full((MOVIES, USERS), mean)
    for i in range(MOVIES):
        for j in range(USERS):
            if train_matrix[i][j] > 0:
                b[i][j] += user_mean[j] + movie_mean[i] - 2 * mean
    return b


def get_collab_results(matrix):
    """Apply appropriate operations on a given matrix to use Collaborative Filtering technique to predict ratings

    Parameters
    ----------
    matrix : numpy matrix
        Original matrix on which Collaborative Filtering technique needs to be applied

    Returns
    -------
    tuple
        A tuple containing all performance measures of the Collaborative Filtering with and without Global
        Baseline Approach
     """
    test_matrix = np.copy(matrix[0: int(TEST_RATIO * MOVIES), 0: int(TEST_RATIO * USERS)])
    train_matrix = np.copy(matrix)
    train_matrix[0: int(TEST_RATIO * MOVIES), 0: int(TEST_RATIO * USERS)] = 0

    indices = np.where(test_matrix > 0.0)
    n = len(indices[0])

    try:
        with open("files/collab_scores", 'rb') as f:
            sim_score = pickle.load(f)
    except FileNotFoundError:
        sim_score = gen_sim_score(train_matrix)
        with open("files/collab_scores", 'wb+') as f:
            pickle.dump(sim_score, f)
    print("Collab Scores generated.")

    try:
        with open("files/collab_b_estimates", 'rb') as f:
            baseline_estimates = pickle.load(f)
    except FileNotFoundError:
        baseline_estimates = get_baseline_estimates(train_matrix)
        with open("files/collab_b_estimates", 'wb+') as f:
            pickle.dump(sim_score, f)
    print("Baseline estimates calculated.")

    start = time.time()
    actual = np.empty(n)
    predicted1 = np.empty(n)
    predicted2 = np.empty(n)
    for i in range(n):
        x, y = int(indices[0][i]), int(indices[1][i])
        actual[i] = test_matrix[x][y]
        predicted1[i], predicted2[i] = get_prediction(train_matrix, x, y, sim_score, baseline_estimates)
    common_time = (time.time() - start) / 2
    start = time.time()
    result_one = final_results(actual, predicted1)
    print("Time taken for Collaborative filtering = " + str(time.time() - start + common_time) + " seconds")
    start = time.time()
    result_two = final_results(actual, predicted2)
    print("Time taken for Collaborative filtering with global baseline = " + str(time.time() - start + common_time) + " seconds")
    return result_one, result_two
