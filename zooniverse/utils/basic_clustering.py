"""
functions to help clustering

"""

import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.cluster import KMeans

from scipy.spatial.distance import (
    cdist,
)  # for calculating distances between points and clusters

plt.rcParams["figure.figsize"] = [8, 8]
sns.set_style("whitegrid")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_data(
    X, kmeans_model=None, num_clusters=0, colors=colors, title=None, output_path=None
):
    """
    Helper function for visualizing the clusters and the data.
    """
    if num_clusters == 0:
        plt.scatter(X[:, 0], X[:, 1], c=colors)
    else:
        # loop over kmeans clusters
        for cluster in range(num_clusters):
            plt.scatter(
                X[kmeans_model.labels_ == cluster, 0],
                X[kmeans_model.labels_ == cluster, 1],
                c=colors[cluster],
            )
    if title is not None:
        plt.title(title)
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
        plt.close()


def find_inertia(X, num_max_clusters=10):
    inertia_list = []
    distortions = []

    for num_clusters in range(1, num_max_clusters):
        kmeans_model = KMeans(n_clusters=num_clusters, n_init="auto", init="k-means++")
        kmeans_model.fit(X)

        distortions.append(
            sum(np.min(cdist(X, kmeans_model.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0]
        )
        inertia_list.append(kmeans_model.inertia_)

    return inertia_list


def calculate_cohesion(X, y):
    """
    Calculate the cohesion for a given cluster label.
    :param X:
    :param y:
    :return:
    """
    from sklearn.metrics.pairwise import cosine_similarity


def find_calinski_harabasz_score(X, y):
    from sklearn.metrics import calinski_harabasz_score


def calculate_inertia(X, y):
    """
    Calculate the inertia for a given cluster label.
    :param X:
    :param y:
    :return:
    """
    inertia_list = []
    for label in np.unique(y):
        cluster = X[y == label]
        inertia_list.append(np.sum(np.square(cluster - cluster.mean())))
    return inertia_list


def calculate_distortion(X, y):
    """
    Calculate the distortion for a given cluster label.
    :param X:
    :param y:
    :return:
    """
    distortion_list = []
    for label in np.unique(y):
        cluster = X[y == label]
        distortion_list.append(np.sum(np.square(cluster - cluster.mean())))
    return distortion_list


def bic_score(X, labels):
    """
    BIC score for the goodness of fit of clusters.
    This Python function is directly translated from the GoLang code made by the author of the paper.
    https://towardsdatascience.com/are-you-still-using-the-elbow-method-5d271b3063bd
    The original code is available here: https://github.com/bobhancock/goxmeans/blob/a78e909e374c6f97ddd04a239658c7c5b7365e5c/km.go#L778
    """

    n_points = len(labels)
    n_clusters = len(set(labels))
    n_dimensions = X.shape[1]

    n_parameters = (n_clusters - 1) + (n_dimensions * n_clusters) + 1

    loglikelihood = 0
    for label_name in set(labels):
        X_cluster = X[labels == label_name]
        n_points_cluster = len(X_cluster)
        centroid = np.mean(X_cluster, axis=0)
        if len(X_cluster) == 1:
            logger.warning(f"only one point in the cluster")
        variance = np.sum((X_cluster - centroid) ** 2) / (len(X_cluster) - 1)

        loglikelihood += (
            n_points_cluster * np.log(n_points_cluster)
            - n_points_cluster * np.log(n_points)
            - n_points_cluster * n_dimensions / 2 * np.log(2 * math.pi * variance)
            - (n_points_cluster - 1) / 2
        )

    bic = loglikelihood - (n_parameters / 2) * np.log(n_points)

    return bic


def plot_inertia(
    inertia_list,
    num_max_clusters=10,
    knee_point=3,
    title=None,
    output_path=None,
    show=False,
):
    """
    Plot the inertia values for different number of clusters.

    :param inertia_list:
    :param num_max_clusters:
    :return:
    """
    plt.plot(range(1, num_max_clusters), inertia_list)
    plt.scatter(range(1, num_max_clusters), inertia_list)
    if knee_point is not None:
        plt.scatter(knee_point, inertia_list[knee_point], marker="X", s=300, c="r")
    else:
        logger.error(
            f"There is no knee point given. The inertia plot will be saved without a knee point."
        )

    plt.xlabel("Number of Clusters", size=13)
    plt.ylabel("Inertia Value", size=13)
    if title is None:
        title = f"Different Inertia Values for Different Number of Clusters"
    plt.title(title, size=17)

    if show:
        plt.show()
    else:
        plt.savefig(output_path)

    plt.close()


def get_curve_direction(ls):
    # test if the list is increasing or decreasing
    decreasing = {all(ls[i] >= ls[i + 1] for i in range(len(ls) - 1))}
    if decreasing:
        direction = "decreasing"
    else:
        direction = "increasing"

    return direction


def get_curve_shape(ls):
    ## test if the list is convex or concave
    convex = {all(2 * ls[i] <= ls[i - 1] + ls[i + 1] for i in range(1, len(ls) - 1))}

    if convex:
        curve = "convex"
    else:
        curve = "concave"

    return curve
