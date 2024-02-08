from pathlib import Path

from PIL import Image
from loguru import logger
from matplotlib import pyplot as plt, cm
from matplotlib import image
import statistics

## first user marks
from matplotlib.pyplot import figure
import random

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def plot_zooniverse_user_marks(metadata_records, image_path, output_path: Path = None):
    """
    plot all the marks done by Zooniverse users and return the marks

    :param metadata_records:
    :return:
    """

    figure(figsize=(12, 12), dpi=80)

    ## visualise the image
    data = image.imread(image_path)
    implot = plt.imshow(data)

    get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]

    # one color per user
    colors = get_colors(
        len(metadata_records)
    )  # sample return:  ['#8af5da', '#fbc08c', '#b741d0']

    marker_dict = {
        "Adult Male alone": "o",
        "Others (females, young males, juveniles)": "2",
        "Partial iguana": 3,
        "Adult Male with a lek": 4,
        "Adult Male not in a lek": 5,  # Is this the same as 'Adult Male alone'
    }

    n = 0
    labels_all = []
    for image_metadata in metadata_records:
        # to read the image stored in the working directory
        try:
            user_id = int(image_metadata["user_id"])
        except:
            user_id = -1
        image_name = image_metadata["image_name"]

        x_list = [int(dm["x"]) for dm in image_metadata["marks"]]
        y_list = [int(dm["y"]) for dm in image_metadata["marks"]]

        labels = [dm["tool_label"] for dm in image_metadata["marks"]]
        try:
            marker_labels = [marker_dict[v] for v in labels]
        except Exception:
            logger.error(f"These labels are not known: {labels}")
            raise KeyError()

        labels_all = labels_all + marker_labels
        # put a red dot, size 40, at 2 locations:

        ## TODO plot the objects by the label with different markers. So scatter has to be called once for each marker
        plt.scatter(x=x_list, y=y_list, c=colors[n], s=40)

        n = n + 1

    if output_path is not None:
        plt.savefig(
            f"{output_path.joinpath(image_name)}_markers.png"
        )  # ./labelimg_{user_id}.png")
        logger.info(f"{output_path.joinpath(image_name)}_markers.png")
        # plt.show()
        plt.close()

    plt.close()

    return 0


def plot_zooniverse_user_marks_v2(
    df_marks, image_path, image_name, output_path: Path = None, show=False, title=None, fig_size=(8, 8)
):
    """
    plot all the marks done by Zooniverse users and return the marks

    :param df_marks:
    :param image_path:
    :param image_name:
    :param output_path:
    :param metadata_records:
    :return:
    """
    fig, ax = plt.subplots(1, figsize=fig_size)
    print(f"show plots: {show}")
    ## visualise the image
    data = image.imread(image_path)
    pil_im = Image.open(image_path)
    # implot = plt.imshow(data)
    implot = plt.imshow(pil_im)  # , origin='lower')
    plt.grid(None)
    get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]

    # one color per user
    colors = get_colors(
        len(df_marks["user_id"].unique())
    )  # sample return:  ['#8af5da', '#fbc08c', '#b741d0']

    marker_dict = {
        "Adult Male alone": "o",
        "Others (females, young males, juveniles)": "2",
        "Partial iguana": 3,
        "Adult Male with a lek": 4,
        "Adult Male not in a lek": 5,  # Is this the same as 'Adult Male alone'
    }
    df_marks["user_id"] = df_marks["user_id"].fillna(0)

    n = 0
    labels_all = []

    color_mapping = {
        v: k for k, v in dict(enumerate(df_marks["user_id"].unique())).items()
    }

    dot_colors = [colors[color_mapping[user_id]] for user_id in df_marks["user_id"]]

    ax.scatter(x=df_marks["x"], y=df_marks["y"], c=dot_colors, s=40)

    if title is not None:
        ax.set_title(title)
    n = n + 1
    fig.tight_layout()
    if output_path is not None:
        markers_plot_path = f"{output_path.joinpath(image_name)}_markers.png"
        plt.savefig(markers_plot_path)
        logger.info(markers_plot_path)
    if show:
        plt.show()

    plt.close()


def plot_clusters(
    cluster_labels,
    X,
    image_name,
    centers,
    figure_path=None,
    figure_title=None,
    main_title=None,
):
    """
    plot the clusters
    :param output_path:
    :param cluster_labels:
    :param n_clusters:
    :param X:
    :param image_name:
    :return:
    """

    # Create a subplot with 1 row and 2 columns
    fig, (ax2) = plt.subplots(1, 1)
    # fig.set_size_inches(9, 7)
    n_clusters = len(np.unique(cluster_labels))
    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

    ax2.scatter(X[:, 0], X[:, 1], marker="o", s=300, lw=1, c=colors, edgecolor="k")

    ax2.set_title(main_title)
    ax2.set_xlabel("Image pixel in X - Dimension")
    ax2.set_ylabel("Image pixel in Y - Dimension")
    ax2.invert_yaxis()
    plt.suptitle(
        figure_title,
        fontsize=14,
        fontweight="bold",
    )

    if figure_path is not None:
        logger.info(f"{figure_path}")
        plt.savefig(f"{figure_path}")
    else:
        plt.show()
    plt.close()


def plot_clusters_v2(
    cluster_labels,
    X,
    image_name,
    centers,
    figure_path=None,
    figure_title=None,
    main_title=None,
):
    """
    plot the clusters
    :param output_path:
    :param cluster_labels:
    :param n_clusters:
    :param X:
    :param image_name:
    :return:
    """

    # exclude noise points
    cluster_labels_nn = cluster_labels[cluster_labels > -1]
    Xnn = X[cluster_labels > -1]
    n_clusters = len(np.unique(cluster_labels_nn))

    # Create a subplot with 1 row and 2 columns
    fig, (ax2) = plt.subplots(1, 1)
    fig.set_size_inches(9, 7)
    # 2nd Plot showing the actual clusters formed
    # We start from 0.1 instead of 0 to skip black and near-black colors
    modified_nipy_spectral = cm.nipy_spectral(np.linspace(0.1, 1, 256))

    # Create a new colormap object
    new_colormap = cm.colors.ListedColormap(modified_nipy_spectral)
    colors = new_colormap(cluster_labels_nn.astype(float) / n_clusters)

    # colors = cm.nipy_spectral(cluster_labels_nn.astype(float) / n_clusters)

    # plot the clustered points
    ax2.scatter(Xnn[:, 0], Xnn[:, 1], marker="o", s=300, lw=1, c=colors, edgecolor="k")
    # plot the noise
    cluster_labels_noise = cluster_labels[cluster_labels == -1]
    Xnnoise = X[cluster_labels == -1]
    ax2.scatter(
        Xnnoise[:, 0], Xnnoise[:, 1], marker=".", s=30, lw=1, c="black", edgecolor="k"
    )

    # Labeling the clusters
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title(main_title)
    ax2.set_xlabel("Image pixel in X - Dimension")
    ax2.set_ylabel("Image pixel in Y - Dimension")
    ax2.invert_yaxis()
    plt.suptitle(
        figure_title,
        fontsize=14,
        fontweight="bold",
    )

    if figure_path is not None:
        logger.info(f"{figure_path}")
        plt.savefig(f"{figure_path}")
    else:
        plt.show()
    plt.close()
