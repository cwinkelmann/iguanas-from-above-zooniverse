import glob
import json
import random
import shutil
import statistics

from kneed import KneeLocator
from matplotlib import pyplot as plt
from pathlib import Path

import pandas as pd
from loguru import logger

import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, mean_squared_error

from PIL import Image

import matplotlib.cm as cm


from zooniverse.utils.basic_clustering import (
    find_inertia,
    plot_data,
    get_curve_direction,
    get_curve_shape,
    bic_score,
)
from zooniverse.utils.plotting import plot_clusters, plot_clusters_v2

BOX_SIZE = 60

## a list of users which are supposed to be trustworthy
trustworthy_users = [
    "Pamelavans" "robert1601",
    "Darkstar1977",
    "H.axon",
    "Quynhanhdo" "Taylor_Q" "databanana" "Heuvelmans" "Big_Ade" "babyemperor" "HW1881",
]


def get_all_image_paths(image_source: Path, cache_dir: Path) -> pd.DataFrame:
    """
    search for images in subfolders which we use to join the real path to the dataframe with classification report

    :type image_source: Path
    :type cache_dir: Path
    :param cache_dir:
    :param image_source:
    :return:
    """
    if image_source is None:
        return None

    image_list = glob.glob(str(image_source.joinpath("**/*.jpg")), recursive=True)

    if len(image_list) == 0:
        logger.warning(f"Found {len(image_list)} images in the folder {image_source}")

    images_split_list = [Path(x).parts for x in image_list]

    image_dict = {
        x[-1]: {
            "mission_name": x[-2],
            "image_name": x[-1],
            "image_path": str(Path(*list(x))),
        }
        for x in images_split_list
    }

    for key, value in image_dict.items():
        # logger.info(f"load image: {value['image_path']}")
        im = Image.open(value["image_path"])
        width, height = im.size
        image_dict[key]["width"] = width
        image_dict[key]["height"] = height

        im.close()

    logger.info("done with processing Image Metadata.")
    return pd.DataFrame(image_dict).T


def deduplicate_entries(merged_dataset):
    """
    Each image is marked multiple times. Once per user, N marks for N iguanas.

    :param merged_dataset:
    :return:
    """
    # iterate over the dataset and merge the marks
    compacted_marks_per_data_frame = []
    marks_per_data_frame = (
        pd.DataFrame(merged_dataset)
        .reset_index(drop=False)
        .groupby("image_name")["marks"]
        .apply(list)
        .reset_index()
    )

    for image_with_list_of_list_of_mark in marks_per_data_frame.to_dict(
        orient="records"
    ):
        image_name = image_with_list_of_list_of_mark["image_name"]
        compacted_marks = []
        for mark_group in image_with_list_of_list_of_mark["marks"]:
            for mark in mark_group:
                compacted_marks.append(mark)

        compacted_marks_per_data_frame.append(
            {"image_name": image_name, "marks": compacted_marks}
        )

    return compacted_marks_per_data_frame


def get_mark_overview(df_marks):
    """
    group all the marks by the zooniverse volunteers

    :param metadata_records:
    :return:
    """

    annotations_count = list(df_marks.groupby("user_name")["user_name"].count())

    return annotations_count


def reformat_marks(metadata_record_FMO01_68):
    """
    get one part from the flat structure

    :param metadata_record_FMO01_68:
    :return:
    """
    flat_struct = []

    for metadata_record in metadata_record_FMO01_68:
        for mark in metadata_record["marks"]:
            marks = {
                "x": int(mark["x"]),
                "y": int(mark["y"]),
                "user": metadata_record["user_name"],
            }
            flat_struct.append(marks)

    df_structure = pd.DataFrame(flat_struct)

    return df_structure


def get_estimated_DBSCAN_count(
    df_marks, image_name, params, output_path=None, plot=False
):
    """
    DBSCAN clustering
    :param df_marks:
    :param image_name:
    :param params:
    :param output_path:
    :return:
    """

    X = df_marks.to_numpy()
    # X = StandardScaler().fit_transform(X)
    sc = StandardScaler().fit(X)
    X = sc.transform(X)
    result = {}

    eps, min_samples = params

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    fig, ax = plt.subplots(1)
    X = sc.inverse_transform(X)

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )
        # noise
        xy = X[class_member_mask & ~core_samples_mask]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(
        f"DBSCAN: {n_clusters_} for eps={eps} and min_samples={min_samples} for {image_name}"
    )
    ax.invert_yaxis()
    if output_path:
        fig.savefig(
            output_path.joinpath(f"{image_name}_dbscan_{eps}_{min_samples}.png")
        )
    if plot:
        plt.show()
    else:
        pass

    plt.close()

    if n_clusters_ in (0, 1):
        # logger.warning(f"DBSCAN found {n_clusters_} clusters for {image_name}")
        return {
            "image_name": image_name,
            "dbscan_count": n_clusters_,
            "dbscan_noise": n_noise_,
            "dbscan_silouette_score": None,
            "dbscan_BIC_score": None,
        }

    result[f"dbscan_count"] = n_clusters_
    result[f"dbscan_noise"] = n_noise_
    result[f"dbscan_silouette_score"] = metrics.silhouette_score(X, labels)
    result[f"dbscan_BIC_score"] = bic_score(X, labels)

    result["image_name"] = image_name
    return result


def kmeans_knee(
    df_marks, annotations_count, output_path, image_name, plot_diagrams=True, show=False
):
    """
    estimate the location of objects using the elbow method

    :param df_marks:
    :param annotations_count:
    :param output_path:
    :param image_name:
    :return:
    """
    from utils.basic_clustering import plot_inertia

    X = df_marks.to_numpy()
    num_max_clusters = max(annotations_count) + 3
    inertia_list = find_inertia(X, num_max_clusters=num_max_clusters)

    x = list(range(1, num_max_clusters))

    direction = get_curve_direction(inertia_list)
    curve = get_curve_shape(inertia_list)
    # logger.info(f"curve: {curve}, direction: {direction}")

    kneedle = KneeLocator(x, inertia_list, S=1.0, curve=curve, direction=direction)

    if kneedle.knee is None:
        return {"image_name": image_name, "kmeans_knee": None}

    if plot_diagrams:
        # plot the inertia curve
        plot_inertia(
            inertia_list,
            num_max_clusters,
            knee_point=kneedle.knee,
            title=f"Inertia Values for Number of Clusters of {image_name}",
            output_path=(
                output_path.joinpath(f"{image_name}_inertia.png")
                if output_path
                else None
            ),
            show=show,
        )

    clusterer = KMeans(n_clusters=kneedle.knee, n_init="auto", random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    if plot_diagrams and output_path:
        (
            plot_data(
                X=X,
                kmeans_model=clusterer,
                num_clusters=int(kneedle.knee),
                title=f"Clustering with optimized number of clusters for {image_name}",
                output_path=output_path.joinpath(
                    f"{image_name}_kmeans_knee={int(kneedle.knee)}.png"
                ),
            )
            if output_path
            else None
        )

    return {"image_name": image_name, "kmeans_knee": int(kneedle.knee)}


def kmeans_silouette(
    df_marks,
    annotations_count,
    image_name,
    output_path: Path = None,
    plot_diagrams=False,
    show=False,
):
    """
    estimate the location using Silhouette analysis
    """

    silhouettes = []
    X = df_marks.to_numpy()
    if max(annotations_count) == 1:
        logger.warning(f"It seems there can be only one cluster")
        return {"image_name": image_name, "kmeans_sillouette_count": 1}

    if max(annotations_count) == 0:
        logger.warning(f"There is no cluster")
        return {"image_name": image_name, "kmeans_sillouette_count": 0}

    range_n_clusters = range(2, min(len(X), max(annotations_count) + 2))

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouettes.append(silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        ## Visualise the silhouette scores
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
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

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Image pixel in X - Dimension")
        ax2.set_ylabel("Image pixel in Y - Dimension")
        ax2.invert_yaxis()
        plt.suptitle(
            f"Silhouette analysis for KMeans clustering on {image_name} n_clusters = {n_clusters}",
            fontsize=14,
            fontweight="bold",
        )
        if plot_diagrams:
            if output_path:
                figure_path = (
                    f"{output_path.joinpath(image_name)}_kmeans_n={n_clusters}.png"
                )
                logger.info(f"{figure_path}")
                plt.savefig(f"{figure_path}")
        if show:
            plt.show()
        plt.close()

    plt.close()
    # logger.info(f"finished annotations count for {image_name}")

    # logger.info("visualise the optimal cluster count.")
    try:
        fig, ax = plt.subplots(1, 1)
        plt.plot(range_n_clusters, silhouettes)
        # print(max(silhouettes))
        max_sillouette_score = max(zip(silhouettes, range_n_clusters))[1]

        plt.axvline(x=max_sillouette_score, color="b", label="axvline - full height")
        plt.text(
            max_sillouette_score - 3.5,
            1.5,
            s=f"amount of cluster, max. sillouette score: {round(max_sillouette_score) ,3}",
            bbox=dict(facecolor="red", alpha=0.5),
        )

        if plot_diagrams:
            if output_path is not None:
                plt.savefig(
                    f"{output_path.joinpath(image_name)}_kmeans_optimal_silouette_score.png"
                )
                # logger.info(f"save figure: {output_path.joinpath(image_name)}_kmeans_optimal_silouette_score.png")
            else:
                plt.show()

    except Exception as e:
        logger.error(e)

    plt.close()

    try:
        return {
            "image_name": image_name,
            "kmeans_sillouette_count": max(zip(silhouettes, range_n_clusters))[1],
        }

    except ValueError as e:
        logger.error(
            {
                "image_name": image_name,
                "kmeans_sillouette_count": max(annotations_count),
            }
        )
        return {
            "image_name": image_name,
            "kmeans_sillouette_count": max(annotations_count),
        }


def kmeans_BIC(
    df_marks, annotations_count, image_name, plot=False, output_path: Path = None
):
    """
    estimate the kmeans cluster location using BIC analysis
    """

    bics = []
    X = df_marks.to_numpy()

    if max(annotations_count) == 0:
        logger.warning(f"There is no cluster")
        return {"image_name": image_name, "kmeans_BIC_count": 0}

    range_n_clusters = range(1, min(len(X), max(annotations_count) + 2))

    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        bic_avg = bic_score(X, cluster_labels)
        bics.append(bic_avg)

        if plot:
            # Create a subplot with 1 row and 2 columns
            fig, (ax2) = plt.subplots(1, 1)
            fig.set_size_inches(9, 7)

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

            ax2.scatter(
                X[:, 0],
                X[:, 1],
                marker=".",
                s=30,
                lw=0,
                alpha=0.7,
                c=colors,
                edgecolor="k",
            )

            # Labeling the clusters
            centers = clusterer.cluster_centers_
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

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Image pixel in X - Dimension")
            ax2.set_ylabel("Image pixel in Y - Dimension")

            plt.suptitle(
                f"BIC analysis for KMeans clustering on {image_name} n_clusters = {n_clusters}",
                fontsize=14,
                fontweight="bold",
            )

            if output_path:
                figure_path = (
                    f"{output_path.joinpath(image_name)}_kmeans_bic_n={n_clusters}.png"
                )
                logger.info(f"{figure_path}")
                plt.savefig(f"{figure_path}")
            else:
                plt.show()
            plt.close()

    if plot:
        plt.close()

        logger.info("visualise the optimal cluster count.")
        try:
            fig, ax = plt.subplots(1, 1)
            plt.plot(range_n_clusters, bics)
            # print(max(bics))
            max_sillouette_score = max(zip(bics, range_n_clusters))[1]

            plt.axvline(
                x=max_sillouette_score, color="b", label="axvline - full height"
            )
            plt.text(
                max_sillouette_score - 3.5,
                1.5,
                s=f"amount of cluster, max. BIC score: {round(max_sillouette_score) ,3}",
                bbox=dict(facecolor="red", alpha=0.5),
            )

            if output_path is not None:
                plt.savefig(
                    f"{output_path.joinpath(image_name)}_kmeans_optimal_BIC_score.png"
                )
                logger.info(
                    f"save figure: {output_path.joinpath(image_name)}_kmeans_optimal_BIC_score.png"
                )
            else:
                plt.show()
            plt.close()
        except Exception as e:
            logger.error(e)
            plt.close()

    return {
        "image_name": image_name,
        "kmeans_BIC_count": max(zip(bics, range_n_clusters))[1],
    }


def HDBSCAN_Wrapper(
    df_marks,
    annotations_count,
    image_name,
    params,
    min_cluster_size=3,
    plot=False,
    show=False,
    output_path: Path = None,
):
    """
    estimate the BIC score based on HDBSCAN clustering
    """
    from sklearn.cluster import HDBSCAN

    bics = []
    X = df_marks.to_numpy()
    # max_cluster_size = None
    for eps, min_cluster_size, max_cluster_size in params:

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            cluster_selection_epsilon=eps,
            allow_single_cluster=True,
            store_centers="centroid",
        )
        cluster_labels = clusterer.fit_predict(X)
        non_noise_cluster_labels = cluster_labels[cluster_labels >= 0]
        if output_path is not None:
            plot_clusters_v2(
                cluster_labels,
                X,
                image_name,
                centers=clusterer.centroids_,
                figure_title=f"HDBSCAN Clustering for {image_name}",
                main_title=f"n={len(clusterer.centroids_)} eps={eps} min_cluster_size={min_cluster_size} max_cluster_size={max_cluster_size}",
                figure_path=f"{output_path.joinpath(image_name)}_hdbscan_bic_n={len(clusterer.centroids_)}.png",
            )

        if plot or show:
            plot_clusters_v2(
                cluster_labels,
                X,
                image_name,
                centers=clusterer.centroids_,
                figure_title=f"HDBSCAN Clustering for {image_name}",
                main_title=f"n={len(clusterer.centroids_)} eps={eps} min_cluster_size={min_cluster_size} max_cluster_size={max_cluster_size}",
            )

        bic_avg = bic_score(X, cluster_labels)
        bic_avg = {
            "image_name": image_name,
            "with_noise": True,
            "HDBSCAN_count": len(clusterer.centroids_),
            "bic_avg": bic_avg,
            "eps": eps,
            "min_cluster_size": min_cluster_size,
            "max_cluster_size": max_cluster_size,
            "noise_points": len(cluster_labels[cluster_labels < 0]),
        }
        bics.append(bic_avg)

    df_bics = pd.DataFrame(bics)
    if output_path is not None:
        df_bics.to_csv(output_path.joinpath(f"{image_name}_hdbscan_bic.csv"))

    return df_bics


def stats_calculation(df_exp):
    """
    compare the ground truth from the gold standard
    :param df_comparison:
    :return:
    """

    df_exp = df_exp[~df_exp.median_count.isna()]

    df_exp["count_diff_median"] = df_exp.count_total - df_exp.median_count
    df_exp["count_diff_dbscan"] = df_exp.count_total - df_exp.dbscan_count

    df_exp.sort_values(by="median_count", ascending=False)

    df_exp_sum = df_exp[
        [
            "count_total",
            "mean_count",
            "median_count",  # "sillouette_count",
            "dbscan_count",
            "count_diff_median",  # "count_diff_kmeans",
            "count_diff_dbscan",
        ]
    ].sum()

    mse_errors = {}

    mse_errors["rmse_median"] = mean_squared_error(
        df_exp.count_total, df_exp.median_count, squared=False
    )
    mse_errors["rmse_mean"] = mean_squared_error(
        df_exp.count_total, df_exp.mean_count, squared=False
    )
    # mse_errors["rmse_silloutte"] = mean_squared_error(df_exp.count_total, df_exp.sillouette_count, squared=False)
    mse_errors["rmse_dbscan"] = mean_squared_error(
        df_exp.count_total, df_exp.dbscan_count, squared=False
    )

    return df_exp, df_exp_sum, pd.Series(mse_errors)


def get_annotation_count_stats(annotations_count, image_name):
    """
    build a dictionary with the statistics of the annotations count
    :param annotations_count:
    :param image_name:
    :return:
    """
    return {
        "image_name": image_name,
        "median_count": statistics.median(annotations_count),
        "mean_count": round(statistics.mean(annotations_count), 2),
        "mode_count": statistics.mode(annotations_count),
        "users": len(annotations_count),
        "sum_annotations_count": sum(annotations_count),
        "annotations_count": sorted(annotations_count),
    }


def compare_dbscan_hyp_v2(df_flat, params, output_plot_path: Path, plot=False):
    """
    run dbscan clustering on all images and compare the results with the gold standard

    :param params:
    :param phase_tag:
    :return:
    """

    dbscan_localizations = []

    image_name = df_flat.iloc[0]["image_name"]
    for eps, min_samples in params:
        ds_SCAN_localization = get_estimated_DBSCAN_count(
            df_marks=df_flat[["x", "y"]],
            output_path=output_plot_path,
            plot=plot,
            image_name=image_name,
            params=(eps, min_samples),
        )
        ds_SCAN_localization["eps"] = eps
        ds_SCAN_localization["min_samples"] = min_samples
        dbscan_localizations.append(ds_SCAN_localization)

    return dbscan_localizations
