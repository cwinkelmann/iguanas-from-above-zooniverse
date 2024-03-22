import unittest
from pathlib import Path

import pandas as pd
import pytest

from zooniverse.analysis import get_mark_overview, get_annotation_count_stats, HDBSCAN_Wrapper
from zooniverse.utils.filters import filter_df_user_threshold


@pytest.fixture(scope="module")
def df_merged_dataset():
    """Fixture to provide sample data for testing."""
    df_merged_dataset = pd.read_csv(Path(__file__).parent.resolve() / "flat_dataset_filtered_Iguanas 3rd launch.csv")

    df_merged_dataset = filter_df_user_threshold(df_merged_dataset, user_threshold=3)

    from zooniverse.utils.filters import filter_remove_marks
    # Check if partials are still in the data. There shouldn't be any
    df_merged_dataset = filter_remove_marks(df_merged_dataset)

    return pd.DataFrame(df_merged_dataset)


def test_basic_statistics(df_merged_dataset):
    basic_stats = []
    for image_name, df_image_name in df_merged_dataset.groupby("image_name"):
        annotations_count = get_mark_overview(df_image_name)

        annotations_count_stats = get_annotation_count_stats(annotations_count=annotations_count,
                                                             image_name=df_image_name.iloc[0]["image_name"])

        ### basic statistics like mean, median
        basic_stats.append(annotations_count_stats)

    df_basic_stats = pd.DataFrame(basic_stats)
    assert df_basic_stats["median_count"].sum() == 3734.5, "3734.5 median count of iguanas"

def test_hdbscan_clustering(df_merged_dataset):
    """test if the hdbscan clustering works as expected with subject_id and image_name"""
    hdbscan_values = []
    for image_name, df_image_name in df_merged_dataset.groupby("image_name"):
        # if less than min_cluster_sizes points are available clustering makes no sense
        if df_image_name.shape[0] >= 5:  # If num_samples is 5 for the min_cluster_size is 5 there is no point in passing data with less than 5 samples

            df_hdbscan = HDBSCAN_Wrapper(df_marks=df_image_name[["x", "y"]],
                                         output_path=None,
                                         plot=False,
                                         show=False,
                                         image_name=image_name,
                                         params=[(0.0, 5, None)])
            hdbscan_values.append(df_hdbscan)

    df_hdbscan = pd.concat(hdbscan_values)
    assert df_hdbscan.shape[0] == 1184, "1184 images!"
    assert df_hdbscan["HDBSCAN_count"].sum() == 4135, "4135 iguanas according to HDBSCAN"

