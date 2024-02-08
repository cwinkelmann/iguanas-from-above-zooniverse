import pandas as pd
from loguru import logger


def filter_df_user_threshold(df_, user_threshold=None):
    """
    threshold means how many marks should exist in for an image. Otherwise, it is considered noise.

    :return:
    """
    n = 0
    ls_groups = []
    for image_name, df_group in df_.groupby("image_name"):
        if user_threshold is None:
            ls_groups.append(df_group)

        else:
            ls_users = df_group.groupby("user_name")
            if len(ls_users) > user_threshold:
                ls_groups.append(df_group)
            else:
                logger.warning(f"The image {image_name} has only {len(ls_users)} users, which is less than the threshold of {user_threshold}")
                n = n + 1

    logger.warning(f"filtered out {n} images")
    return pd.concat(ls_groups)


def filter_remove_marks(df: pd.DataFrame):
    """
    remove marks, for partials etc
    :param df:
    :return:
    """
    df_partials = df[df["tool_label"].isin(["Partial iguana"])]
    # remove partials
    df = df[~df["tool_label"].isin(["Partial iguana"])]
    logger.warning(f"removed {len(df_partials)} partial marks")
    logger.warning(f"After filter_func {len(df.image_name.unique())} images are left")
    return df

