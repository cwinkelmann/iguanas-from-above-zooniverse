import json
import shutil
from csv import reader
import glob
from pathlib import Path
from time import sleep
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from zooniverse.config import get_config
from zooniverse.utils.filters import filter_remove_marks
from zooniverse.analysis import get_all_image_paths


def rename_2023_scheme_images_to_zooniverse(image_source: Path):
    """
    get all images in the folder, remove the prefix and use it as a mapping
    :param image_source:
    :return:
    """

    image_list = glob.glob(str(image_source.joinpath("**/*.jpg")), recursive=True)

    new_name = [Path(i) for i in image_list]  # the new name in the dataset is something like ABC_...
    old_name = [Path(i).parent.joinpath(Path(i).name[4:]) for i in image_list]

    df_name_mapping = pd.DataFrame.from_dict({"old_name": old_name, "new_name": new_name})

    return df_name_mapping


def copy(source, target):
    """
    move a foler to a new location
    :param source:
    :param target:
    :return:
    """
    shutil.move(source, target)


def rename_from_schema(df_mapping: pd.DataFrame):
    assert sorted(list(df_mapping.columns)) == sorted(["old_name", "new_name"])

    df_mapping.apply(lambda row: copy(
        row['new_name'],
        row['old_name']),
                     axis=1)


def group_by_image(merged_dataset: list[dict], n=None, threshold=None):
    """
    return list of records for only one image

    threshold means how many marks should exist. Otherwise, it is considered noise.

    :return:
    """
    records_for_one_image = {}
    for record in merged_dataset:
        im = record["image_name"]
        if im not in records_for_one_image:
            records_for_one_image[im] = []
        records_for_one_image[im].append(record)

    if threshold is None:
        records_for_one_image = [r for i, r in records_for_one_image.items()]
    else:
        records_for_one_image = [r for i, r in records_for_one_image.items() if len(r) > threshold]
    # slice the list to a shorter one
    if n is not None:
        records_for_one_image = records_for_one_image[:n]

    return records_for_one_image


def process_zooniverse_phases_flat(df_zooniverse_flat: pd.DataFrame,
                                   image_source: Path,
                                   image_names=None,
                                   subject_ids=None,
                                   filter_func=None) -> pd.DataFrame:
    """
    merge Zooniverse data with the image dictionary, filter for subjects ids and a custom filter function

    @param image_source:
    @param cache_folder:
    @param image_names:
    @param subject_ids:
    @param filter_func:
    @return:

    """

    if image_names:
        logger.info(f"filtering the image dataset for  {len(image_names)} images")
        df_zooniverse_flat = df_zooniverse_flat[
            df_zooniverse_flat.image_name.isin(image_names)]
        logger.info(
            f"working with {len(df_zooniverse_flat.image_name.unique())} images after filtering with image_names")

    if subject_ids:
        logger.info(f"filtering the image dataset for  {len(subject_ids)} subject_ids")
        df_zooniverse_flat = df_zooniverse_flat[
            df_zooniverse_flat.subject_id.isin(subject_ids)]
        logger.info(
            f"working with {len(df_zooniverse_flat.image_name.unique())} images after filtering with subject_ids")

        difference = set(subject_ids).difference(set(df_zooniverse_flat.subject_id.unique()))
        if len(difference) > 0:
            logger.warning(
                f"Some of the subjects ids you used filter are not present in the set. These are: {difference}")

    image_dict = get_all_image_paths(image_source)

    if filter_func is not None:
        df_zooniverse_flat = filter_func(df_zooniverse_flat)

    ## add the image information if it is available
    if image_dict is not None:
        df_zooniverse_flat = df_zooniverse_flat.merge(image_dict, on='image_name', how='left')

    logger.info(f"working with {len(df_zooniverse_flat)} records after filtering with filter_func")

    return df_zooniverse_flat


def read_zooniverse_annotations_v2(annotations_source, phase_tags, cache_dir=None):
    """
    Iterate throughe the zooniverse annotations

    @param annotations_source:
    @param phase_tag: "Iguanas 1st launch" or "Iguanas 2nd launch" or "Iguanas 3rd launch"
    @param cache_dir:
    @return:
    """
    # cache_file = cache_dir.joinpath("cache_zooniverse_annotations.json")
    # https://help.zooniverse.org/next-steps/data-exports/
    # index for certain information.
    idx_USER_NAME = 1
    idx_USER_ID = 2
    idx_WORKFLOW_ID = 4
    idx_PHASE = 5
    idx_WORKFLOW_VERSION = 6
    idx_LABEL_TIME = 7
    idx_USER_INFORMATION = 10
    idx_TASK_INFORMATION = 11
    idx_IMAGE_INFORMATION = 12
    idx_SUBJECT_IDS = 13

    # TASK LABELS PHASE 1
    TASK_LABEL_ARE_THERE_ANY_IGUANAS = 0
    TASK_LABEL_MARK_ALL_IGUANAS = 1
    TASK_LABEL_DIFFICULTY_MARKING_IGUANAS = 2
    TASK_LABEL_ANYTHING_ELSE = 3

    n = 0

    phases = []

    reduced_dataset = []
    flat_dataset = []
    yes_no_dataset = []

    # open file in read mode
    with open(annotations_source, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:

                phase_information = row[idx_PHASE]
                if phase_information not in phases:
                    phases.append(phase_information)
                    logger.info(f"found a new phase tag: {phase_information}")
                user_id = row[idx_USER_ID]
                user_name = row[idx_USER_NAME]
                label_time = row[idx_LABEL_TIME]
                user_information = json.loads(row[idx_USER_INFORMATION])
                labeling_started_at = user_information["started_at"]
                labeling_finished_at = user_information["finished_at"]

                if phase_information in phase_tags:
                    phase_information_passed = True
                    n = n + 1
                    # if n % 10000 == 0:
                    #     logger.info(f"phase: {phase_information}, row number {n} of many, row itself: {row}")

                task_information = json.loads(row[idx_TASK_INFORMATION])
                subject_id = int(row[idx_SUBJECT_IDS])

                # Are there any Iguanas?
                if task_information[TASK_LABEL_ARE_THERE_ANY_IGUANAS][
                    "task"] == "T0" and phase_information in phase_tags:  ## is there anything?

                    # create a dataset only the yes/no information
                    yes_no_dataset.append({"phase_tag": phase_information,
                                           "subject_id": subject_id,
                                           "user_id": user_id,
                                           "user_name": user_name,
                                           "label_time": label_time,
                                           "labeling_started_at": labeling_started_at,
                                           "labeling_finished_at": labeling_finished_at,
                                           "value": task_information[TASK_LABEL_ARE_THERE_ANY_IGUANAS]["value"]
                                           })

                    if task_information[TASK_LABEL_ARE_THERE_ANY_IGUANAS]["value"] == "Yes":
                        # yes there are iguanas
                        # are there any marks
                        if len(task_information[TASK_LABEL_MARK_ALL_IGUANAS]["value"]) > 0:
                            image_information = json.loads(row[idx_IMAGE_INFORMATION])

                            for key, image_information_value in image_information.items():

                                # this has been renamed quite a bit
                                flight_site_code = image_information_value.get("flight_site_code",
                                                                               image_information_value.get(
                                                                                   "flight_code",
                                                                                   image_information_value.get(
                                                                                       "Flight")))
                                if flight_site_code is None:
                                    flight_site_code = image_information_value.get("site")

                                image_name = image_information_value.get("image_name",
                                                                         image_information_value.get("Image_name",
                                                                                                     image_information_value.get(
                                                                                                         "Filename")))
                                if image_name is None:
                                    print(image_information_value)
                                marks = task_information[TASK_LABEL_MARK_ALL_IGUANAS]['value']

                                for mark in marks:
                                    flat_dataset.append({"flight_site_code": flight_site_code,
                                                         "workflow_id": row[idx_WORKFLOW_ID],
                                                         "workflow_version": row[idx_WORKFLOW_VERSION],
                                                         "image_name": image_name,
                                                         "subject_id": subject_id,
                                                         # "mark": mark,
                                                         "x": mark["x"],
                                                         "y": mark["y"],
                                                         "tool_label": mark["tool_label"],
                                                         "phase_tag": phase_information,
                                                         "user_id": user_id,
                                                         "user_name": user_name,
                                                         })

                                reduced_dataset.append(
                                    {
                                        "phase_tag": phase_information,
                                        "flight_site_code": flight_site_code,
                                        "image_name": image_name,
                                        "subject_id": subject_id,
                                        "marks": marks,
                                        "user_id": user_id,
                                        "user_name": user_name,
                                        "label_time": label_time,
                                        "labeling_started_at": labeling_started_at,
                                        "labeling_finished_at": labeling_finished_at,
                                    })

    logger.info("generating flat dataframe from the dataset")
    flat_dataset = pd.DataFrame(flat_dataset)
    reduced_dataset = pd.DataFrame(reduced_dataset)
    df_yes_no_dataset = pd.DataFrame(yes_no_dataset)

    dict_result = {}
    dict_result["reduced_dataset"] = reduced_dataset
    dict_result["flat_dataset"] = flat_dataset
    dict_result["yes_no_dataset"] = df_yes_no_dataset

    return dict_result


def data_prep(phase_tag: str,
              output_path: Path,
              config: Optional[dict],
              input_path,
              filter_combination="expert_goldstandard"
              ):
    """
    prepare the zooniverse classifications
    :param filter_combination: either expert_goldstandard or expert
    :param phase_tag:
    :param output_path:
    :return:
    """
    ds_stats = []
    if not config:
        config = get_config(phase_tag=phase_tag, input_path=input_path, output_path=output_path)
    # images for plotting marks on them
    image_source = config["image_source"]
    # image_source = None
    annotations_source = config["annotations_source"]
    zooniverse_annotation_dataset = read_zooniverse_annotations_v2(annotations_source=annotations_source,
                                                                   phase_tags=[phase_tag])
    # the path of the flat dataset
    flatdataset_path = config["flat_dataset"]
    df_zooniverse_data = zooniverse_annotation_dataset["flat_dataset"]
    df_yes_no_dataset = zooniverse_annotation_dataset["yes_no_dataset"]

    df_yes_no_dataset.to_csv(config["yes_no_dataset"])

    # this user is a spammer
    df_zooniverse_data = df_zooniverse_data[df_zooniverse_data.user_id != 2581179]
    # df_zooniverse_data = df_zooniverse_data[df_zooniverse_data.workflow_version == "134.236"] # DO NOT COMMIT

    # df_zooniverse_data = df_zooniverse_data[df_zooniverse_data.subject_id.isin([47969478])] # TODO do not commit

    df_zooniverse_data.to_csv(flatdataset_path, index=False)
    logger.info(
        f"flat_dataset_Iguanas {phase_tag}.csv: {len(df_zooniverse_data.groupby('image_name').count())} images in classification for {phase_tag}")
    ds_stats.append({"filename": f"{flatdataset_path.name}",
                     "images": len(df_zooniverse_data.groupby('image_name').count())}
                    )

    ## get all images in the launch folder, remove the prefix and use it as a mapping
    ## remove the images so they are not considered in the process
    if image_source is not None:
        image_names = [Path(i).name for i in glob.glob(str(image_source.joinpath("**/*.jpg")), recursive=True)]
        logger.info(f"images source: found {len(image_names)} images in {image_source}")
        ds_stats.append({"filename": image_source.name, "images": len(image_names)})
    else:
        image_names = None

    output_path.mkdir(exist_ok=True)

    # get all the expert count goldstandart data
    goldstandard_expert_count_data = config["goldstandard_data"]

    ## get the expert count
    df_goldstandard_expert_count = pd.read_csv(goldstandard_expert_count_data, sep=";")
    logger.info(
        f"found {len(df_goldstandard_expert_count)} images in {goldstandard_expert_count_data}, the expert counts")
    ds_stats.append({"filename": goldstandard_expert_count_data.name, "images": len(df_goldstandard_expert_count)})

    ## get the gold standard images
    gold_standard_image_subset_path = config["gold_standard_image_subset"]
    df_gold_standard_image_subset = pd.read_csv(gold_standard_image_subset_path, sep=";")

    logger.info(
        f"working with {len(df_gold_standard_image_subset)} images in {gold_standard_image_subset_path}, the goldstandard file.")
    ds_stats.append(
        {"filename": gold_standard_image_subset_path.name, "images": len(df_gold_standard_image_subset)})

    if filter_combination == "expert_goldstandard":
        subject_ids_filter = df_gold_standard_image_subset.subject_id.to_list()
    elif filter_combination == "expert":
        subject_ids_filter = df_goldstandard_expert_count.subject_id.to_list()

    ## flatten, filter and metadata to it
    df_merged_dataset = process_zooniverse_phases_flat(df_zooniverse_flat=df_zooniverse_data,
                                                       image_source=image_source,
                                                       image_names=image_names,
                                                       subject_ids=subject_ids_filter,
                                                       filter_func=filter_remove_marks
                                                       )

    imagename_subject_id_map = df_merged_dataset[["image_name", "subject_id"]].groupby(
        "image_name").first().reset_index(drop=False)
    imagename_subject_id_map.to_csv(output_path.joinpath(f"imagename_subjectid_map_{phase_tag}.csv"))

    df_gold_standard_image_subset = imagename_subject_id_map.merge(df_gold_standard_image_subset, on="subject_id")
    df_gold_standard_and_expert = df_gold_standard_image_subset.merge(df_goldstandard_expert_count, on="subject_id")
    df_gold_standard_and_expert.to_csv(output_path.joinpath(f"{config['gold_standard_and_expert_count']}"))

    df_goldstandard_expert_count[
        df_goldstandard_expert_count.subject_id.isin(df_gold_standard_image_subset.subject_id.to_list())].to_csv(
        config["gold_standard_and_expert_count"])

    logger.info(f"working with {len(df_merged_dataset)} records after process function 'process_zooniverse_phases'")
    df_merged_dataset.to_csv(config["merged_dataset"])
    logger.info(f"saved finished dataset to {config['merged_dataset']}")
    ds_stats.append(
        {"filename": f"{config['merged_dataset'].name}", "images": len(df_merged_dataset.image_name.unique())})

    pd.DataFrame(ds_stats).to_csv(output_path.joinpath(f"ds_stats_{phase_tag}.csv"))

    return pd.DataFrame(ds_stats)


def data_prep_all(phase_tag: str,
                  output_path: Path,
                  config: Optional[dict],
                  input_path,
                  ):
    """
    prepare the zooniverse classifications

    :param filter_combination: either expert_goldstandard or expert
    :param phase_tag:
    :param output_path:
    :return:
    """
    ds_stats = []

    annotations_source = config["annotations_source"]
    zooniverse_annotation_dataset = read_zooniverse_annotations_v2(annotations_source=annotations_source,
                                                                   phase_tags=[phase_tag])
    # the path of the flat dataset
    flatdataset_path = config["flat_dataset"]
    df_zooniverse_data = zooniverse_annotation_dataset["flat_dataset"]
    df_yes_no_dataset = zooniverse_annotation_dataset["yes_no_dataset"]

    df_yes_no_dataset.to_csv(config["yes_no_dataset"])

    # this user is a spammer
    df_zooniverse_data = df_zooniverse_data[df_zooniverse_data.user_id != 2581179]
    df_zooniverse_data.to_csv(flatdataset_path, index=False)
    logger.info(
        f"flat_dataset_Iguanas {phase_tag}.csv: {len(df_zooniverse_data.groupby('image_name').count())} images in classification for {phase_tag}")
    ds_stats.append({"filename": f"{flatdataset_path.name}",
                     "images": len(df_zooniverse_data.groupby('image_name').count())}
                    )

    image_names = None

    output_path.mkdir(exist_ok=True)
    subject_ids_filter = None

    # flatten, filter and metadata to it
    df_merged_dataset = process_zooniverse_phases_flat(df_zooniverse_flat=df_zooniverse_data,
                                                       image_source=None,
                                                       image_names=image_names,
                                                       subject_ids=subject_ids_filter,
                                                       filter_func=filter_remove_marks,
                                                       )

    imagename_subject_id_map = df_merged_dataset[["image_name", "subject_id"]].groupby(
        "image_name").first().reset_index(drop=False)

    # writing to csv
    imagename_subject_id_map.to_csv(output_path.joinpath(f"imagename_subjectid_map_{phase_tag}.csv"))

    logger.info(f"working with {len(df_merged_dataset)} records after process function 'process_zooniverse_phases'")
    df_merged_dataset.to_csv(config["merged_dataset"])
    logger.info(f"saved finished dataset to {config['merged_dataset']}")
    ds_stats.append(
        {"filename": f"{config['merged_dataset'].name}", "images": len(df_merged_dataset.image_name.unique())})

    pd.DataFrame(ds_stats).to_csv(output_path.joinpath(f"ds_stats_{phase_tag}.csv"))

    return pd.DataFrame(ds_stats)


def download_image(url, filename):
    """download an url to a file"""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
            return True
        else:
            logger.warning(f"Failed to download {url}")
            logger.error(response)
            sleep(5)
            return False
    except Exception as e:
        logger.error(e)
        sleep(5)
        return False


def harmonize_field_names(df_subjects: pd.DataFrame) -> pd.DataFrame:
    """
    :param df_subjects:
    :return:
    """
    df_subjects["image_name"] = df_subjects['metadata'].apply(lambda x: json.loads(x).get('Image_name')
                                                                        or json.loads(x).get('image_name')
                                                                        or json.loads(x).get('Filename')).sort_values(
        ascending=True)

    # 'site', 'flight', 'Flight', 'Site', 'flight_code' depict the same
    df_subjects["flight_code"] = df_subjects['metadata'].apply(lambda x: json.loads(x).get('flight_code')
                                                                         or json.loads(x).get('site')
                                                                         or json.loads(x).get('flight')
                                                                         or json.loads(x).get('Flight')
                                                                         or json.loads(x).get('Site')).sort_values(
        ascending=True)

    df_subjects["url"] = df_subjects['locations'].apply(lambda x: json.loads(x)["0"])
    df_subjects["filepath"] = None

    return df_subjects


def anonymise_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    from hashlib import blake2b

    df[column_name] = df[column_name].apply(
        lambda x: blake2b(str(x).encode(), digest_size=16).hexdigest() if not pd.isnull(x) else x)

    return df


def data_prep_panoptes(df_panoptes_point_extractor: pd.DataFrame,
                       df_panoptes_question: pd.DataFrame,
                       df_subjects: pd.DataFrame,
                       config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    scripted version from the notebook step by step instruction
    There is no point to wrap the config creation, data extraction and point_extractor/question extractor files.



    """
    df_panoptes_point_extractor = df_panoptes_point_extractor.merge(df_subjects[["subject_id", "image_name"]],
                                                                    left_on="subject_id", right_on="subject_id")

    df_panoptes_point_extractor = df_panoptes_point_extractor[
        df_panoptes_point_extractor.subject_id.isin(df_subjects.subject_id)]

    # data anonymisation
    df_panoptes_point_extractor = anonymise_column(df_panoptes_point_extractor, "user_id")
    df_panoptes_point_extractor = anonymise_column(df_panoptes_point_extractor, "user_name")
    df_panoptes_question = anonymise_column(df_panoptes_question, "user_id")
    df_panoptes_question = anonymise_column(df_panoptes_question, "user_name")

    # question data summary
    df_panoptes_question_r = df_panoptes_question[df_panoptes_question.task == "T0"][
        ["subject_id", "data.no", "data.yes"]].groupby("subject_id").sum()

    df_panoptes_question_r = df_panoptes_question_r.reset_index()
    df_panoptes_question_r = df_panoptes_question_r[df_panoptes_question_r.subject_id.isin(df_subjects.subject_id)]

    df_panoptes_question_r.to_csv(config["panoptes_question"], index=False)

    # Filter for T2 only
    df_panoptes_point_extractor_r = df_panoptes_point_extractor[
        (df_panoptes_point_extractor.task == "T2")
    ]

    # create a flat structure from the nested marks over multiple columns from that.
    from ast import literal_eval

    columns_keep_x = ['data.frame0.T2_tool0_x', 'data.frame0.T2_tool1_x', 'data.frame0.T2_tool2_x']
    columns_keep_y = ['data.frame0.T2_tool0_y', 'data.frame0.T2_tool1_y', 'data.frame0.T2_tool2_y']

    for col in columns_keep_x + columns_keep_y:
        df_panoptes_point_extractor_r[col] = df_panoptes_point_extractor_r[col].apply(
            lambda x: literal_eval(x) if pd.notnull(x) else [])

    # Merge the lists in 'x' and 'y' coordinates
    df_panoptes_point_extractor_r['x'] = df_panoptes_point_extractor_r[columns_keep_x].values.tolist()
    df_panoptes_point_extractor_r['y'] = df_panoptes_point_extractor_r[columns_keep_y].values.tolist()

    # Flatten the lists in each row for 'x' and 'y'
    df_panoptes_point_extractor_r['x'] = df_panoptes_point_extractor_r['x'].apply(
        lambda x: [item for sublist in x for item in sublist])
    df_panoptes_point_extractor_r['y'] = df_panoptes_point_extractor_r['y'].apply(
        lambda x: [item for sublist in x for item in sublist])

    # Drop the columns that are not needed anymore
    df_panoptes_point_extractor_r = df_panoptes_point_extractor_r[
        ['classification_id', 'user_name', 'user_id', 'workflow_id', 'workflow_version', 'task',
         'created_at', 'subject_id', "image_name",
         'x', 'y'
         ]].reset_index(drop=True)

    # explode the lists of marks per user into one row per mark
    df_panoptes_point_extractor_r_ex = df_panoptes_point_extractor_r.apply(
        lambda x: x.explode() if x.name in ['x', 'y'] else x)

    # images with no marks have NaN values in the 'merged_x' and 'merged_y' columns
    df_panoptes_point_extractor_r_ex_dropped = df_panoptes_point_extractor_r_ex.dropna(subset=['x', 'y'],
                                                                                       how='all').sort_values(
        by=['user_id', 'subject_id', 'task', 'created_at'])

    # cast x and y to int
    df_panoptes_point_extractor_r_ex_dropped = df_panoptes_point_extractor_r_ex_dropped.astype(
        {'x': 'int32', 'y': 'int32'})

    df_panoptes_point_extractor_r_ex_dropped.to_csv(config["flat_panoptes_points"], sep=",", index=False)

    return df_panoptes_question_r, df_panoptes_point_extractor_r_ex_dropped
