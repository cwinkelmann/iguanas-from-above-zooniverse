import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pytest

from zooniverse.config import get_config
from zooniverse.utils.data_format import data_prep_panoptes


@pytest.fixture(scope="module")
def df_panoptes_point_extractor():
    """
    load the data we expect after the extract
    :return:
    """
    return pd.read_csv(Path(__file__).parent.joinpath("panoptes_points_raw.csv"))


@pytest.fixture(scope="module")
def df_panoptes_question():
    """
    load the data we expect after the extract
    :return:
    """
    return pd.read_csv(Path(__file__).parent.joinpath("panoptes_question_raw.csv"))


@pytest.fixture(scope="module")
def df_subjects():
    """
    load the data we expect after the extract
    :return:
    """

    return pd.read_csv(Path(__file__).parent.joinpath("panoptes_subjects_raw.csv"))


def test_dataprep_panoptes(df_panoptes_point_extractor,
                           df_panoptes_question,
                           df_subjects):
    """
    TODO finalise this test case
    run the panoptes data prep
    :return:
    """

    phase_tag = "Iguanas 2nd launch"
    data_folder = "./data/phase_2"

    workflow_id_p1 = 14370.0
    workflow_id_p2 = 20600.0
    workflow_id_p3 = 22040.0

    df_subjects = df_subjects[df_subjects.workflow_id.isin([workflow_id_p1, workflow_id_p2, workflow_id_p3])]
    df_panoptes_point_extractor.drop(columns=["image_name"], inplace=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_path = Path(tmpdirname)
        input_path = Path(__file__).parent

        config = get_config(phase_tag=phase_tag, input_path=input_path, output_path=output_path)

        df_question, df_point = data_prep_panoptes(df_panoptes_point_extractor,
                                                   df_panoptes_question,
                                                   df_subjects,
                                                   config)

        assert df_question.to_dict(orient="records") == [{'data.no': 22.0, 'data.yes': 0.0, 'subject_id': 72332768},
                                                         {'data.no': 21.0, 'data.yes': 0.0, 'subject_id': 72332769},
                                                         {'data.no': 21.0, 'data.yes': 0.0, 'subject_id': 72332770},
                                                         {'data.no': 0.0, 'data.yes': 23.0, 'subject_id': 72336457},
                                                         {'data.no': 0.0, 'data.yes': 22.0, 'subject_id': 72337793},
                                                         {'data.no': 0.0, 'data.yes': 23.0, 'subject_id': 72338137}]

        assert df_point[df_point.classification_id == 410206948].to_dict(orient="records") == [
            {'classification_id': 410206948,
             'created_at': '2022-04-20 16:18:42 UTC',
             'image_name': 'ESCH02-1_369.jpg',
             'subject_id': 72338137,
             'task': 'T2',
             'user_id': 'fa41e1fa6d691c03221c3649d6251e93',
             'user_name': 'b84374aa2e89a0ed7f6973b910a3ff44',
             'workflow_id': 20600,
             'workflow_version': 94.166,
             'x': 424,
             'y': 532},
            {'classification_id': 410206948,
             'created_at': '2022-04-20 16:18:42 UTC',
             'image_name': 'ESCH02-1_369.jpg',
             'subject_id': 72338137,
             'task': 'T2',
             'user_id': 'fa41e1fa6d691c03221c3649d6251e93',
             'user_name': 'b84374aa2e89a0ed7f6973b910a3ff44',
             'workflow_id': 20600,
             'workflow_version': 94.166,
             'x': 559,
             'y': 537},
            {'classification_id': 410206948,
             'created_at': '2022-04-20 16:18:42 UTC',
             'image_name': 'ESCH02-1_369.jpg',
             'subject_id': 72338137,
             'task': 'T2',
             'user_id': 'fa41e1fa6d691c03221c3649d6251e93',
             'user_name': 'b84374aa2e89a0ed7f6973b910a3ff44',
             'workflow_id': 20600,
             'workflow_version': 94.166,
             'x': 638,
             'y': 480},
            {'classification_id': 410206948,
             'created_at': '2022-04-20 16:18:42 UTC',
             'image_name': 'ESCH02-1_369.jpg',
             'subject_id': 72338137,
             'task': 'T2',
             'user_id': 'fa41e1fa6d691c03221c3649d6251e93',
             'user_name': 'b84374aa2e89a0ed7f6973b910a3ff44',
             'workflow_id': 20600,
             'workflow_version': 94.166,
             'x': 375,
             'y': 504},
            {'classification_id': 410206948,
             'created_at': '2022-04-20 16:18:42 UTC',
             'image_name': 'ESCH02-1_369.jpg',
             'subject_id': 72338137,
             'task': 'T2',
             'user_id': 'fa41e1fa6d691c03221c3649d6251e93',
             'user_name': 'b84374aa2e89a0ed7f6973b910a3ff44',
             'workflow_id': 20600,
             'workflow_version': 94.166,
             'x': 295,
             'y': 608},
            {'classification_id': 410206948,
             'created_at': '2022-04-20 16:18:42 UTC',
             'image_name': 'ESCH02-1_369.jpg',
             'subject_id': 72338137,
             'task': 'T2',
             'user_id': 'fa41e1fa6d691c03221c3649d6251e93',
             'user_name': 'b84374aa2e89a0ed7f6973b910a3ff44',
             'workflow_id': 20600,
             'workflow_version': 94.166,
             'x': 645,
             'y': 697}]


if __name__ == '__main__':
    unittest.main()
