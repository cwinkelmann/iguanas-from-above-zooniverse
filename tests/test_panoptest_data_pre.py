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

    phase_tag = "Iguanas 3rd launch"
    # data_folder = "./data/phase_3"

    workflow_id_p1 = 14370.0
    workflow_id_p2 = 20600.0
    workflow_id_p3 = 22040.0

    input_path = Path("/Users/christian/data/zooniverse")

    # use_gold_standard_subset = "expert" # Use the expert-GS-Xphase as the basis


    df_subjects = pd.read_csv("./data/zooniverse/iguanas-from-above-subjects.csv", sep=",")

    df_subjects = df_subjects[df_subjects.workflow_id.isin([workflow_id_p1, workflow_id_p2, workflow_id_p3])]



    with tempfile.TemporaryDirectory() as tmpdirname:
        output_path = Path(tmpdirname)

        config = get_config(phase_tag=phase_tag, input_path=input_path, output_path=output_path)

        data_prep_panoptes(df_panoptes_point_extractor,
                           df_panoptes_question,
                           df_subjects,
                           config)


if __name__ == '__main__':
    unittest.main()
