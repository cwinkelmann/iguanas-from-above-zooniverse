import unittest
from pathlib import Path
import pandas as pd

from zooniverse.config import get_config, get_config_all
from zooniverse.utils.data_format import data_prep, data_prep_all

class DataPrepTestCase(unittest.TestCase):
    """
    These data preparation tests are long running and require the data to be present in the right location.
    Those involve the raw data exports from Zooniverse and the expert.
    """
    def test_data_prep(self):
        phase_tag = "Iguanas 1st launch"  # how the phase is named in the zooniverse classifications

        input_path = Path("/Users/christian/data/zooniverse")
        output_path = input_path.joinpath("test_analysis").joinpath(phase_tag)
        output_path.mkdir(parents=True, exist_ok=True)
        config = get_config(phase_tag=phase_tag, input_path=input_path, output_path=output_path)
        ds_stats = data_prep(phase_tag, input_path=input_path, output_path=output_path, config=config)

        print(ds_stats)  # add assertion here
        self.assertEqual(
            [{'filename': 'flat_dataset_Iguanas 1st launch.csv', 'images': 8260},
             {'filename': '1st launch', 'images': 2737},
             {'filename': 'expert-GS-1stphase.csv', 'images': 2733},
             {'filename': '1-T2-GS-results-5th-0s.csv', 'images': 107},
             {'filename': 'flat_dataset_filtered_Iguanas 1st launch.csv',
              'images': 107}],
            ds_stats.to_dict(orient="records"))

        df_merged_dataset = pd.read_csv(config["merged_dataset"])

        self.assertEqual(['Unnamed: 0',
                          'flight_site_code',
                          'workflow_id',
                          'workflow_version',
                          'image_name',
                          'subject_id',
                          'x',
                          'y',
                          'tool_label',
                          'phase_tag',
                          'user_id',
                          'user_name',
                          'mission_name',
                          'image_path',
                          'width',
                          'height'], list(df_merged_dataset.columns), "ensure the columns are the right ones in there")


    def test_data_prep_phase_3_no_filter_all(self):
        """
        Test the data prep without any filtering the images
        :return:
        """
        phase_tag = "Iguanas 3rd launch"  # how the phase is named in the zooniverse classifications

        input_path = Path("/Users/christian/data/zooniverse")
        output_path = input_path.joinpath("test_analysis").joinpath(phase_tag)
        output_path.mkdir(parents=True, exist_ok=True)
        config = get_config_all(phase_tag=phase_tag, input_path=input_path, output_path=output_path)

        ds_stats = data_prep_all(phase_tag=phase_tag,
                             input_path=input_path,
                             output_path=output_path,
                             config=config)

        self.assertEqual(
            [{'filename': 'flat_dataset_Iguanas 3rd launch.csv', 'images': 8330},
             {'filename': 'flat_dataset_filtered_Iguanas 3rd launch.csv',
              'images': 7664}],
            ds_stats.to_dict(orient="records"))


if __name__ == '__main__':
    unittest.main()
