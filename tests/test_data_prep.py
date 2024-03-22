import unittest
from pathlib import Path

import pandas as pd

from zooniverse.config import get_config, get_config_all
from zooniverse.utils.data_format import data_prep, data_prep_all


class DataPrepTestCase(unittest.TestCase):

    def test_get_config(self):
        phase_tag = "Iguanas 1st launch"
        output_path = Path(f"/Users/A/B/C/test_analysis").joinpath(phase_tag)

        input_path = Path("/Users/A/B/C")
        config = get_config(phase_tag=phase_tag, input_path=input_path, output_path=output_path)

        self.assertEqual(['annotations_source',
                          'goldstandard_data',
                          'gold_standard_image_subset',
                          'image_source',
                          'yes_no_dataset',
                          'flat_dataset',
                          'merged_dataset',
                          'gold_standard_and_expert_count',
                          'comparison_dataset',
                          'method_sums',
                          'rmse_errors',
                          'dbscan_hyperparam_grid'], list(config.keys()))

        self.assertEqual({'annotations_source': Path(
            '/Users/A/B/C/IguanasFromAbove/2023-10-15/iguanas-from-above-classifications.csv'),
            'comparison_dataset': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/Iguanas 1st launch_method_comparison.csv'),
            'dbscan_hyperparam_grid': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/Iguanas 1st launch_hyperparam_grid.csv'),
            'flat_dataset': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/flat_dataset_Iguanas 1st launch.csv'),
            'gold_standard_and_expert_count': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/Iguanas 1st launch_gold_standard_and_expert_count.csv'),
            'gold_standard_image_subset': Path(
                '/Users/A/B/C/Images/Zooniverse_Goldstandard_images/1-T2-GS-results-5th-0s.csv'),
            'goldstandard_data': Path(
                '/Users/A/B/C/Images/Zooniverse_Goldstandard_images/expert-GS-1stphase.csv'),
            'image_source': Path('/Users/A/B/C/Images/Zooniverse_Goldstandard_images/1st launch'),
            'merged_dataset': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/flat_dataset_filtered_Iguanas 1st launch.csv'),
            'method_sums': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/Iguanas 1st launch_method_sums.csv'),
            'rmse_errors': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/Iguanas 1st launch_rmse_errors.csv'),
            'yes_no_dataset': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/yes_no_dataset_Iguanas 1st launch.csv')}
            , config)

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
