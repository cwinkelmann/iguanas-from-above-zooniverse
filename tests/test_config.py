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
                          'flat_panoptes_points',
                          'panoptes_question',
                          'merged_dataset',
                          'gold_standard_and_expert_count',
                          'comparison_dataset',
                          'comparison_dataset_expert',
                          'comparison_dataset_yes_no',
                          'method_sums',
                          'rmse_errors',
                          'dbscan_hyperparam_grid'], list(config.keys()))

        self.assertEqual({'annotations_source': Path(
            '/Users/A/B/C/IguanasFromAbove/2023-10-15/iguanas-from-above-classifications.csv'),
            'comparison_dataset': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/Iguanas 1st launch_method_comparison.csv'),
            'comparison_dataset_expert': Path('/Users/A/B/C/test_analysis/Iguanas 1st launch/Iguanas 1st launch_method_comparison_expert.csv'),
            'comparison_dataset_yes_no': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/Iguanas 1st launch_method_comparison_yes_no.csv'),
            'dbscan_hyperparam_grid': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/Iguanas 1st launch_hyperparam_grid.csv'),
            'flat_dataset': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/flat_dataset_Iguanas 1st launch.csv'),
            'flat_panoptes_points': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/flat_panoptes_points_Iguanas 1st launch.csv'),

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
            'panoptes_question': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/panoptes_question_Iguanas 1st launch.csv'),
            'rmse_errors': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/Iguanas 1st launch_rmse_errors.csv'),
            'yes_no_dataset': Path(
                '/Users/A/B/C/test_analysis/Iguanas 1st launch/yes_no_dataset_Iguanas 1st launch.csv')}
            , config)



if __name__ == '__main__':
    unittest.main()
