"""
Script version of the data preparation for the zooniverse data.
"""
from pathlib import Path

from zooniverse.config import get_config
from zooniverse.utils.data_format import data_prep

if __name__ == "__main__":

    phase_tags = ["Iguanas 1st launch", "Iguanas 2nd launch", "Iguanas 3rd launch"]

    for phase_tag in phase_tags:
        output_path = Path(f"/Users/christian/data/zooniverse/2024_04_08_analysis").joinpath(phase_tag)
        output_path.mkdir(parents=True, exist_ok=True)
        input_path = Path("/Users/christian/data/zooniverse")
        config = get_config(phase_tag=phase_tag, input_path=input_path, output_path=output_path)
        ds_stats = data_prep(phase_tag, input_path=input_path, output_path=output_path, config=config)


        print(ds_stats)