"""
Script version of the data preparation for the zooniverse data.
"""
from pathlib import Path

from zooniverse.utils.data_format import data_prep

if __name__ == "__main__":

    phase_tags = ["Iguanas 1st launch", "Iguanas 2nd launch", "Iguanas 3rd launch"]
    # phase_tag = "Iguanas 1st launch"
    phase_tags = ["Iguanas 1st launch"]
    # phase_tag = "Iguanas 3rd launch"

    for phase_tag in phase_tags:
        output_path = Path(f"/Users/christian/data/zooniverse/2024_02_08_analysis").joinpath(phase_tag)
        output_path.mkdir(parents=True, exist_ok=True)
        ds_stats = data_prep(phase_tag, output_path=output_path)


        print(ds_stats)