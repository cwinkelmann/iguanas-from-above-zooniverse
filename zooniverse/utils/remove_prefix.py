"""
Code to rename images in case they are not named according to the zooniverse schema
"""

import shutil
from pathlib import Path

from zooniverse.utils.data_format import (
    rename_2023_scheme_images_to_zooniverse,
    rename_from_schema,
)

unedited_images = Path(
    "/Users/christian/data/zooniverse/images/Zooniverse_Goldstandard_images/3rd launch"
)
editet_images = Path(
    "/Users/christian/data/zooniverse/images/Zooniverse_Goldstandard_images/3rd launch_without_prefix"
)

## TODO uncomment this line if you really want to copy them
# shutil.copytree(unedited_images, editet_images)


df_renamed = rename_2023_scheme_images_to_zooniverse(editet_images)

rename_from_schema(df_renamed)
