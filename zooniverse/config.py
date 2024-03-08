from pathlib import Path

# /Users/christian/data/zooniverse

def get_config(phase_tag, input_path, output_path=None):
    configs = {}
    if output_path is None:
        output_path = input / Path("current_analysis").joinpath(phase_tag)

    ## first phase
    configs["Iguanas 1st launch"] = {
        # classifications downloaded from zooniverse
        "annotations_source": input_path.joinpath("IguanasFromAbove/2023-10-15/iguanas-from-above-classifications.csv"),

        # gold standard datatable with the expert count, used for filtering the dataset
        "goldstandard_data": input_path / Path(
            "Images/Zooniverse_Goldstandard_images/expert-GS-1stphase.csv"),

        # which images/subject ids to consider. filters the data.
        "gold_standard_image_subset":
            input_path.joinpath("Images/Zooniverse_Goldstandard_images/1-T2-GS-results-5th-0s.csv"),

        # images for plot on them
        "image_source": input_path.joinpath("Images/Zooniverse_Goldstandard_images/1st launch"),

    }

    ## second phase
    configs["Iguanas 2nd launch"] = {
        # classifications
        "annotations_source": input_path.joinpath("IguanasFromAbove/2023-10-15/iguanas-from-above-classifications.csv"),

        # gold standard datatable with the expert count
        "goldstandard_data": input_path / Path(
            "Images/Zooniverse_Goldstandard_images/expert-GS-2ndphase.csv"),

        # which images/subject ids to consider. filters the data. output from zooniverse
        "gold_standard_image_subset":
            input_path.joinpath("Images/Zooniverse_Goldstandard_images/2-T2-GS-results-5th-0s.csv"),

        # images for plot on them
        "image_source": input_path.joinpath("Images/Zooniverse_Goldstandard_images/2nd launch_without_prefix"),
        # "image_source": None,

    }

    # third phase
    configs["Iguanas 3rd launch"] = {
        # classifications
        "annotations_source": input_path.joinpath("IguanasFromAbove/2023-10-15/iguanas-from-above-classifications.csv"),

        # gold standard datatable
        "goldstandard_data": input_path / Path(
            "Images/Zooniverse_Goldstandard_images/expert-GS-3rdphase_renamed.csv"),

        # which images/subject ids to consider. filters the data.
        "gold_standard_image_subset": input_path.joinpath(
            "Images/Zooniverse_Goldstandard_images/3-T2-GS-results-5th-0s.csv"),

        # images for plot on them
        ## TODO too many images where renamed in there
        # "image_source": input_path.joinpath("Images/Zooniverse_Goldstandard_images/3rd launch")
        "image_source": None,

    }

    # the dataset which reduced tool 1 to iguana yes or iguana no
    configs[phase_tag]["yes_no_dataset"] = output_path.joinpath(
        f"yes_no_dataset_{phase_tag}.csv")

    # the flattened dataset of zooniverse classifications
    configs[phase_tag]["flat_dataset"] = output_path.joinpath(
        f"flat_dataset_{phase_tag}.csv")

    ### filtered datasets
    configs[phase_tag]["merged_dataset"] = output_path.joinpath(
        f"merged_dataset_gold_standard_expert_{phase_tag}_filtered.csv")

    ## intersection gold standard and expert count
    configs[phase_tag]["gold_standard_and_expert_count"] = output_path.joinpath(
        f"{phase_tag}_gold_standard_and_expert_count.csv")

    configs[phase_tag]["gold_standard_and_expert_count"] = output_path.joinpath(
        f"{phase_tag}_gold_standard_and_expert_count.csv")

    ### ===== calculated metrics =====
    ### comparison of the methods per image
    configs[phase_tag]["comparison_dataset"] = output_path.joinpath(f"{phase_tag}_method_comparison.csv")

    ## sums of the methods for each image individually
    configs[phase_tag]["method_sums"] = output_path.joinpath(f"{phase_tag}_method_sums.csv")

    ## rmse of the methods for the full phase
    configs[phase_tag]["rmse_errors"] = output_path.joinpath(f"{phase_tag}_rmse_errors.csv")

    ## dbscan hyperparameter search parameters
    configs[phase_tag]["dbscan_hyperparam_grid"] = output_path.joinpath(f"{phase_tag}_hyperparam_grid.csv")

    return configs[phase_tag]
