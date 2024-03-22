import unittest
from pathlib import Path

import pandas as pd
import pytest

from zooniverse.analysis import get_mark_overview, get_annotation_count_stats, HDBSCAN_Wrapper, compare_dbscan_hyp_v2
from zooniverse.utils.filters import filter_df_user_threshold


@pytest.fixture(scope="module")
def expert_subjectids():
    ls_expert_subjectids = [78925728, 78925730, 78925747, 78925781, 78925808, 78925818, 78925861, 78925864, 78925884,
                            78925956, 78925960, 78926032, 78926052, 78926076, 78926089, 78926119, 78926124, 78926144,
                            78926215, 78926225, 78926285, 78926336, 78926344, 78926353, 78926384, 78926410, 78926435,
                            78926444, 78926457, 78926502, 78926594, 78926599, 78926622, 78926692, 78926732, 78926786,
                            78926821, 78926842, 78926873, 78926950, 78926957, 78927033, 78927083, 78927098, 78927107,
                            78927165, 78927179, 78927200, 78927246, 78927260, 78927276, 78927290, 78927306, 78927343,
                            78927351, 78927429, 78927466, 78927498, 78927506, 78927517, 78927519, 78927608, 78927635,
                            78927699, 78927789, 78927817, 78927830, 78927885, 78927939, 78927976, 78928010, 78928030,
                            78928059, 78928064, 78928083, 78928165, 78928198, 78928201, 78928211, 78928263, 78928338,
                            78928410, 78928411, 78928414, 78928432, 78928433, 78928486, 78928499, 78928554, 78928579,
                            78928589, 78928604, 78928642, 78928646, 78928652, 78928657, 78928669, 78928678, 78928689,
                            78928704, 78928708, 78928713, 78928782, 78928809, 78928815, 78928839, 78928843, 78929126,
                            78929160, 78929165, 78929218, 78929294, 78929300, 78929326, 78929380, 78929387, 78929425,
                            78929511, 78929513, 78929520, 78929547, 78929549, 78929572, 78929591, 78929698, 78929715,
                            78929716, 78929720, 78929726, 78929752, 78929766, 78929784, 78929804, 78929808, 78929880,
                            78929902, 78929928, 78929939, 78929961, 78929977, 78929985, 78930028, 78930052, 78930072,
                            78930073, 78930110, 78930120, 78930146, 78930151, 78930155, 78930164, 78930170, 78930173,
                            78930179, 78930206, 78930208, 78930230, 78930238, 78930243, 78930271, 78930273, 78930316,
                            78930317, 78930319, 78930331, 78930353, 78930367, 78930368, 78930382, 78930393, 78930397,
                            78930399, 78930402, 78930405, 78930411, 78930452, 78930461, 78930466, 78930468, 78930469,
                            78930470, 78930480, 78930487, 78930497, 78930544, 78930552, 78930561, 78930616, 78930619,
                            78930627, 78930652, 78930687, 78930695, 78930722, 78930724, 78930740, 78930741, 78930752,
                            78930754, 78930759, 78930762, 78930768, 78930776, 78930787, 78938152, 78938178, 78938181,
                            78938212, 78938221, 78938228, 78938244, 78938256, 78938265, 78938278, 78938280, 78938286,
                            78938296, 78938302, 78938306, 78938316, 78938321, 78938354, 78938358, 78938384, 78938410,
                            78938422, 78938459, 78938460, 78938486, 78938487, 78938495, 78938496, 78938516, 78938518,
                            78938524, 78938542, 78938566, 78938584, 78938603, 78938625, 78938667, 78938704, 78938713,
                            78938721, 78938723, 78938728, 78938731, 78938741, 78938748, 78938776, 78938802, 78938814,
                            78938817, 78938829, 78938831, 78938834, 78938839, 78938855, 78938860, 78938862, 78938865,
                            78938867, 78938907, 78938921, 78938930, 78938931, 78938945, 78938946, 78938957, 78938968,
                            78938974, 78938977, 78938980, 78938986, 78938992, 78939000, 78939002, 78939003, 78939007,
                            78939009, 78939022, 78939023, 78939026, 78939028, 78939033, 78939035, 78939041, 78939045,
                            78939049, 78939072, 78939092, 78939122, 78939125, 78939126, 78939134, 78939137, 78939142,
                            78939160, 78939175, 78939192, 78939212, 78939215, 78939228, 78939235, 78939238, 78939239,
                            78939242, 78939253, 78939260, 78939267, 78939270, 78939277, 78939279, 78939293, 78939294,
                            78939298, 78939322, 78939331, 78939347, 78939351, 78939359, 78939360, 78939364, 78939370,
                            78939371, 78939373, 78939376, 78939391, 78939392, 78939395, 78939416, 78939417, 78939443,
                            78939452, 78939469, 78939472, 78939478, 78939479, 78939483, 78939496, 78939499, 78939500,
                            78939502, 78939505, 78939506, 78939513, 78939516, 78939517, 78939524, 78939532, 78939533,
                            78939535, 78939538, 78939539, 78939540, 78939555, 78939559, 78939562, 78939563, 78939572,
                            78939576, 78939580, 78939581, 78939586, 78939595, 78939611, 78939621, 78939624, 78939644,
                            78939648, 78939654, 78939661, 78939663, 78939664, 78939665, 78939666, 78939678, 78939686,
                            78939690, 78939693, 78939708, 78939711, 78939713, 78939717, 78939718, 78939720, 78939724,
                            78939732, 78939739, 78939747, 78939754, 78939762, 78939765, 78939770, 78939773, 78939784,
                            78939801, 78939805, 78939809, 78939819, 78939820, 78939822, 78939823, 78939825, 78939848,
                            78939852, 78939854, 78939881, 78957268, 78957431, 78957564, 78957587, 78957905, 78957955,
                            78958080, 78958190, 78958235, 78958290, 78958295, 78958310, 78958403, 78958414, 78958440,
                            78958539, 78958713, 78958890, 78958895, 78958896, 78958900, 78958940, 78959008, 78959019,
                            78959032, 78959092, 78959101, 78959142, 78959197, 78959206, 78959356, 78959441, 78959446,
                            78959511, 78959534, 78959749, 78959818, 78959819, 78959832, 78959833, 78959873, 78959878,
                            78959993, 78960071, 78960074, 78960116, 78960182, 78960194, 78960257, 78960438, 78960464,
                            78960523, 78960695, 78960803, 78960904, 78960949, 78960988, 78961017, 78961031, 78961077,
                            78961166, 78961354, 78961444, 78961486, 78961512, 78961586, 78961662, 78961692, 78961703,
                            78961733, 78961744, 78961755, 78961772, 78961824, 78961840, 78961898, 78961901, 78961917,
                            78961924, 78961929, 78961937, 78961953, 78961955, 78961959, 78961969, 78961972, 78961982,
                            78962000, 78962005, 78962009, 78962020, 78962049, 78962114, 78962119, 78962131, 78962146,
                            78962154, 78962161, 78962195, 78962196, 78962206, 78962244, 78962298, 78962338, 78962360,
                            78962367, 78962376, 78962391, 78962396, 78962402, 78962470, 78962491, 78962498, 78962501,
                            78962508, 78962514, 78962536, 78962554, 78962562, 78962582, 78962585, 78962588, 78962599,
                            78962614, 78962623, 78962647, 78962682, 78962722, 78962744, 78962748, 78962751, 78962768,
                            78962770, 78962774, 78962778, 78962799, 78921849, 78921851, 78921852, 78921862, 78921865,
                            78921870, 78921893, 78921969, 78921976, 78921982, 78922003, 78922011, 78922014, 78922024,
                            78922028, 78922029, 78922045, 78922064, 78922071, 78922085, 78922093, 78922110, 78922132,
                            78922139, 78922143, 78922197, 78922202, 78922205, 78922221, 78922224, 78922225, 78922228,
                            78922242, 78922272, 78922276, 78922282, 78922283, 78922331, 78922349, 78922354, 78922359,
                            78922375, 78922378, 78922383, 78922389, 78922398, 78922408, 78922423, 78922430, 78922433,
                            78922457, 78922463, 78922465, 78922466, 78922470, 78922483, 78922502, 78922505, 78922515,
                            78922536, 78922548, 78922558, 78922563, 78922573, 78922613, 78922625, 78922632, 78922645,
                            78930808, 78930848, 78930887, 78930919, 78930976, 78931007, 78931019, 78931042, 78931061,
                            78931126, 78931138, 78931144, 78931190, 78931225, 78931234, 78931250, 78931258, 78931261,
                            78931289, 78931297, 78931372, 78931384, 78931446, 78931469, 78931510, 78931542, 78931604,
                            78931636, 78931651, 78931657, 78931724, 78931783, 78931801, 78931805, 78931854, 78931874,
                            78931990, 78932065, 78932123, 78932195, 78932207, 78932209, 78932213, 78932229, 78932233,
                            78932239, 78932247, 78932305, 78932329, 78932344, 78932361, 78932377, 78932382, 78932391,
                            78932413, 78932418, 78932464, 78932478, 78932512, 78932521, 78932531, 78932552, 78932575,
                            78932601, 78932650, 78932662, 78932671, 78932692, 78963975, 78963984, 78963987, 78963994,
                            78964000, 78964022, 78964023, 78964039, 78964056, 78964061, 78964063, 78964064, 78964076,
                            78964082, 78964083, 78964084, 78964088, 78964112, 78964120, 78964123, 78964139, 78964144,
                            78964172, 78964174, 78964181, 78964183, 78964202, 78964204, 78964206, 78964209, 78964212,
                            78964218, 78964245, 78964255, 78964279, 78964290, 78964291, 78964303, 78964307, 78964309,
                            78964343, 78964347, 78964353, 78964369, 78964380, 78964397, 78964404, 78964428, 78964435,
                            78964437, 78964496, 78964497, 78964518, 78964540, 78964541, 78964555, 78964556, 78964603,
                            78964616, 78964636, 78964644, 78964652, 78964659, 78964670, 78964682, 78964684, 78964692,
                            78964698, 78932769, 78932776, 78932792, 78932797, 78932798, 78932801, 78932803, 78932807,
                            78932812, 78932833, 78932836, 78932851, 78932872, 78932904, 78932926, 78932935, 78932940,
                            78932947, 78932959, 78932972, 78932988, 78933014, 78933026, 78933029, 78933055, 78933096,
                            78933100, 78933108, 78933112, 78933113, 78933134, 78933138, 78933139, 78933168, 78933203,
                            78933253, 78933261, 78933282, 78933301, 78933322, 78933329, 78933380, 78933430, 78933433,
                            78933438, 78933492, 78933563, 78933592, 78933594, 78933618, 78933636, 78933661, 78933685,
                            78933697, 78933709, 78933713, 78933731, 78933747, 78933757, 78933758, 78933760, 78933803,
                            78933839, 78933847, 78933849, 78933854, 78933855, 78933867, 78964714, 78964720, 78964722,
                            78964723, 78964724, 78964735, 78964739, 78964742, 78964753, 78964762, 78964769, 78964776,
                            78964779, 78964785, 78964808, 78964822, 78964844, 78964846, 78964848, 78964850, 78964856,
                            78964887, 78964894, 78964896, 78964898, 78964900, 78964906, 78964907, 78964914, 78964922,
                            78964924, 78964927, 78964932, 78964935, 78964938, 78964945, 78964952, 78964958, 78964972,
                            78964976, 78964979, 78964986, 78965007, 78965008, 78965013, 78965017, 78965019, 78965032,
                            78965045, 78965047, 78965058, 78965065, 78965066, 78965074, 78965076, 78965086, 78965103,
                            78965108, 78965112, 78965129, 78965131, 78965132, 78965135, 78965143, 78965147, 78965156,
                            78965171, 78965172, 78934070, 78934130, 78934231, 78934253, 78934369, 78934398, 78934399,
                            78934409, 78934452, 78934453, 78934557, 78934568, 78934595, 78934659, 78934756, 78934784,
                            78934808, 78934872, 78934942, 78934998, 78935056, 78935066, 78935114, 78935117, 78935144,
                            78935439, 78935461, 78935591, 78935616, 78935644, 78935657, 78935743, 78935850, 78935998,
                            78936134, 78936157, 78936243, 78936280, 78936348, 78936371, 78936394, 78936462, 78936537,
                            78936566, 78936580, 78936707, 78936742, 78936795, 78936854, 78936942, 78937019, 78937081,
                            78937090, 78937093, 78937126, 78937127, 78937161, 78937366, 78937385, 78937524, 78937565,
                            78937737, 78937778, 78937793, 78937831, 78937854, 78937915, 78938011, 78962811, 78962814,
                            78962820, 78962840, 78962844, 78962845, 78962847, 78962854, 78962863, 78962871, 78962875,
                            78962886, 78962887, 78962894, 78962901, 78962908, 78962912, 78962921, 78962926, 78962936,
                            78962951, 78962952, 78962963, 78962964, 78962978, 78962985, 78962987, 78962998, 78963000,
                            78963015, 78963029, 78963042, 78963083, 78963093, 78963106, 78963107, 78963137, 78963140,
                            78963146, 78963152, 78963157, 78963158, 78963164, 78963186, 78963189, 78963202, 78963214,
                            78963231, 78963265, 78963271, 78963279, 78963297, 78963316, 78963321, 78963323, 78963383,
                            78963391, 78963394, 78963396, 78963403, 78963409, 78963443, 78963447, 78963459, 78963463,
                            78963467, 78963481, 78963494, 78923839, 78923842, 78923863, 78923891, 78923919, 78923928,
                            78923930, 78923977, 78923984, 78923987, 78923996, 78924003, 78924026, 78924066, 78924089,
                            78924093, 78924107, 78924108, 78924170, 78924172, 78924183, 78924218, 78924227, 78924241,
                            78924266, 78924280, 78924283, 78924310, 78924324, 78924345, 78924363, 78924365, 78924378,
                            78924381, 78924402, 78924435, 78924440, 78924465, 78924472, 78924516, 78924578, 78924579,
                            78924589, 78924592, 78924612, 78924639, 78924658, 78924677, 78924679, 78924704, 78924734,
                            78924743, 78924752, 78924753, 78924779, 78924823, 78924836, 78924856, 78924860, 78924865,
                            78924875, 78924884, 78924890, 78924908, 78924943, 78924961, 78924965, 78924967, 78924973,
                            78924991, 78925000, 78925007, 78925008, 78925017, 78925028, 78925075, 78925082, 78925101,
                            78925109, 78925118, 78925137, 78925145, 78925166, 78925167, 78925173, 78925179, 78925202,
                            78925206, 78925212, 78925224, 78925230, 78925231, 78925237, 78925243, 78925246, 78925257,
                            78925267, 78925269, 78925282, 78925294, 78925299, 78925316, 78925318, 78925329, 78925330,
                            78925339, 78925358, 78925364, 78925366, 78925375, 78925384, 78925388, 78925402, 78925415,
                            78925435, 78925457, 78925467, 78925471, 78925475, 78925499, 78925508, 78925513, 78925526,
                            78925531, 78925536, 78925548, 78925551, 78925572, 78925576, 78925591, 78925597, 78925600,
                            78925604, 78925605, 78925608, 78925614]
    return ls_expert_subjectids


@pytest.fixture(scope="module")
def expert_5th():
    ls_experth_th = [78922029, 78922093, 78922433, 78922625, 78924089, 78924592, 78924658, 78924875, 78925243, 78925366,
                     78925388, 78925457, 78925467, 78925536, 78925551, 78926344, 78928708, 78932361, 78937831, 78938221,
                     78938603, 78938992, 78939007, 78957268, 78961354, 78961512, 78961972, 78962844, 78962845, 78962998,
                     78963042, 78963297, 78963984, 78964000, 78964022, 78964023, 78964039, 78964056, 78964061, 78964076,
                     78964120, 78964123, 78964144, 78964245, 78964279, 78964291, 78964343, 78964347, 78964353, 78964369,
                     78964404, 78964437, 78964616, 78964652, 78964659, 78964670, 78964684, 78964714, 78964720, 78964722,
                     78964723, 78964724, 78964735, 78964739, 78964769, 78964779, 78964785, 78964822, 78964844, 78964848,
                     78964856, 78964906, 78964907, 78964924, 78964952, 78964958, 78964972, 78965007, 78965008, 78965013,
                     78965017, 78965019, 78965032, 78965058, 78965066, 78965103, 78965135]
    return ls_experth_th


@pytest.fixture(scope="module")
def df_merged_dataset():
    """Fixture to provide sample data for testing."""
    df_merged_dataset = pd.read_csv(Path(__file__).parent.resolve() / "flat_dataset_filtered_Iguanas 3rd launch_small.csv")

    df_merged_dataset = filter_df_user_threshold(df_merged_dataset, user_threshold=3)

    from zooniverse.utils.filters import filter_remove_marks
    # Check if partials are still in the data. There shouldn't be any
    df_merged_dataset = filter_remove_marks(df_merged_dataset)

    return pd.DataFrame(df_merged_dataset)


def test_basic_statistics(df_merged_dataset):
    basic_stats = []
    for image_name, df_image_name in df_merged_dataset.groupby("image_name"):
        annotations_count = get_mark_overview(df_image_name)

        annotations_count_stats = get_annotation_count_stats(annotations_count=annotations_count,
                                                             image_name=df_image_name.iloc[0]["image_name"])

        ### basic statistics like mean, median
        basic_stats.append(annotations_count_stats)

    df_basic_stats = pd.DataFrame(basic_stats)
    assert df_basic_stats["median_count"].sum() == 314.5, "314.5 median count of iguanas"


def test_basic_statistics_expert(df_merged_dataset, expert_subjectids):
    basic_stats = []

    df_merged_dataset = df_merged_dataset[df_merged_dataset["subject_id"].isin(expert_subjectids)]

    for image_name, df_image_name in df_merged_dataset.groupby("image_name"):
        annotations_count = get_mark_overview(df_image_name)

        annotations_count_stats = get_annotation_count_stats(annotations_count=annotations_count,
                                                             image_name=df_image_name.iloc[0]["image_name"])

        ### basic statistics like mean, median
        basic_stats.append(annotations_count_stats)

    df_basic_stats = pd.DataFrame(basic_stats)
    assert df_basic_stats["median_count"].sum() == 314.5


def test_db_scan_expert_5th(df_merged_dataset, expert_5th):
    df_merged_dataset = df_merged_dataset[df_merged_dataset["subject_id"].isin(expert_5th)]

    eps_variants = [0.3]
    min_samples_variants = [3]
    params = [(eps, min_samples) for eps in eps_variants for min_samples in min_samples_variants]

    db_scan_results = {}
    db_scan_best_results = []
    db_scan_best_bic_results = []
    for image_name, df_image_name in df_merged_dataset.groupby("image_name"):

        dbscan_localization = compare_dbscan_hyp_v2(
            params=params,
            df_flat=df_image_name,
            output_plot_path=None,
            plot=None,
        )

        db_scan_results[image_name] = pd.DataFrame(dbscan_localization)

        # DBSCAN tends to classfy all points as noise if min_samples is too high. Often only a single user marked an iguana.
        # Sillouette Scoring needs a minimum of 2 clusters
        # if there are points in decent radius they will belong to a cluster
        if pd.DataFrame(dbscan_localization).dbscan_count.max() == 1:
            db_scan_best_results.append(
                pd.DataFrame(dbscan_localization).sort_values("dbscan_count", ascending=False).iloc[0])
            db_scan_best_bic_results.append(
                pd.DataFrame(dbscan_localization).sort_values("dbscan_count", ascending=False).iloc[0])
            # If two or more cluster seem to exists take ones with the best Silouette score
        else:
            # take the best result by silouette score if there are more clusters then 1
            db_scan_best_results.append(
                pd.DataFrame(dbscan_localization).sort_values(["dbscan_silouette_score", "dbscan_count"],
                                                              ascending=[False, False]).iloc[0])

    df_dbscan_localization = pd.concat([*db_scan_results.values()])
    df_scan_best_results = pd.DataFrame(db_scan_best_results)

    assert df_scan_best_results.dbscan_count.sum() == 282, "282 iguanas according to DBSCAN"
    assert df_scan_best_results.shape[0] == 86, "86 images had enough marks"


def test_hdbscan_clustering_filtered(df_merged_dataset, expert_subjectids):
    """test if the dbscan clustering works as expected with subject_id and image_name"""
    hdbscan_values = []
    df_merged_dataset = df_merged_dataset[df_merged_dataset["subject_id"].isin(expert_subjectids)]
    for image_name, df_image_name in df_merged_dataset.groupby("image_name"):
        # if less than min_cluster_sizes points are available clustering makes no sense
        if df_image_name.shape[0] >= 5:  # If num_samples is 5 for the min_cluster_size is 5
            # there is no point in passing data with less than 5 samples

            df_hdbscan = HDBSCAN_Wrapper(df_marks=df_image_name[["x", "y"]],
                                         output_path=None,
                                         image_name=image_name,
                                         params=[(0.0, 5, None)])
            hdbscan_values.append(df_hdbscan)

    df_hdbscan = pd.concat(hdbscan_values)
    assert df_hdbscan.shape[0] == 86, "86 images!"
    assert df_hdbscan["HDBSCAN_count"].sum() == 358, "358 iguanas according to HDBSCAN"


