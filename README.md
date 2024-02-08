# iguanas-from-above-zooniverse
Process to cluster marks set by Volunteers on zooniverse

## Installation
Python 3.8, 3.9, 3.10 are tested. To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

If the install doesn't work, try to install the packages one by one:
```bash
pip install matplotlib jupyterlab pandas scikit-learn loguru black  
```

## Usage
The Notebooks [Zooniverse_Clustering](./Zooniverse_Clustering.ipynb) illustrates the process to cluster the marks set by volunteers on zooniverse. It results in csv files with the clustering results and images with the marks and the clusters. The method_comparison.csv file contains the comparison between the clustering methods per image.

| image_name      | subject_id | count_total | median_count | mean_count | mode_count | users | sum_annotations_count | annotations_count                            | dbscan_count_sil | HDBSCAN_count |
|-----------------|------------|-------------|--------------|------------|------------|-------|-----------------------|----------------------------------------------|------------------|---------------|
| EGI08-2_78.jpg  | 72333835   | 1           | 1.0          | 1.00       | 1          | 12    | 12                    | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]         | 1                | 1             |
| FMO03-1_65.jpg  | 72338628   | 5           | 4.0          | 3.42       | 4          | 19    | 65                    | [1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, ... | 4                | 4             |
| FMO03-1_72.jpg  | 72338635   | 4           | 3.0          | 2.65       | 4          | 20    | 53                    | [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, ... | 3                | 4             |


### Example 1 
<img src="images/FMO03-1_65.jpg_markers.png" alt="Markers" width=500>
<img src="images/FMO03-1_65.jpg_dbscan_0.3_3.png" alt="DBSCAN" width=500>
<img src="images/FMO03-1_65.jpg_hdbscan_bic_n=4.png" alt="HDBSCN" width=500>

### Example 2
<img src="images/FMO03-1_72.jpg_markers.png" alt="Markers" width=500>
<img src="images/FMO03-1_72.jpg_dbscan_0.3_3.png" alt="DBSCAN" width=500>
<img src="images/FMO03-1_72.jpg_hdbscan_bic_n=4.png" alt="HDBSCN" width=500>
