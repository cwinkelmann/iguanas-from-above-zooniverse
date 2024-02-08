# iguanas-from-above-zooniverse
Process to cluster marks set by Volunteers on zooniverse

## Installation
Python 3.8, 3.9, 3.10 are supported. To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

If the install doesn't work, try to install the packages one by one:
```bash
pip install matplotlib jupyterlab pandas scikit-learn loguru black  
```

## Usage
The Notebooks [Zooniverse_Clustering](./Zooniverse_Clustering.ipynb) illustrates the process to cluster the marks set by volunteers on zooniverse.

### Example 1 
<img src="images/FMO03-1_65.jpg_markers.png" alt="Markers" width=500>
<img src="images/FMO03-1_65.jpg_dbscan_0.3_3.png" alt="DBSCAN" width=500>
<img src="images/FMO03-1_65.jpg_hdbscan_bic_n=4.png" alt="HDBSCN" width=500>

### Example 2
<img src="images/FMO03-1_72.jpg_markers.png" alt="Markers" width=500>
<img src="images/FMO03-1_72.jpg_dbscan_0.3_3.png" alt="DBSCAN" width=500>
<img src="images/FMO03-1_72.jpg_hdbscan_bic_n=4.png" alt="HDBSCN" width=500>
