# auterior
**AU**tomatic **TER**restrial **I**mage **OR**ienation based on the visible horizon in images. 

## Installation

## Usage 
First we need to calculate the terrain horizons
```
calc_terhor.py
  -h
  -h_obs        Height of the observer above the terrain.
  -d_max        Maximum distance in meters for calculating the horizon.
  -d_min        Minimum distance in meters for calculating the horizon.
  -dtm_path     Path to the digital terrain model (DTM).
  -shp_path     Path to shapefile containing the grid points used for calculating the terrain horizons.
  -hdf_path     Path to the .hdf5 file used to store the calculated horizons.
  -step_deg     Resolution of the final terrain horizon. Step width in degrees for rotating ray around grid point.
  -n_cpus       Number of cores used for multiprocessing. -1 for all cores.
```
and extract smaller terrain horizon parts of predefined width represented as IMTF features
```
desc_terhor.py
  -hdf_path     Path to the *.hdf5 file containing the calculated terrain horizons created using calc_terhor.py
  -out_path     File path used to store the terrain horizon parts.
  -feat         Feature descriptor used to describe the terrain horizon parts. Currently only 'imtf' is supported.
  -w            Width in ° of the extracted horizon parts. Example: -w 10 20 or -w 10
  -s            Step width between horizon parts in °.
  -n            Number of sampling points for each horizon part.
```
After that its possible to estimate the coarse orientation based on a provided image horizon using
```
coarse.py 
  -hdf_path     Path to the *.hdf5 file containing the terrain horizon features created using desc_terhor.py
  -hor_path     Path to the image containing the horizon. If a directory is given all *.tif are processed.
  -hfov         HFOV in ° of the image. If unknown provide range of values which will be used e.g. -hfov 35, 40, 45
  -knn          Number of neighbors returned in the matching.
  -dir_bin      Bin size used for direction counting.
  -topn         Topn candidates kept for each HFOV which will be further used in the spatial resection.
```
and calculate the spatial resection with
```
spatres.py 
  -coarse_path    Path to *.hdf5 containing the results of the coarse orientation created using coarse.py
  -ransac_thresh  Threshold in px to consider points as inliers.
  -dx_lim         Limit in ° speciyfing width for considering potential pairs.
```
The result of spatres.py is a csv with the following attributes:
| Attribute  | Description |
| ------------- | ------------- |
| HFOV  | Used HFOV [°]|
| CNT_RANK  | Rank using coarse CNT|
| NCNT_RANK  | Rank using normalized coarse CNT|
| P3P_RANK  | Rank using P3P_INCNT|
| GID  | Grid Point ID|
| GID_E  | GID East [m] |
| GID_N  | GID North [m] |
| GID_H  | GID Height [m]  |
| GID_AZIMUT  | Azimut [rad] clockwise from north|
| P3P_E  | East [m] using spatial resection |
| P3P_N  | North [m] using spatial resection |
| P3P_H  | Height [m] using spatial resection |
| P3P_H_DGM  | Height [m] above terrain. |
| P3P_ALPHA  | Alpha [rad] counted counter-clockwise from east. |
| P3P_ALPHA_NORTH  | Alpha [rad] counted clockwise from north. |
| P3P_ZETA  | Zeta [rad]  |
| P3P_KAPPA  | Kappa [rad]  |
| P3P_INCNT  | Ransac Inlier Count  |
| P3P_ERR  | Reprojection erroro [px]  |
| MM_IMG  | Image coordinates (row,col) of the inliers |
| MM_OBJ  | Object coordinates (E,N,H) of the inliers |

## Cite

## ToDo
- [ ] Demo dataset
- [ ] Auxillary plotting functions
- [ ] Installation details
