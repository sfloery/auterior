from osgeo import ogr, gdal
import numpy as np
import os
from skimage import draw
import time
from joblib import Parallel, delayed
from multiprocessing import Manager
import argparse
import h5py
from scipy.signal import find_peaks
from scipy.spatial import cdist
from scipy.interpolate import splev, splrep
from auterior.transform import world2pixel, pixel2world

def calc_horizon(pnt_gid, pnt_img_coords, pnt_obj_coords, dgm_arr, h_obs, step_deg, min_dist_px, max_dist_px, ds_gt, smooth, res_list):
    
    pnt_img_row = pnt_img_coords[0]
    pnt_img_col = pnt_img_coords[1]
    
    dgm_h, dgm_w = np.shape(dgm_arr)
    
    ix_arr = np.indices((dgm_h, dgm_w))
    ix_row_arr = ix_arr[0]
    ix_col_arr = ix_arr[1]
    
    dist_arr = np.sqrt(np.abs((ix_row_arr - pnt_img_row)*ds_gt[1])**2 + np.abs((ix_col_arr - pnt_img_col)*ds_gt[1])**2)
    dist_arr[pnt_img_row, pnt_img_col] = 1*10**-6

    spher_corr = (1-0.13)*(dist_arr**2/(2*(6371*1000)))
         
    dh_arr = (dgm_arr-spher_corr) - (pnt_obj_coords[2] + h_obs)
    
    dir_arr = np.arctan(dh_arr/dist_arr)
    dir_arr[np.isnan(dgm_arr)] = 0
            
    hor_row_ix = []
    hor_col_ix = []
    
    # step_width = 0.0004
    step_rad = step_deg * np.pi / 180
    
    hor_steps = np.arange(0, 2*np.pi+step_rad/2., step_rad)
    
    for step_rad in hor_steps: #0.0004 equals 10m distance at 25000m distance
        line_row, line_col = draw.line(np.floor(pnt_img_row - min_dist_px*np.cos(step_rad)).astype(np.int32),
                                       np.floor(pnt_img_col + min_dist_px*np.sin(step_rad)).astype(np.int32), 
                                       np.floor(pnt_img_row - max_dist_px*np.cos(step_rad)).astype(np.int32),
                                       np.floor(pnt_img_col + max_dist_px*np.sin(step_rad)).astype(np.int32))
        
        valid_line_row = line_row[(line_row < dgm_h) & (line_col < dgm_w)]
        valid_line_col = line_col[(line_row < dgm_h) & (line_col < dgm_w)]
                
        max_dir_ix = np.argmax(dir_arr[valid_line_row.ravel(), valid_line_col.ravel()].ravel())
        hor_row_ix.append(valid_line_row[max_dir_ix])
        hor_col_ix.append(valid_line_col[max_dir_ix])
    
    hor_dhs = dir_arr[hor_row_ix, hor_col_ix]
    hor_h = dgm_arr[hor_row_ix, hor_col_ix]
    hor_data = np.rad2deg(np.column_stack((hor_steps, hor_dhs)))

    hor_img_coords = np.column_stack((hor_row_ix, hor_col_ix))   
    hor_obj_coords = pixel2world(hor_img_coords, ds_gt)
    
    if smooth == True:
        dist_gid2hor = cdist(pnt_obj_coords, hor_obj_coords).ravel()
        weights_hor = np.full(len(hor_img_coords), fill_value=100)
        weights_hor[dist_gid2hor <= 1000] = 3
        weights_hor[dist_gid2hor <= 500] = 1.5
        weights_hor[dist_gid2hor <= 100] = 1

        spl = splrep(hor_img_coords[:, 0], hor_img_coords[:, 1], k=1, w=weights_hor)
        assert np.isnan(spl[1]).any() == False, "Terrain horizon could not be represented using a spline."
        
        spl_ys = splev(hor_img_coords[:, 0], spl)
        hor_data[:, 1] = spl_ys
        
    hor_prom_pnts = np.zeros((len(hor_data), 1))
    
    peaks, _ = find_peaks(hor_data[:, 1], threshold=0, distance=None, prominence=0, width=0, plateau_size=0)
    valls, _ = find_peaks(hor_data[:, 1]*(-1), threshold=0, distance=None, prominence=0, width=0, plateau_size=0)
    
    hor_prom_pnts[peaks, 0] = 1
    hor_prom_pnts[valls, 0] = -1
    
    hor_data = np.column_stack((hor_data, hor_obj_coords, hor_h, hor_prom_pnts))
    res_list.append({"gid":pnt_gid, 
                     "gid_e":pnt_obj_coords[0], 
                     "gid_n":pnt_obj_coords[1], 
                     "gid_h":pnt_obj_coords[2]+ h_obs, 
                     "gid_hor":hor_data})

    return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-h_obs', type=float, default=2, help='Height of the observer above the terrain.')
    parser.add_argument('-d_max', type=int, default=25000, help="Maximum distance in meters for calculating the horizon.")
    parser.add_argument('-d_min', type=int, default=0, help="Minimum distance in meters for calculating the horizon.")
    parser.add_argument('-dtm_path', type=str, required=True, help="Path to the digital terrain model (DTM).")
    parser.add_argument('-shp_path', type=str, required=True, help="Path to shapefile containing the grid points used for calculating the terrain horizons.")
    parser.add_argument('-hdf_path', type=str, required=True, help="Path to the .hdf5 file used to store the calculated horizons.")
    parser.add_argument('-step_deg', type=float, default=0.1, help="Resolution of the final terrain horizon. Step width in degrees for rotating ray around grid point.")
    parser.add_argument('-n_cpus', type=int, default=1, help="Number of cores used for multiprocessing. -1 for all cores.")
    parser.add_argument('-no_smooth', action="store_true")
    parser.set_defaults(no_smooth=False)
    
    args = parser.parse_args()
    if args.no_smooth == False:
        smooth = True
    else:
        smooth = False
    
    if not os.path.exists(args.dtm_path):
        raise FileNotFoundError(args.dtm_path)
        
    if not os.path.exists(args.shp_path):
        raise FileNotFoundError(args.shp_path)
    
    grid_pnts_ds = ogr.Open(args.shp_path)
    grid_pnts_lyr = grid_pnts_ds.GetLayer(0)
    grid_pnts_lyr_defn = grid_pnts_lyr.GetLayerDefn()

    grid_pnts_gtype = ogr.GeometryTypeToName(grid_pnts_lyr_defn.GetGeomType())
    assert grid_pnts_gtype == "Point", "Geometry type of %s does not match 'Point'." % (args.shp_path)
    
    gid_lyr_ix = grid_pnts_lyr_defn.GetFieldIndex("gid")
    assert gid_lyr_ix >= 0, "gid not a attribute contained in %s." % (args.shp_path)
    
    print("Calculating terrain horizons with %.3f° step width on %i cpus:" % (args.step_deg, args.n_cpus))
    
    grid_pnts = []
    grid_ids = []

    for feature in grid_pnts_lyr:
        feat_geom = feature.GetGeometryRef()
        feat_x = feat_geom.GetX()
        feat_y = feat_geom.GetY()
        grid_pnts.append([feat_x, feat_y])
        grid_ids.append(str(feature.GetField("gid")))
        
    grid_obj_pnts = np.array(grid_pnts).astype(np.float32)
    
    ds = gdal.Open(args.dtm_path)
    ds_gt=ds.GetGeoTransform()
    ds_proj = ds.GetProjection()
    
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    dgm_arr = band.ReadAsArray()
    dgm_arr[dgm_arr == nodata] = np.nan

    dgm_h, dgm_w = np.shape(dgm_arr)
    
    grid_img_pnts = world2pixel(grid_obj_pnts, ds_gt)
    grid_img_pnts_h = dgm_arr[grid_img_pnts[:, 0].ravel(), grid_img_pnts[:, 1].ravel()]
    
    grid_obj_pnts_h = np.hstack((grid_obj_pnts.reshape(-1, 2), grid_img_pnts_h.reshape(-1, 1)))
    
    min_dist_px = args.d_min/ds_gt[1]
    max_dist_px = args.d_max/ds_gt[1]
    
    start_time = time.time()
    process_list= []
    
    manager = Manager()
    output_list = manager.list()
    
    for ix in range(0, len(grid_img_pnts)):
        
        curr_gid = grid_ids[ix]
        gid_img_coords = grid_img_pnts[ix, :]
        gid_obj_coords = grid_obj_pnts_h[ix, :]
                    
        process_list.append([curr_gid,
                             gid_img_coords,
                             gid_obj_coords,
                             dgm_arr, 
                             args.h_obs,
                             args.step_deg,
                             min_dist_px,
                             max_dist_px, 
                             ds_gt,
                             smooth,
                             output_list])
    
    Parallel(n_jobs=args.n_cpus, verbose=11)(delayed(calc_horizon)(*args) for args in process_list)
    end_time = time.time()
    print("Overall time: %.1fs" % (end_time-start_time))
    
    with h5py.File(args.hdf_path, "w") as hdf5_file:
        hdf5_file.attrs["dtm_path"] = args.dtm_path
        hdf5_file.attrs["cols"] = ["L", "B", "E", "N", "H", "is_peak"]
        
        for elem in output_list:
            
            elem_gid = elem["gid"]
            elem_data = elem["gid_hor"].astype(np.float32)
            
            elem_ds = hdf5_file.create_dataset(elem_gid, data=elem_data, dtype=np.float32, compression="gzip")
            elem_ds.attrs["E"] = elem["gid_e"]
            elem_ds.attrs["N"] = elem["gid_n"]
            elem_ds.attrs["H"] = elem["gid_h"].astype(np.float32)