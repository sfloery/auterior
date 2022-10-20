import argparse
import cv2
import h5py
import pandas as pd
from skimage import measure
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
import numpy as np
from scipy.interpolate import splrep
from auterior.features import imtf4points
from auterior.transform import rot2alzeka, rot_cv2ori, select_alzeka, world2pixel
import os
from osgeo import gdal

def load_img(img_path, nr_bands=3, band_dtype=np.uint8):

    ds = gdal.Open(img_path)
    
    ds_w = ds.RasterXSize
    ds_h = ds.RasterYSize
    
    ds_gt = ds.GetGeoTransform()
    
    band_arr = np.zeros((ds_h, ds_w, nr_bands), dtype=band_dtype)
    
    for i in range(nr_bands):
        curr_band = ds.GetRasterBand(i+1)
        curr_arr = curr_band.ReadAsArray().astype(band_dtype)
        band_arr[:, :, i] = curr_arr
    
    return band_arr.squeeze(), ds_gt, ds_h, ds_w

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument('-coarse_path', type=str, required=True, help="Path to *.hdf5 containing the results of the coarse orientation.")
    parser.add_argument('-ransac_thresh', type=float, default=10, required=False, help="Threshold in px to consider points as inliers.")
    parser.add_argument('-dx_lim', type=float, default=1, required=False, help="Limit in degrees speciyfing width for considering potential pairs.")
    
    args = parser.parse_args()
    
    coarse_hdf_paths = []
    if os.path.isdir(args.coarse_path):
        for file in os.listdir(args.coarse_path):
            if file.endswith(".hdf5"):
                coarse_hdf_paths.append(os.path.join(args.coarse_path, file))
    else:
        coarse_hdf_paths.append(args.coarse_path)
    
    up = cv2.UsacParams()
    up.sampler = cv2.SAMPLING_UNIFORM
    up.score = cv2.SCORE_METHOD_MSAC
    up.maxIterations = 10000
    up.threshold = args.ransac_thresh
    up.loMethod = cv2.LOCAL_OPTIM_GC
    
    

    dtm_path = None
    
    for coarse_path in coarse_hdf_paths:
        
        if not os.path.exists(coarse_path):
            continue
        
        print("Calculating spatial resection for %s:" % (coarse_path))
        
        with h5py.File(coarse_path, mode="r") as coarse_file:
            
            if coarse_file.attrs["dtm_path"] != dtm_path:
                dtm_path = coarse_file.attrs["dtm_path"]
                curr_dgm, curr_gt, curr_dgm_h, curr_dgm_w = load_img(dtm_path, nr_bands=1, band_dtype=np.float32)
                
            topn_data = coarse_file["coarse_topn"][()]
            topn_data_cols = coarse_file["coarse_topn"].attrs["cols"].split(";")

            hor_img_w = int(coarse_file["hor_geom"].attrs["img_w"])
            hor_img_h = int(coarse_file["hor_geom"].attrs["img_h"])
            hor_geom = coarse_file["hor_geom"][()]
            
            terhor_hdf_path = coarse_file.attrs["terhor_path"]
            
        pd_topn = pd.DataFrame(topn_data, columns=topn_data_cols)
        pd_topn[["GID", "CNT", "RANK", "HFOV"]] = pd_topn[["GID", "CNT", "RANK", "HFOV"]].astype(int)
            
        hor_geom = measure.approximate_polygon(hor_geom, 1)
        hor_row = hor_img_h - hor_geom[:, 0]
        hor_col = hor_geom[:, 1]
        
        peaks_ix, _ = find_peaks(hor_row, threshold=0, distance=None, prominence=0, width=0, plateau_size=0)
        img_peaks_px = hor_geom[peaks_ix, :]
        nr_img_peaks = len(img_peaks_px)
        
        terhor_hdf = h5py.File(terhor_hdf_path, "r")
        cols = list(terhor_hdf.attrs["cols"])
        
        topn_data = pd_topn.to_dict(orient="records")
        
        for curr_data in topn_data:
            
            curr_data.update({"P3P_E":None, 
                            "P3P_N":None, 
                            "P3P_H":None, 
                            "P3P_ALPHA":None, 
                            "P3P_ZETA":None, 
                            "P3P_KAPPA":None, 
                            "P3P_F":None,
                            "P3P_INCNT":None,
                            "P3P_ERR":None,
                            "MM_IMG":None,
                            "MM_OBJ":None
                            })
            
            curr_gid = str(curr_data["GID"])
            curr_ori = curr_data["GID_ORI"]
            
            curr_hfov_deg = float(curr_data["HFOV"])
            curr_vfov_deg = (hor_img_h / hor_img_w)*curr_hfov_deg
            curr_f = (hor_img_w/2.)/np.tan(np.deg2rad(curr_hfov_deg)/2.)
            
            curr_data["P3P_F"] = curr_f
            
            #load image horizon and rescale to degrees
            img_hor_col_deg = np.interp(hor_col, (0, hor_img_w), (curr_ori-curr_hfov_deg/2., curr_ori+curr_hfov_deg/2.))
            img_hor_row_deg = np.interp(hor_row, (0, hor_img_h), (0, curr_vfov_deg))
            img_hor_row_deg -= curr_vfov_deg/2.
            
            #extract image peaks
            img_peaks_col = img_hor_col_deg[peaks_ix]
            img_peaks_row = img_hor_row_deg[peaks_ix]
            img_peaks = np.column_stack((img_peaks_col, img_peaks_row))
            
            #prepare image horizon for splines
            img_hor_col, uq_cols_ix = np.unique(img_hor_col_deg, return_index=True)
            img_hor_row = img_hor_row_deg[uq_cols_ix]
            
            #load reference horizon
            ref_hor_data = terhor_hdf[curr_gid][()]
            pd_ref = pd.DataFrame(ref_hor_data, columns=cols)
            
            #if img horizon extens below 0 or 360: add parts from the other end
            if np.min(img_hor_col_deg) < 0:
                pd_ref_lo = pd_ref[pd_ref["L"] >= np.min(img_hor_col_deg)+360].copy()
                pd_ref_lo["L"] = pd_ref_lo["L"] - 360
                pd_ref = pd.concat((pd_ref_lo, pd_ref), axis=0)

            if np.max(img_hor_col_deg) > 360:
                pd_ref_hi = pd_ref[pd_ref["L"] <= np.max(img_hor_col_deg)-360].copy()
                pd_ref_hi["L"] = pd_ref_hi["L"] + 360
                pd_ref = pd.concat((pd_ref, pd_ref_hi), axis=0)
            
            pd_ref = pd_ref[pd_ref['L'].between(np.min(img_hor_col_deg), np.max(img_hor_col_deg))]
            pd_ref.sort_values(by=["L"], inplace=True)
            
            #extract reference horizon part correspondong to actual view
            # ref_peaks_ix = np.nonzero(pd_ref["is_peak"].values.astype(int))[0]
            ref_peaks_ix =np.nonzero((pd_ref["is_peak"].astype(int) == 1).values)[0]
            nr_ref_peaks = np.shape(ref_peaks_ix)[0]
            
            #4 peaks are at least necessary to calcualte the spatial resection
            if nr_ref_peaks < 4:
                continue
            
            ref_hor = pd_ref[["L", "B"]].values
            ref_hor_col = ref_hor[:, 0]
            ref_hor_row = ref_hor[:, 1]
            
            #reference peaks in "image" space
            ref_peaks = ref_hor[ref_peaks_ix.ravel(), :]
            ref_peaks_col = ref_peaks[:, 0]
            ref_peaks_row = ref_peaks[:, 1]
            
            #reference peaks in object space
            ref_hor_obj = pd_ref[["E", "N", "H"]].values 
            ref_peaks_obj = ref_hor_obj[ref_peaks_ix.ravel(), :]

            #fit splines to both horizons
            ref_spl = splrep(ref_hor_col, ref_hor_row, k=1) 
            assert np.isnan(ref_spl[1]).any() == False, "Terrain horizon could not be represented using a spline."
            
            img_spl = splrep(img_hor_col, img_hor_row, k=1)
            assert np.isnan(img_spl[1]).any() == False, "Image horizon could not be represented using a spline."
            
            #extract feature descriptors for all points    
            img_feats, spl_img_x = imtf4points(img_spl, w_deg=1, nr_samples=32, points=img_peaks, normalize=True, mode="dist")
            ref_feats, spl_ref_x = imtf4points(ref_spl, w_deg=1, nr_samples=32, points=ref_peaks, normalize=True, mode="dist")
            
            #calculate difference in x coordinates between all image and reference peaks; used to spatially constrain the correspondence search
            dx_img2ref = np.abs(cdist(img_peaks[:, 0].reshape(-1, 1), ref_peaks[:, 0].reshape(-1, 1), metric="minkowski"))
            dx_bool = np.ones_like(dx_img2ref, dtype=np.uint8)
            
            #calculate actual distance between all image and reference features in feature space
            cost_matrix = cdist(img_feats, ref_feats, metric="euclidean")
            
            #change cost infinite high that those points are never considered which are more than 1 degree away
            cost_matrix[dx_img2ref >= args.dx_lim] = 1*10**6
            dx_bool[dx_img2ref >= args.dx_lim] = 0
            
            #matches img2ref
            sort_ix = np.argsort(cost_matrix, axis=1)
            sc_topn = sort_ix[:, :1]
            sc_topn_rix = np.repeat(np.arange(0, nr_img_peaks).reshape(-1, 1), 1, axis=1)
            matches = np.column_stack((sc_topn_rix.reshape(-1, 1), sc_topn.reshape(-1, 1)))
            matches = matches[np.argwhere(dx_bool[matches[:, 0].ravel(), matches[:,1].ravel()] == 1).ravel(), :]
            
            #inverse matches ref2img
            dx_ref2img = np.abs(cdist(ref_peaks[:, 0].reshape(-1, 1), img_peaks[:, 0].reshape(-1, 1), metric="minkowski"))
            dx_bool = np.ones_like(dx_ref2img, dtype=np.uint8)
            cost_matrix = cdist(ref_feats, img_feats, metric="euclidean")
            
            cost_matrix[dx_ref2img >= args.dx_lim] = 1*10**6
            dx_bool[dx_ref2img >= args.dx_lim] = 0
            
            sort_ix = np.argsort(cost_matrix, axis=1)
            sc_topn = sort_ix[:, :1]
            sc_topn_rix = np.repeat(np.arange(0, nr_ref_peaks).reshape(-1, 1), 1, axis=1)
            matches_inv = np.column_stack((sc_topn_rix.reshape(-1, 1), sc_topn.reshape(-1, 1)))
            matches_inv = matches_inv[np.argwhere(dx_bool[matches_inv[:, 0].ravel(), matches_inv[:,1].ravel()] == 1).ravel(), :]
            
            matches_inv = matches_inv[:, [1, 0]]
            mutual_matches = np.array([x for x in set(tuple(x) for x in matches) & set(tuple(x) for x in matches_inv)])
            
            if len(mutual_matches) < 4:
                continue
        
            #extract points based on the mutual matches
            all_img_pnts = img_peaks_px[:, [1, 0]]
            img_pnts = all_img_pnts[mutual_matches[:, 0].ravel(), :].reshape(1, -1, 2).astype(np.float64)
            obj_pnts = ref_peaks_obj[mutual_matches[:, 1].ravel(), :].reshape(1, -1, 3).astype(np.float64)
            
                            
            cam_mat = np.array([[curr_f, 0, hor_img_w/2.], 
                                [0, curr_f, hor_img_h/2.], 
                                [0, 0, 1]], dtype="double")

            retval, _, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=obj_pnts,
                                                                imagePoints=img_pnts, 
                                                                cameraMatrix=cam_mat,
                                                                distCoeffs=None,
                                                                params=up)
            
            if retval:
                inliers = inliers.ravel()
                nr_inliers = len(inliers)
                
                if nr_inliers == 4:
                    continue
                
                #reproject points from object space into image space to calculate projection error
                inliers_proj, _ = cv2.projectPoints(obj_pnts[:, inliers, :], rvec, tvec, cam_mat, None)
                dist = np.linalg.norm(inliers_proj.squeeze() - img_pnts[:, inliers, :].squeeze(), axis=1)
                proj_error = np.sum(dist) / nr_inliers
                                                
                rmat, _ = cv2.Rodrigues(rvec)
                rmat_ori = rot_cv2ori(rmat)
                est_prc = np.matmul(np.transpose(rmat) * (-1) , tvec).ravel()
                
                est_euler = rot2alzeka(rmat_ori)
                est_euler = select_alzeka(alzekas=est_euler, prc=est_prc, gcps=obj_pnts[:, inliers, :].squeeze())
                                
                mm_ix = mutual_matches[inliers, :]
                mm_img = all_img_pnts[mm_ix[:, 0].ravel(), :]
                mm_obj = ref_peaks_obj[mm_ix[:, 1].ravel(), :]
                
                curr_data["P3P_E"] = est_prc[0]
                curr_data["P3P_N"] = est_prc[1]
                curr_data["P3P_H"] = est_prc[2]
                curr_data["P3P_ALPHA"] = est_euler[0]
                curr_data["P3P_ZETA"] = est_euler[1]
                curr_data["P3P_KAPPA"] = est_euler[2]
                curr_data["P3P_INCNT"] = nr_inliers
                curr_data["P3P_ERR"] = proj_error
                curr_data["MM_IMG"] = np.array2string(mm_img, separator=',',formatter={'float_kind':lambda x: "%.1f" % x})
                curr_data["MM_OBJ"] = np.array2string(mm_obj, separator=',',formatter={'float_kind':lambda x: "%.3f" % x})
        
        pd_spatres = pd.DataFrame(topn_data)
        pd_spatres["P3P_INCNT"] = pd_spatres["P3P_INCNT"].fillna(-1).astype(int)
        pd_spatres.fillna(np.NaN, inplace=True)
        
        pd_spatres.sort_values(["CNT", "MEAN_DIST"], ascending=[False, True], inplace=True)
        pd_spatres["CNT_RANK"] = np.arange(1, len(pd_spatres)+1, dtype=np.uint32)
        
        pd_spatres.sort_values(["NCNT", "MEAN_DIST"], ascending=[False, True], inplace=True)
        pd_spatres["NCNT_RANK"] = np.arange(1, len(pd_spatres)+1, dtype=np.uint32)
        
        pd_spatres.sort_values(["P3P_INCNT", "P3P_ERR"], ascending=[False, True], inplace=True)
        pd_spatres["P3P_RANK"] = np.arange(1, len(pd_spatres)+1, dtype=np.uint32)
        
        pd_spatres.drop(columns=["MEAN_DIST", "RANK"], inplace=True)
        pd_spatres.rename(columns={"GID_ORI":"GID_AZIMUT"}, inplace=True)
        pd_spatres["GID_AZIMUT"] = np.deg2rad(pd_spatres["GID_AZIMUT"])
        
        obj_coords = pd_spatres[["P3P_E", "P3P_N"]].values
        obj_coords_px = world2pixel(obj_coords, curr_gt)
        
        obj_h = pd_spatres[["P3P_H"]].values.ravel()
        obj_dgm_h = np.full(len(obj_coords_px), fill_value=9999).ravel()
        
        valid_mask = (obj_coords_px[:, 0] >= 0)  &  (obj_coords_px[:, 0] < curr_dgm_h) & (obj_coords_px[:,1] >= 0) & (obj_coords_px[:,1] < curr_dgm_w)
        
        valid_obj_coords_px = obj_coords_px[valid_mask, :]
        valid_obj_coords_dgm_h = curr_dgm[valid_obj_coords_px[:,0].ravel(), valid_obj_coords_px[:, 1].ravel()]
        
        obj_dgm_h[valid_mask] = valid_obj_coords_dgm_h
        
        obj_h_above_dgm = obj_h - obj_dgm_h
        obj_h_above_dgm[~valid_mask] = np.NaN        
        pd_spatres["P3P_H_DGM"] = obj_h_above_dgm

        pd_spatres["P3P_ALPHA_NORTH"] = (np.deg2rad(450) - pd_spatres["P3P_ALPHA"]) % np.deg2rad(360)
        
        pd_spatres = pd_spatres[["HFOV", "CNT_RANK", "NCNT_RANK", "P3P_RANK", "GID", "GID_E", "GID_N", "GID_H", "GID_AZIMUT", "P3P_E", "P3P_N", "P3P_H", "P3P_H_DGM", "P3P_ALPHA", "P3P_ALPHA_NORTH", "P3P_ZETA", "P3P_KAPPA", "P3P_INCNT", "P3P_ERR", "MM_IMG", "MM_OBJ"]]
        pd_spatres.to_csv(coarse_path.replace(".hdf5", ".csv"), sep=";", encoding="utf-8", index=False, float_format="%.6f")
        
        print("==============================================")
