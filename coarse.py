import argparse
import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import splrep
from skimage import io, measure
from auterior.features import imtf
from tqdm import tqdm
import os
import pandas as pd

def horimg2geom(hor_arr):
    hor_coords = np.argwhere(hor_arr == 1)
    #sort coordinates ascending by image column; hence horizon goes from left to right
    hor_coords = hor_coords[np.argsort(hor_coords[:, 1]), :]
    return hor_coords

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument('-hdf_path', type=str, required=True, help="Path to the *.hdf5 file containing the terrain horizon features.")
    parser.add_argument('-hor_path', type=str, required=True, help="Path to the image containing the horizon. If directory is given all *.tif are processed.")
    parser.add_argument('-hfov', type=float, required=False, nargs="*", default=[25, 30, 35, 40, 45, 50, 55, 60, 65], help="HFOV in degrees of the image. If unknown provide range of values which will be used.")
    parser.add_argument('-knn', type=int, default=1000, required=False, help="Number of neighbors returned in the matching.")
    parser.add_argument('-dir_bin', type=float, default=1, required=False, help="Bin size used for direction counting.")
    parser.add_argument('-topn', type=int, default=10, required=False, help="Topn candidates kept for each HFOV which will be further used in the spatial resection.")
    
    args = parser.parse_args()

    hor_img_paths = []
    if os.path.isdir(args.hor_path):
        for file in os.listdir(args.hor_path):
            if file.endswith(".tif"):
                hor_img_paths.append(os.path.join(args.hor_path, file))
    else:
        hor_img_paths.append(args.hor_path)
    
    hdf_terhor = h5py.File(args.hdf_path, "r")
    
    for hor_path in hor_img_paths:
        
        if not os.path.exists(hor_path):
            continue
        
        print("Matching %s with %s:" % (hor_path, args.hdf_path))
        
        #Laden des Horizontes
        hor_img = io.imread(hor_path)
        hor_img_h, hor_img_w = np.shape(hor_img)
        assert hor_img.ndim == 2, "Only single channel images are supported."
        assert hor_img.dtype == "uint8", "Only 8-bit images are supported."
        assert np.array_equal(np.unique(hor_img), np.array([0, 1])), "Horizon image must only contain 0 and 1."
        
        hdf_coarse = h5py.File(os.path.splitext(hor_path)[0] + ".hdf5", "w")
        hdf_coarse.attrs["terhor_path"] = hdf_terhor.attrs["terhor_path"]
        hdf_coarse.attrs["dtm_path"] = hdf_terhor.attrs["dtm_path"]
        
        hor_geom = horimg2geom(hor_img)
        
        hdf_coarse_hor = hdf_coarse.create_dataset("hor_geom", data=hor_geom)
        hdf_coarse_hor.attrs["img_h"] = hor_img_h
        hdf_coarse_hor.attrs["img_w"] = hor_img_w
        hdf_coarse_hor.attrs["cols"] = "rc"
        
        hor_geom = measure.approximate_polygon(hor_geom, 1)
        hor_row = hor_img_h - hor_geom[:, 0]
        hor_col = hor_geom[:, 1]
        
        # =====================================================================
        # matching with the individual terrain horizon parts
        # =====================================================================
        cgrp_name = "imtf"
        
        #store gid ids and coordinates for later usage
        gids = hdf_terhor["gid"][()].squeeze().astype(np.uint32)
        gids_coords = hdf_terhor["gid_coords"][()].squeeze()
        pd_gids = pd.DataFrame({"gid":gids, "GID_E":gids_coords[:, 0], "GID_N":gids_coords[:, 1], "GID_H":gids_coords[:, 2]})
        pd_gids["gid"] = gids.astype(str)
            
        for fx, feat_key in enumerate(list(hdf_terhor.keys())):
            
            if feat_key == "gid" or feat_key == "gid_coords":
                continue
            
            print("...using %s:" % (feat_key))            
            hdf_feat = hdf_terhor[feat_key]
            
            feat_w = int(feat_key.split("_")[1])
            feat_nrs = int(hdf_feat.attrs["n"])
            feat_step = float(hdf_feat.attrs["step_deg"])
            
            cgrp_name += "_%s" % (feat_w)
            
            feat_vec = hdf_feat["feat"][()]
            feat_vec = np.nan_to_num(feat_vec, nan=0)

            feat_meta = hdf_feat["feat_meta"][()]
            
            nn_searcher = NearestNeighbors(n_neighbors=args.knn, algorithm='auto', metric="manhattan", n_jobs=-1).fit(feat_vec)
            
            hdf_coarse_w = hdf_coarse.create_group(feat_key)
            hdf_coarse_w.attrs["n"] = feat_nrs
            hdf_coarse_w.attrs["step_deg"] = feat_step
            
            for hfov_deg in tqdm(args.hfov):

                vfov_deg = hfov_deg * (hor_img_h/float(hor_img_w))
                
                cam_hor_col_deg = np.interp(hor_col, (0, hor_img_w), (0, hfov_deg))
                cam_hor_row_deg = np.interp(hor_row, (0, hor_img_h), (0, vfov_deg))
                cam_hor_row_deg -= vfov_deg/2.
                
                hor_min_azi = cam_hor_col_deg.min()
                hor_max_azi = cam_hor_col_deg.max()
                
                uq_cols, uq_cols_ix = np.unique(cam_hor_col_deg, return_index=True)
                uq_rows = cam_hor_row_deg[uq_cols_ix]
                hor_range_deg = np.max(cam_hor_col_deg) - np.min(cam_hor_col_deg)
                                
                spl = splrep(uq_cols, uq_rows, k=1)
                assert np.isnan(spl[1]).any() == False, "Image horizon could not be represented using a spline."

                hor_feat_vec, hor_ori = imtf(spline=spl,
                                            w_deg=feat_w, 
                                            nr_samples=feat_nrs, 
                                            step=feat_step, 
                                            min_azi = hor_min_azi, 
                                            max_azi = hor_max_azi, 
                                            normalize=True, 
                                            mode="dist")
                
                hor_feat_vec = np.nan_to_num(hor_feat_vec, nan=0)
                        
                if hor_feat_vec.ndim == 1:
                    hor_feat_vec = hor_feat_vec.reshape(1, -1)
                
                hor_ori = hor_ori - hfov_deg/2.
                    
                dd, ii = nn_searcher.kneighbors(hor_feat_vec) ##dd: distances; ii: contains indices to the k nearest neighbors of hor_feat_vec to feat_vec
                
                #extract the point grid ids corresponding to the ii's
                ii_grd_pnt_ids = feat_meta[ii, 0].astype(np.uint16)#.ravel().astype(np.uint16)
                ii_grd_pnt_oris = feat_meta[ii, 1]
                
                # ii_grd_pnt_oris = abolute alpha counted from north (=0) - east (=90)...
                #hor_ori = relative alpha from image center (-XX -> left of center; +XX --> right of center)
                ii_grd_pnt_oris = ii_grd_pnt_oris-hor_ori.reshape(-1, 1)#).ravel()
                
                hdf_coarse_w_hfov = hdf_coarse_w.create_group(str(hfov_deg))
                hdf_coarse_w_hfov.create_dataset("nn_pnts_ids", dtype=np.uint16, data=ii_grd_pnt_ids)
                hdf_coarse_w_hfov.create_dataset("nn_dists", dtype=np.float32, data=dd)
                hdf_coarse_w_hfov.create_dataset("nn_oris", dtype=np.float32, data=ii_grd_pnt_oris)
                
        # =====================================================================
        # combining individual matching results
        # =====================================================================
        print("...combining results.")
        hdf_coarse_comb = hdf_coarse.require_group(cgrp_name)
        
        for hor_deg in args.hfov:
            hor_deg_str = "%i" % (hor_deg)
            
            hdf_coarse_comb_hfov = hdf_coarse_comb.require_group(hor_deg_str)
            
            comb_nn_pnts_ids = None
            comb_nn_dists = None
            comb_nn_oris = None
            
            for feat_key in list(hdf_coarse.keys()):
                
                if feat_key in [cgrp_name, "hor_geom"]:
                    continue
                
                if hor_deg_str in list(hdf_coarse[feat_key].keys()):
                    
                    nn_pnts_ids = hdf_coarse[feat_key][hor_deg_str]["nn_pnts_ids"][()]
                    nn_dists = hdf_coarse[feat_key][hor_deg_str]["nn_dists"][()]
                    nn_oris = hdf_coarse[feat_key][hor_deg_str]["nn_oris"][()]
                    
                    if comb_nn_pnts_ids is None:
                        comb_nn_pnts_ids = nn_pnts_ids
                        comb_nn_dists = nn_dists
                        comb_nn_oris = nn_oris
                    else:
                        comb_nn_pnts_ids = np.vstack((comb_nn_pnts_ids, nn_pnts_ids))
                        comb_nn_dists = np.vstack((comb_nn_dists, nn_dists))
                        comb_nn_oris = np.vstack((comb_nn_oris, nn_oris))
            
            hdf_coarse_comb_hfov.create_dataset("nn_pnts_ids", dtype=np.uint16, data=comb_nn_pnts_ids)
            hdf_coarse_comb_hfov.create_dataset("nn_dists", dtype=np.float32, data=comb_nn_dists)
            hdf_coarse_comb_hfov.create_dataset("nn_oris", dtype=np.float32, data=comb_nn_oris)
        
        # =====================================================================
        # ranking matching results
        # =====================================================================
        print("...ranking results.") 
        topn_data = []
        for hfov_deg in args.hfov:
            
            hfov_deg_str = "%i" % (hfov_deg)
                    
            if not hfov_deg_str in list(hdf_coarse_comb.keys()):
                continue
            
            hdf_coarse_comb_hfov = hdf_coarse_comb[hfov_deg_str]
            
            knn_gids = hdf_coarse_comb_hfov["nn_pnts_ids"][()].astype(np.uint16).ravel()
            knn_dist = hdf_coarse_comb_hfov["nn_dists"][()].astype(np.float32).ravel()
            knn_oris = hdf_coarse_comb_hfov["nn_oris"][()].astype(np.float32).ravel()
            
            nr_parts = np.shape(hdf_coarse_comb_hfov["nn_pnts_ids"][()])[0]
            
            uq_ids, uq_inv, uq_cnts = np.unique(knn_gids, return_counts=True, return_inverse=True)
            
            hfov_gids = []
            hfov_oris = []
            hfov_dist = []
        
            for ix, gid in enumerate(uq_ids):
                gid_cnt = uq_cnts[ix]
                
                uid_ix = np.argwhere(uq_inv == ix).ravel()
                uid_oris = knn_oris[uid_ix]
                uid_dists = knn_dist[uid_ix]
                
                if gid_cnt == 1:
                    hfov_gids.append(gid)
                    hfov_oris.append(uid_oris)
                    hfov_dist.append(uid_dists)
                
                else:
                    
                    asc_ix = np.argsort(uid_oris)
                    uid_oris = np.array(uid_oris)[asc_ix]
                    uid_dists = np.array(uid_dists)[asc_ix]
                    
                    ori_diff = np.diff(uid_oris) > args.dir_bin
                    ori_diff_ix = np.nonzero(ori_diff)[0] + 1
                    
                    oris_grps = np.split(uid_oris, ori_diff_ix)
                    dist_grps = np.split(uid_dists, ori_diff_ix)
                                        
                    hfov_gids.extend([gid]*len(oris_grps))
                    hfov_oris.extend(oris_grps)
                    hfov_dist.extend(dist_grps)
                    
            pd_uid = pd.DataFrame(columns=["GID", "CNT", "ORIS", "DISTS"])
            pd_uid["GID"] = hfov_gids
            pd_uid["ORIS"] = hfov_oris
            pd_uid["DISTS"] = hfov_dist
            pd_uid["HFOV"] = hfov_deg_str
            
            pd_uid["GID_ORI"] = pd_uid['ORIS'].apply(np.mean)
            pd_uid["MEAN_DIST"] = pd_uid['DISTS'].apply(np.mean)
            pd_uid["CNT"] = pd_uid["ORIS"].apply(len)
            pd_uid["NCNT"] = pd_uid["CNT"] / nr_parts
            
            pd_uid.drop(columns=["ORIS", "DISTS"], inplace=True)
            
            pd_uid.sort_values(["CNT", "MEAN_DIST"], ascending=[False, True], inplace=True)
            pd_uid.insert(2, 'RANK', np.arange(1, len(pd_uid)+1, dtype=np.uint32))
                        
            pd_uid = pd_uid[pd_uid['RANK'] <= 100]
            pd_uid["GID"] = pd_uid["GID"].astype(str)
                    
            pd_uid = pd_uid.merge(pd_gids, left_on="GID", right_on="gid")
            pd_uid.drop(columns=["gid"], inplace=True)
            pd_uid["GID"] = pd_uid["GID"].astype(int)
            
            cols = ["GID", "CNT", "NCNT", "RANK", "GID_ORI", "MEAN_DIST", "GID_E", "GID_N", "GID_H"]
            pd_uid = pd_uid[cols]
            # pd_uid_vals = pd_uid[cols].values
                            
            coarse_ds = hdf_coarse_comb_hfov.create_dataset("coarse", data=pd_uid.values)
            coarse_ds.attrs.create("COLS", ";".join(cols))
            
            pd_uid["HFOV"] = hfov_deg
            topn_data.append(pd_uid[pd_uid["RANK"] <= args.topn])
        
        pd_topn = pd.concat(topn_data, axis=0, ignore_index=True)
        hdf_coarse_topn = hdf_coarse.create_dataset("coarse_topn", data=pd_topn)
        hdf_coarse_topn.attrs["cols"] = ";".join(cols) + ";HFOV"
        
        hdf_coarse.close()
        print("==============================================")
        
    hdf_terhor.close()
