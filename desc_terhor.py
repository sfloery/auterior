import argparse
import os
import h5py
from tqdm import tqdm
import numpy as np
from scipy.interpolate import splrep
from scipy.spatial.distance import cdist
from auterior.features import imtf

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument('-hdf_path', type=str, required=True, help="Path to the *.hdf5 file containing the calculated terrain horizons.")
    parser.add_argument('-out_path', type=str, required=True, help="File path used to store the terrain horizon parts.")
    parser.add_argument('-feat', required=False, choices=["imtf"], default="imtf", help="Feature descriptor used to describe the terrain horizon parts. Currently only 'imtf' is supported.")
    parser.add_argument('-w', required=False, type=int, default=[10, 20], nargs='*', help="Width in ° of the extracted horizon parts. Example: -w 10 20 or -w 10")
    parser.add_argument('-s', required=False, type=float, default=1, help="Step width between horizon parts in °.")
    parser.add_argument('-n', required=False, type=int, default=16, help="Number of sampling points for each horizon part.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf_path):
        raise FileNotFoundError("%s not found." % (args.hdf_path))
    
    if os.path.exists(args.out_path):
        raise FileExistsError("%s already exists." % (args.out_path))

    in_hdf = h5py.File(args.hdf_path, mode="r")
    
    out_hdf5 = h5py.File(args.out_path, "w")
    out_hdf5.attrs["terhor_path"] = args.hdf_path
    out_hdf5.attrs["dtm_path"] = in_hdf.attrs["dtm_path"]

    for hx, hpw in enumerate(args.w):
        hpw_name = "%s_%s" % (args.feat, hpw)
        hpw_grp = out_hdf5.create_group(hpw_name)
        
        hpw_grp.attrs["feat"] = args.feat
        hpw_grp.attrs["w_deg"] = hpw
        hpw_grp.attrs["n"] = args.n
        hpw_grp.attrs["step_deg"] = args.s

        print("Extracting terrain horizon parts with w=%i°, n=%i, s=%i°:" % (hpw, args.n, args.s))
        
        for ix, gid in enumerate(tqdm(in_hdf.keys())):
            
            gid_data = in_hdf[gid][()] #gid_data: l, b, E, N, H
                
            gid_pnt_e = in_hdf[gid].attrs["E"]
            gid_pnt_n = in_hdf[gid].attrs["N"]
            gid_pnt_h = in_hdf[gid].attrs["H"]
            gid_pnt_coords = np.array([[gid_pnt_e, gid_pnt_n, gid_pnt_h]])
            
            #extract horizon coordinates both in image and object space
            gid_hor_img = gid_data[:, [0, 1]] #l,b: cylindrical coordinates in deg; l=0 --> North; b=0 horizont;
            gid_hor_obj = gid_data[:, [2, 3, 4]] #E,N,H: coordinates in UTM-32N
            
            left_hor_ix = np.argwhere(gid_hor_img[:, 0] <= hpw - args.s)
            left_hor = gid_hor_img[left_hor_ix.ravel(), :]
            left_hor[:, 0] += 360
            
            gid_hor_img = np.vstack((gid_hor_img, left_hor))
            gid_hor_img = gid_hor_img[np.argsort(gid_hor_img[:, 0]), :]

            gid_hor_obj = np.vstack((gid_hor_obj, gid_hor_obj[left_hor_ix.ravel(), :]))
            
            #depreceated; smooth is now done while calculatint the terrain horizons
            #terrain horizons very close to the current gid have staircase effects;
            #hence we use a smoothed spline with weights based on the distance to the gid
            #these values have been empirically found
            # dist_gid2hor = cdist(gid_pnt_coords, gid_hor_obj).ravel()
            # weights_hor = np.full(len(gid_hor_img), fill_value=100)
            # weights_hor[dist_gid2hor <= 1000] = 3
            # weights_hor[dist_gid2hor <= 500] = 1.5
            # weights_hor[dist_gid2hor <= 100] = 1
            
            hor_min_azi = np.min(gid_hor_img[:, 0])
            hor_max_azi = np.max(gid_hor_img[:, 0])

            spl = splrep(gid_hor_img[:, 0], gid_hor_img[:, 1], k=1)#, w=weights_hor)
            assert np.isnan(spl[1]).any() == False, "Terrain horizon could not be represented using a spline."
            
            parts_feat_vec, parts_ori = imtf(spl, 
                                             w_deg=hpw, 
                                             nr_samples=args.n, 
                                             step=args.s, 
                                             min_azi = hor_min_azi, 
                                             max_azi = hor_max_azi, 
                                             normalize=True,
                                             mode="dist")
            
            parts_ori[parts_ori >= 360] -= 360
            parts_feat_vec = parts_feat_vec[np.argsort(parts_ori), :]
            parts_ori = np.sort(parts_ori)
            
            parts_meta_vec = np.column_stack((np.repeat(int(gid), len(parts_feat_vec)), parts_ori))
            
            nr_parts = np.shape(parts_feat_vec)[0]
            
            if ix == 0:
                hpw_grp.create_dataset('feat', data=parts_feat_vec.astype(np.float32), compression="gzip", chunks=True, maxshape=(None, parts_feat_vec.shape[1]))     
                hpw_grp.create_dataset('feat_meta', data=parts_meta_vec.astype(np.float32), compression="gzip", chunks=True, maxshape=(None, parts_meta_vec.shape[1]))
                
                if hx == 0:
                    out_hdf5.create_dataset("gid", data=np.array([gid]).astype(np.int32), compression="gzip", maxshape=(None, ))
                    out_hdf5.create_dataset("gid_coords", data=gid_pnt_coords.astype(np.float32), compression="gzip", chunks=True, maxshape=(None, 3))
                                
            else:               
                hpw_grp["feat"].resize(hpw_grp["feat"].shape[0] + nr_parts, axis = 0)
                hpw_grp["feat"][-nr_parts:] = parts_feat_vec
                
                hpw_grp["feat_meta"].resize(hpw_grp["feat_meta"].shape[0] + nr_parts, axis = 0)
                hpw_grp["feat_meta"][-nr_parts:] = parts_meta_vec
                
                if hx == 0:
                    out_hdf5["gid"].resize(out_hdf5["gid"].shape[0] + 1, axis = 0)
                    out_hdf5["gid"][-1:] = np.array([gid]).astype(np.int32)
                    
                    out_hdf5["gid_coords"].resize(out_hdf5["gid_coords"].shape[0] + 1, axis = 0)
                    out_hdf5["gid_coords"][-1:] = gid_pnt_coords