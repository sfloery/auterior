import numpy as np
from scipy.interpolate import splev

def create_parts(min_azi, max_azi, w_deg, step):
    parts_lt = np.arange(min_azi, max_azi+0.1-w_deg, step=step)
    parts_rt = parts_lt + w_deg
    
    parts = np.column_stack((parts_lt.reshape(-1, 1), parts_rt.reshape(-1, 1)))
    parts_ori = parts_lt + w_deg/2.
    
    return parts, parts_ori

def imtf4points(spline, w_deg=None, nr_samples=None, points=None, normalize=True, mode=None):
    
    for px, p in enumerate(points):
        p_min_azi = p[0] - w_deg / 2.
        p_max_azi = p[0] + w_deg / 2.
        
        p_feat, _ = imtf(spline, w_deg=-1, nr_samples=nr_samples, min_azi=p_min_azi, max_azi=p_max_azi, normalize=normalize, mode=mode)
        
        if px == 0:
            parts_feat = p_feat
        else:
            parts_feat = np.vstack((parts_feat, p_feat))
        
    return parts_feat, points[:, 1]

def imtf(spline, w_deg=10, nr_samples=8, step=1, min_azi=None, max_azi=None, normalize=True, mode="area"):
    if w_deg == -1:
        parts = np.array([[min_azi, max_azi]])
        parts_ori = np.array([[min_azi + (max_azi-min_azi/2.)]])
    else:
        parts, parts_ori = create_parts(min_azi, max_azi, w_deg, step)

    T = int(np.floor(np.log2(nr_samples/2)))
    
    for px, prt in enumerate(parts):
        prt_sx = np.linspace(prt[0], prt[1], num=nr_samples)
        prt_sy = splev(prt_sx, spline)
        prt_pnts = np.column_stack((prt_sx, prt_sy))
        
        curr_ix = np.arange(0, nr_samples)
        
        #area formula
        #1/2*((x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))); #p1=prev, p2=curr, p3=next
        for k in range(1, T+1):
            hk = 2**(k-1)
            prev_ix = curr_ix - hk
            next_ix = curr_ix + hk
            prev_ix[prev_ix < 0] += nr_samples
            next_ix[next_ix >= nr_samples] -= nr_samples
            
            if mode == "complete" or mode == "area":
                #area calculation
                a1 = prt_sx[prev_ix]*(prt_sy[curr_ix]-prt_sy[next_ix])
                a2 = prt_sx[curr_ix]*(prt_sy[next_ix]-prt_sy[prev_ix])
                a3 = prt_sx[next_ix]*(prt_sy[prev_ix]-prt_sy[curr_ix])
                
                k_area = 0.5*(a1+a2+a3)

                if normalize:
                    max_area = np.nanmax(np.abs(k_area))
                    if max_area > 0:
                        k_area = k_area / max_area
            
            if mode == "complete" or mode == "dist":
                #centroid distance
                ctrd_x = (prt_sx[prev_ix] + prt_sx[curr_ix] + prt_sx[next_ix])/3.
                ctrd_y = (prt_sy[prev_ix] + prt_sy[curr_ix] + prt_sy[next_ix])/3.
                ctrd_pnts = np.column_stack((ctrd_x.reshape(-1, 1), ctrd_y.reshape(-1, 1)))
                
                k_dist = np.linalg.norm(ctrd_pnts - prt_pnts, axis=1).ravel()
                if normalize:
                    max_dist = np.nanmax(np.abs(k_dist))
                    if max_dist > 0:
                        k_dist = k_dist / max_dist
            
            if k == 1:
                if mode == "complete":
                    feat_area = k_area
                    feat_dist = k_dist
                elif mode == "area":
                    feat_area = k_area
                elif mode == "dist":
                    feat_dist = k_dist
            else:
                if mode == "complete":
                    feat_area = np.hstack((feat_area, k_area))
                    feat_dist = np.hstack((feat_dist, k_dist))
                elif mode == "area":
                    feat_area = np.hstack((feat_area, k_area))
                elif mode == "dist":
                    feat_dist = np.hstack((feat_dist, k_dist))
                
        if px == 0:
            if mode == "complete":
                parts_feat = np.hstack((feat_area, feat_dist))
            elif mode == "area":
                parts_area = feat_area
            elif mode == "dist":
                parts_dist = feat_dist
        else:
            if mode == "complete":
                parts_feat = np.vstack((parts_feat, np.hstack((feat_area, feat_dist))))
            elif mode == "area":
                parts_area = np.vstack((parts_area, feat_area))
            elif mode == "dist":
                parts_dist = np.vstack((parts_dist, feat_dist))
    
    if mode == "complete":
        return parts_feat, parts_ori
    elif mode == "area":
        return parts_area, parts_ori
    elif mode == "dist":
        return parts_dist, parts_ori