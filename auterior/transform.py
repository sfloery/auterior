import numpy as np

def world2pixel(obj_coords, gt):
    feat_vx_col = np.floor((obj_coords[:, 0] - gt[0]) / gt[1]).astype(int)
    feat_vx_row = np.floor((obj_coords[:, 1] - gt[3]) / gt[5]).astype(int)
    return np.hstack((feat_vx_row.reshape(-1, 1), feat_vx_col.reshape(-1, 1)))

def pixel2world(img_coords, gt):
    
    # xoffset, px_w, rot1, yoffset, rot2, px_h = gt
    # supposing x and y are your pixel coordinate this 
    # is how to get the coordinate in space.
    pos_x = gt[1] * img_coords[:, 1] + gt[2] * img_coords[:, 0] + gt[0]
    pos_y = gt[4] * img_coords[:, 1] + gt[5] * img_coords[:, 0] + gt[3]

    # shift to the center of the pixel
    pos_x += gt[1] * 0.5
    pos_y += gt[5] * 0.5
    
    return np.hstack((pos_x.reshape(-1, 1), pos_y.reshape(-1, 1)))

def rot_cv2ori(rmat):
    rx_200 = np.diag((1., -1., -1.))
    rx_200_T = rx_200.T
    rmat_ori = rx_200_T @ rmat
    rmat_ori = rmat_ori.T
    return rmat_ori

def rot2alzeka(R, unit="rad"):
    #extract both possible sets of rotation angles alpha, zeta and kappa
    ze_1_rad = np.arccos(R[2,2])
    ze_2_rad = 2*np.pi - ze_1_rad
    if ze_2_rad < 0:
        ze_2_rad += 2*np.pi

    ka_1_rad = np.arctan2(R[2, 1], R[2, 0]*(-1))
    if ka_1_rad < 0:
        ka_2_rad = ka_1_rad + np.pi
    else:
        ka_2_rad = ka_1_rad - np.pi

    al_1_rad = np.arctan2(R[1, 2], R[0, 2])
    if al_1_rad < 0:
        al_2_rad = al_1_rad + np.pi
    else:
        al_2_rad = al_1_rad - np.pi

    alzekas = np.array([[al_1_rad, ze_1_rad, ka_1_rad], [al_2_rad, ze_2_rad, ka_2_rad]])
    
    if unit == "rad":
        return alzekas
    elif unit == "deg":
        return np.rad2deg(alzekas)
    
def select_alzeka(alzekas=None, prc=None, gcps=None):
    prc = prc.ravel()
    
    al_1 = alzekas[0, 0]
    al_2 = alzekas[1, 0]
    al_1n = np.deg2rad((450 - np.rad2deg(al_1)) % 360)
    al_2n = np.deg2rad((450 - np.rad2deg(al_2)) % 360)
    
    diff_e = gcps[:, 0] - prc[0]
    diff_n = gcps[:, 1] - prc[1]
    dir_2_gcp = np.arctan2(diff_e, diff_n)
    dir_2_gcp[dir_2_gcp < 0] = dir_2_gcp[dir_2_gcp < 0] + 2*np.pi #make sure directions are between 0 and 360 degrees
    
    d_gcp2al1 = np.mean(np.abs(al_1n - dir_2_gcp))
    d_gcp2al2 = np.mean(np.abs(al_2n - dir_2_gcp))
    
    if d_gcp2al1 < d_gcp2al2:
        return alzekas[0, :]
    else:
        return alzekas[1, :]