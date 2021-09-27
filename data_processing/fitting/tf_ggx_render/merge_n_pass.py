import numpy as np
import cv2
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("texture_map_size",type=int)

    args = parser.parse_args()

    udt_folder_names = [
        # "undistort_feature_1pass",
        "undistort_feature"
    ]

    texture_map_names = [
        "axay_fitted.exr",
        "normal_fitted.exr",
        "normal_fitted_global.exr",
        "normal_geo.exr",
        "normal_geo_gloabl.exr",
        "pd_fitted.exr",
        "position_local_view.exr",
        "ps_fitted.exr",
        "tangent_fitted.exr",
        "tangent_fitted_global.exr",
    ]

    save_root = args.data_root+udt_folder_names[0]+"/texture_{}_final/".format(args.texture_map_size)
    os.makedirs(save_root,exist_ok=True)

    for a_texture_map_name in texture_map_names:
        origin = cv2.imread(args.data_root+udt_folder_names[0]+"/texture_{}/".format(args.texture_map_size)+a_texture_map_name,cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
        for which_pass in range(1,len(udt_folder_names)):
            current = cv2.imread(args.data_root+udt_folder_names[which_pass]+"/texture_{}/".format(args.texture_map_size)+a_texture_map_name,cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
            valid_map = cv2.imread(args.data_root+udt_folder_names[which_pass]+"/valid.exr",cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)

            valid_idx = np.where(valid_map > 0.0)

            origin[valid_idx] = current[valid_idx]
        
        cv2.imwrite(save_root+a_texture_map_name,origin)