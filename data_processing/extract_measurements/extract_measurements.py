import sys
import argparse
import numpy as np
import cv2
import os
import struct

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract measurements from raw images.')
    parser.add_argument('data_root',help='That path to data.please add / at the end of this string.')
    parser.add_argument('view_num',type=int,help="View num.")
    parser.add_argument('lighting_pattern_num',type=int,help="Lighting patterns num.")
    parser.add_argument('model_path',help="Path to pretrained model.")

    args = parser.parse_args()

    data_root = args.data_root + "/raw_images/"

    with open(args.model_path+"/0/maxs.bin","rb") as pf:
        maxs = np.fromfile(pf,np.float32).reshape([args.lighting_pattern_num,1])

    tmp_image = cv2.imread(data_root+"img_0.png")

    print("PHOTOS NUMBER : {} ".format(args.view_num),tmp_image.shape)
    
    mask = np.ones_like(tmp_image)

    if len(mask.shape) == 2:
        mask = np.expand_dims(mask,axis=2)
    if mask.shape[2] == 1:
        mask = np.repeat(mask,3,axis=2)
    valid_idxes = np.where(mask > 0)

    for which_image in range(args.view_num):
        colmap_img_path = os.path.relpath(data_root[:-11])
        colmap_img_path = colmap_img_path.replace("\\", "/")
        

        if not os.path.isfile(data_root+"img_{}.png".format(which_image)) or not os.path.isfile(colmap_img_path+"/sfm/images/img_{}.png".format(which_image)):
            continue

        tmp_image = cv2.imread(data_root+"img_{}.png".format(which_image))[:,:,::-1]
        tmp_measurements = tmp_image[valid_idxes].reshape((-1,3,1))

        tmp_measurements = np.transpose(tmp_measurements,[0,2,1]) / 255.0 * maxs[which_image % args.lighting_pattern_num]
        
        tmp_measurements.astype(np.float32).tofile(data_root+"cam00_data_{}_nocc_compacted.bin".format(which_image))
        print("\rProcess : ", "%.2f"%(which_image/args.view_num*100), "%", end="")

    print("\n[EXTRACT MEASUREMENTS]done.")