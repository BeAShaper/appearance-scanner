import cv2
import numpy as np
import sys
sys.path.append("../utils/")
from dir_folder_and_files import make_dir
from subprocess import Popen

import argparse

parser = argparse.ArgumentParser(usage="recheck valid with undistort mask")
parser.add_argument("data_root",help="data root")
parser.add_argument("view_num",type=int,help="view_num")
args = parser.parse_args()

img_width = 2448
img_height = 2048

if __name__ == "__main__":
    data_root = args.data_root
    #######################
    ######step 1 get mask bin
    #######################
    # queue = []
    # for view_no in range(args.view_num):
    #     theProcess = Popen(
    #         [
    #             "../exe/exr_to_float.exe",
    #             args.data_root+"ae/{}/".format(view_no),
    #             "mask_udt_cam00.exr",
    #             "mask_udt_cam00.bin"
    #         ]
    #     )
    #     queue.append(theProcess)
    #     if len(queue) > 5:
    #         exitcodes = [q.wait() for q in queue]
    #         print("get mask bin:",exitcodes)
    #         queue = []

    # if len(queue) != 0:
    #     exitcodes = [q.wait() for q in queue]
    #     print("get mask bin:",exitcodes)
    #     queue = []
    

    #######################
    ######step 2 build valid map
    #######################
    valid_mask_map = {}
    print("building valid map using undistorted mask...")
    for view_no in range(args.view_num):
        print("{}/{}".format(view_no+1,args.view_num))
        tmp_mask = np.fromfile(args.data_root+"ae/{}/mask_udt_cam00.bin".format(view_no),np.float32).reshape([img_height,img_width,3])[:,:,0]>0.0 #[2048,2448,1]
        tmp_map = {(view_no,u,v):tmp_mask[v,u] for u in range(img_width) for v in range(img_height)}
        valid_mask_map = {**valid_mask_map,**tmp_map}

    #######################
    ######step 3 drop the invalid point
    #######################
    