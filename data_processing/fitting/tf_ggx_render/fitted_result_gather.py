import numpy as np
import argparse
import sys
sys.path.append("../utils/")
from dir_folder_and_files import make_dir
import shutil

parser = argparse.ArgumentParser(usage="fitting result gather")
parser.add_argument("data_root")
parser.add_argument("thread_num",type=int)

args = parser.parse_args()

if __name__ == "__main__":
    save_root = args.data_root+"gathered_all/"
    make_dir(save_root)
    with open(save_root+"params_gathered.bin","wb") as pf:
        for thread_id in range(args.thread_num):
            pf.write(open(args.data_root+"{}/fitted.bin".format(thread_id),"rb").read())
    
    positions = np.zeros([0,3],np.float32)
    with open(save_root+"position_gathered.bin","wb") as pf:
        for thread_id in range(args.thread_num):
            tmp_pos = np.fromfile(args.data_root+"{}/position.bin".format(thread_id),np.float32).reshape([-1,3])
            positions = np.concatenate([positions,tmp_pos],axis=0)
        positions.tofile(pf)
    
    # shutil.copyfile(args.data_root+"best_rotate_angle.bin",save_root+"best_rotate_angle.bin")
    shutil.copyfile("../../nn_normals.bin",save_root+"nn_normals.bin")