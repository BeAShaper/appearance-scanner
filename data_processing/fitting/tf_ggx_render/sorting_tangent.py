import numpy as np
import cv2
import shutil
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_root")
parser.add_argument("which_axis",type=int,choices=[0,1,2])
parser.add_argument("--if_neg",action="store_true")

args = parser.parse_args()

data_root = args.data_root#"F:/Turbot_freshmeat/main_results/1_18/cloth2_1024/"

with open(data_root+"TATB.bin","rb") as pf:
    tA = np.fromfile(pf,np.float32,9).reshape((3,3)).T
    tB = np.fromfile(pf,np.float32,3).reshape((3,1))

tA_inv = np.linalg.inv(tA)

# if not os.path.exists(data_root+"fitted_grey_merged/tangent_fitted_global_bak.exr"):
#     print("this one has been processed.")
#     exit()

# shutil.copyfile(data_root+"fitted_grey_merged/tangent_fitted_global.exr",data_root+"fitted_grey_merged/tangent_fitted_global_bak.exr")

tangent_map = cv2.imread(data_root+"fitted_grey_merged/tangent_fitted_global.exr",cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)[:,:,::-1]

tangent_map = tangent_map*2.0-1.0

tangent_map = np.reshape(tangent_map,(-1,3))
# origin_map = np.zeros_like(tangent_map)

# tangent_map = np.matmul(tA,tangent_map.T)+tB
# origin_map = np.matmul(tA,origin_map.T)+tB

# tangent_map = tangent_map-origin_map

# tangent_map = tangent_map.T

tangent_map = tangent_map / (np.linalg.norm(tangent_map,axis=1,keepdims=True)+1e-6)

# for which_axis in range(3):
z_axis = np.zeros_like(tangent_map)
z_axis[:,args.which_axis] = 1.0 if not args.if_neg else -1.0

tangent_map = np.where(np.sum(tangent_map*z_axis,axis=1,keepdims=True) > 0.0,tangent_map,-tangent_map)

# tangent_map = tangent_map.T#(3,-1)
# origin_map = np.zeros_like(tangent_map)
# tangent_map = np.matmul(tA_inv,tangent_map-tB)
# origin_map = np.matmul(tA_inv,origin_map-tB)
# tangent_map = tangent_map-origin_map
# tangent_map = tangent_map.T

# tangent_map = tangent_map / (np.linalg.norm(tangent_map,axis=1,keepdims=True)+1e-6)

tangent_map = np.reshape(tangent_map,(1024,1024,3))

tangent_map = tangent_map*0.5+0.5

normal_map = cv2.imread(data_root+"fitted_grey_merged/normal_fitted_global.exr",cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)[:,:,::-1]
normal_map_norm = np.linalg.norm(normal_map,axis=2,keepdims=True)
tangent_map = np.where(normal_map_norm>0.0,tangent_map,np.zeros_like(tangent_map))

cv2.imwrite(data_root+"fitted_grey_merged/tangent_fitted_global_{}_{}.exr".format(args.which_axis,("pos" if not args.if_neg else "neg")),tangent_map[:,:,::-1])