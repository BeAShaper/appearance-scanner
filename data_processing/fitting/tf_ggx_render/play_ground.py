import numpy as np
import cv2
from subprocess import Popen
import queue



if __name__ == "__main__":

    data_root = "F:/Turbot_freshmeat/main_results/1_18/rabbit_512_1/"

    # gutter_map = cv2.imread(data_root+"mesh_512/meshed-poisson_obj_result_vis.png")
    # print(gutter_map.min())
    # valid = (gutter_map > 200).any(axis=2,keepdims=True)
    # valid = np.repeat(valid,3,axis=2)

    # pd = np.where(valid, np.ones_like(valid,dtype=np.float32),np.zeros_like(valid,dtype=np.float32))

    # cv2.imwrite(data_root+"fitted_grey/pd_fitted.exr",pd)

    normal_map = cv2.imread(data_root+"fitted_grey/normal_global(5).exr",cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)[:,:,::-1]
    normal_map = normal_map*2.0-1.0

    normal_map = np.reshape(normal_map,(-1,3))
    z = np.zeros_like(normal_map)
    z[:,2] = 1.0

    tangent_map = np.cross(normal_map,z)

    tangent_map = np.reshape(tangent_map,(512,512,3))
    tangent_map = tangent_map*0.5+0.5

    cv2.imwrite(data_root+"fitted_grey/tangent_fitted_global.exr",tangent_map)