import numpy as np
import cv2

data_root = "F:/Turbot_freshmeat/main_results/1_18/mask2_1024_2pass/"

tex_uv = np.fromfile(data_root+"texturemap_uv.bin",np.int32).reshape((-1,2))
normals = np.fromfile(data_root+"normals_geo_global.bin",np.float32).reshape((-1,3))

img = np.zeros((1024,1024,3),np.float32)

img[tex_uv[:,1],tex_uv[:,0]] = normals*0.5+0.5

cv2.imwrite(data_root+"normal_global.exr",img[:,:,::-1])