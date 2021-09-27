import numpy as np
import cv2
import os

data_root1 = "F:/Turbot_freshmeat/main_results/1_18/mask2_1024/"
pass_roots = [
    "F:/Turbot_freshmeat/main_results/1_18/mask2_1024_2pass/",
]


save_root = data_root1+"fitted_grey_merged/"
os.makedirs(save_root,exist_ok=True)

for a_file in ["axay_fitted.exr","normal_fitted_global.exr","pd_fitted.exr","ps_fitted.exr","tangent_fitted_global.exr"]:
    print(a_file)
    origin = cv2.imread(data_root1+"fitted_grey/"+a_file,cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
    for a_pass_root in pass_roots:
        check_map  = cv2.imread(a_pass_root+"check_map.exr",cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
        pass_2 = cv2.imread(a_pass_root+"fitted_grey/"+a_file,cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)

        if not a_pass_root == "F:/Turbot_freshmeat/main_results/1_18/rabbit2_1024_5pass/":
            origin = np.where(check_map>0,pass_2,origin)
        else:
            if a_file == "ps_fitted.exr":
                origin = np.where(check_map>0,check_map*pass_2+(1.0-check_map)*origin,origin)

    cv2.imwrite(save_root+a_file,origin)