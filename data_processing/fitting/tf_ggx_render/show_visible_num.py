import cv2
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",default="F:/Turbot_freshmeat/main_results/1_18/rabbit2_1024/")
    parser.add_argument("--tex_resolution",type=int,default=1024)
    
    args = parser.parse_args()

    visible_view_num = np.fromfile(args.data_root+"visible_views_num.bin",np.int32).reshape((-1,1))
    tex_uv = np.fromfile(args.data_root+"texturemap_uv.bin",np.int32).reshape((-1,2))

    assert tex_uv.shape[0] == visible_view_num.shape[0]

    img = np.zeros((args.tex_resolution,args.tex_resolution,3),np.uint8)

    img[tex_uv[:,1],tex_uv[:,0]] = visible_view_num

    cv2.imwrite(args.data_root+"visible_num.png",img)