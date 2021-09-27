import numpy as np
import cv2
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Brighten the image")
    parser.add_argument("data_root")
    parser.add_argument("view_num",type=int)

    args = parser.parse_args()

    save_root = args.data_root+"sfm/" 
    os.makedirs(save_root,exist_ok=True)

    view_idx = 0
    images_save_root = save_root+"images/"
    os.makedirs(images_save_root,exist_ok=True)

    for which_view in range(args.view_num):
        
        if not os.path.isfile(args.data_root+"raw_images/img_{}.png".format(which_view)):
            continue
        
        img = cv2.imread(args.data_root+"raw_images/img_{}.png".format(which_view))
        
        img = np.clip(img.astype(np.float32) * 2,0,255.0)
        img= img.astype(np.uint8)

        cv2.imwrite(images_save_root+"img_{}.png".format(which_view),img)
        if which_view % 100 == 0:
            print(which_view)