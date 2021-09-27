import numpy as np
import cv2
import argparse
import os
from subprocess import Popen

def error_warning():
    print("ERROR OCCURED!!!!!")
    input()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Undistort LDR")
    parser.add_argument("data_root")
    parser.add_argument("intrinsic_file")

    args = parser.parse_args()

    src_folder = args.data_root+"sfm_images/images_0/"
    dst_folder = args.data_root+"sfm_images_udt/images_0/"

    os.makedirs(dst_folder, exist_ok=True)

    queue = []
    
    for which_view in range(args.view_num):
        theProcess=Popen(
            [
                "../tools/undistort_png.exe",
                src_folder+"pd_predicted_{}_{}.png".format(which_view),
                dst_folder+"pd_predicted_{}_{}.png".format(which_view),
                args.intrinsic_folder+"intrinsic0.yml",
                "0"#linear
            ],
        )
        
        queue.append(theProcess)
        if len(queue) > 10:
            exitcodes = [q.wait() for q in queue]
            print("exit codes:",exitcodes)
            if not all(ec==0 for ec in exitcodes):
                error_warning()
            queue=[]
            
    if len(queue) > 0:
        exitcodes = [q.wait() for q in queue]
        print("exit codes:",exitcodes)
        if not all(ec==0 for ec in exitcodes):
            error_warning()
        queue=[]