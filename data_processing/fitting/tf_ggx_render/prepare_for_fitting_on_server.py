import numpy as np
import os
import cv2
import shutil
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("origin_root",default="F:/Turbot_freshmeat/main_results/Cheongsam_1024_testfitting/",help="this folder should contains images_resolution/")
    parser.add_argument("tex_resolution",type=int,default=512,help="The resolution of texturemap")

    args = parser.parse_args()

    images_root = args.origin_root+"images_{}/".format(args.tex_resolution)
    images_root_noslash = images_root.strip("/")

    tmp_folder = args.origin_root+"fitting_folder_for_server/"
    if os.path.exists(tmp_folder):
        try:
            shutil.rmtree(tmp_folder)
        except Exception as e:
            print(e)
            time.sleep(3)
    os.makedirs(tmp_folder)

    shutil.copytree(images_root_noslash,tmp_folder+"images_{}".format(args.tex_resolution))
    data_file_names = [
        "normal_global.exr",
        "tangent_global.exr",
        "texturemap_uv.bin",
        "normals_geo_global.bin",
        "tangents_geo_global.bin",
    ]
    for data_file_name in data_file_names:
        shutil.copyfile(images_root+data_file_name,tmp_folder+data_file_name)

    shutil.copyfile(images_root+"W.bin",tmp_folder+"W.bin")

    fitting_temp_root = tmp_folder+"fitting_temp/"
    if os.path.exists(fitting_temp_root):
        shutil.rmtree(fitting_temp_root)
    shutil.copytree("../",fitting_temp_root,ignore=shutil.ignore_patterns(".git",".vscode"))
    shutil.copytree(images_root+"data_for_server/python_files/torch_renderer",fitting_temp_root+"torch_renderer",ignore=shutil.ignore_patterns(".git",".vscode"))