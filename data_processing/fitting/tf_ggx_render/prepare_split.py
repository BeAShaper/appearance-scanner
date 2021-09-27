import numpy as np
import shutil
import os
from subprocess import Popen
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_folder")#contains images_max_pooling normal_geo_global.bin tangent_geo_global.bin texturemap_uv.bin W.bin
    parser.add_argument("diff_sample_num")
    parser.add_argument("spec_sample_num")
    parser.add_argument("sample_view_num")
    parser.add_argument("thread_num")
    parser.add_argument('is_for_server')
    parser.add_argument("fix_normal")
    parser.add_argument("m_len_perview")
    parser.add_argument("tex_resolution")
    parser.add_argument("server_num")
    parser.add_argument("which_server")


    args = parser.parse_args()

    root_folder = args.root_folder
    root_folder_files = os.listdir(root_folder)

    fitting_files_wanted = ["normals_geo_global.bin","tangents_geo_global.bin","texturemap_uv.bin","W.bin"]

    for a_fitting_file_wanted in fitting_files_wanted:
        assert a_fitting_file_wanted in root_folder_files,"{}".format(a_fitting_file_wanted)
        
    print("finding images root...")
    for a_root_folder_file in root_folder_files:
        if "images" in a_root_folder_file:
            images_root = a_root_folder_file
            break
    
    BRDF_pointnet_root = root_folder+images_root+"/data_for_server/"
    BRDF_pointnet_data_dir = BRDF_pointnet_root+"data/"
    print("copying fitting files...")
    for a_fitting_file_wanted in fitting_files_wanted:
        shutil.copyfile(root_folder+a_fitting_file_wanted,BRDF_pointnet_data_dir+"images/"+a_fitting_file_wanted)
    
    print("inferencing...")
    the_infering_process = Popen(
        [
            "bash",
            "run.sh"
        ],
        cwd=BRDF_pointnet_root+"python_files/appearance_scanner/test_files/"
    )
    exit_code = the_infering_process.wait()
    print("exit_code:",exit_code)

    print("splitting...")
    the_process = Popen(
        [
            "python",
            "split_data_and_prepare_for_server.py",
            BRDF_pointnet_data_dir,
            args.diff_sample_num,#DIFF_SAMPLE_NUM
            args.spec_sample_num,#SPEC_SAMPLE_NUM
            args.sample_view_num,#SAMPLE_VIEW_NUM
            args.thread_num,#THREAD_NUM
            args.server_num,#THREAD_NUM
            args.which_server,#THREAD_NUM
            args.is_for_server,#IS_FOR_SERVER
            args.fix_normal,#IF_FIX_NORMAL
            args.m_len_perview,#M_LEN_PERVIEW
            args.tex_resolution#TEX_RESOLUTION
        ]
    )
    exit_code = the_process.wait()
    print("exit_code:",exit_code)

    print("fitting....")
    fitting_python_root = BRDF_pointnet_data_dir+"images/data_for_server/python_files/"
    the_process = Popen(
        [
            "bash",
            "run.sh"
        ],
        cwd=fitting_python_root
    )
    exit_code = the_process.wait()
    print("exit_code:",exit_code)

    if args.server_num == 1:
        print("moving result...")
        if os.path.exists(root_folder+"fitted_grey"):
            shutil.rmtree(root_folder+"fitted_grey")
        shutil.copytree(BRDF_pointnet_data_dir+"images/data_for_server/fitted_grey",root_folder+"fitted_grey")
        print("done")
    