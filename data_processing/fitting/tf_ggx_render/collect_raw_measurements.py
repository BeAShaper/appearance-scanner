import shutil
import argparse
import numpy as np
import sys
sys.path.append("../utils/")
from dir_folder_and_files import make_dir,safely_recursively_copy_folder
from parser_related import get_bool_type

parser = argparse.ArgumentParser("collect measurments")
get_bool_type(parser)
parser.add_argument("data_root")
parser.add_argument("view_num",type=int)
parser.add_argument("exr_num",type=int)
parser.add_argument("if_use_cc",type='bool')



args = parser.parse_args()

if __name__ == "__main__":
    print("collecting raw measurements...")
    source_root = args.data_root
    target_root = args.data_root+"images/collected_measurements/"
    make_dir(target_root)
    target_data_root = target_root
    make_dir(target_data_root)
    for which_view in range(args.view_num):
        print("{}/{}".format(which_view,args.view_num))
        current_target_root = target_data_root+"{}/".format(which_view)
        make_dir(current_target_root)

        file_name = "cam00_data_{}_cc_compacted.bin".format(args.exr_num)
        shutil.copyfile(source_root+"{}/".format(which_view)+file_name,current_target_root+file_name)
        file_name = "cam00_index_cc.bin"
        shutil.copyfile(source_root+"{}/".format(which_view)+file_name,current_target_root+file_name)
    
    file_names = [
        "uvs.bin",
        "cam_indexes.bin"
    ]
    target_data_root = target_data_root+"images/"
    make_dir(target_data_root)
    for file_name in file_names:
        shutil.copyfile(source_root+"images/"+file_name,target_data_root+file_name)

    python_root = target_root+"python_files/"
    make_dir(python_root)

    safely_recursively_copy_folder("../utils/",target_root+"utils/")
    file_names = [
        "gather_measurements.py"
    ]
    for file_name in file_names:
        shutil.copyfile("./"+file_name,python_root+file_name)
    
    name = "run.sh"
    with open(python_root+name,"w",newline='\n') as pf:
        pf.write("#!/bin/bash\n")
        pf.write("DATA_ROOT=\"../\"\n")
        pf.write("VIEW_NUM={}\n".format(args.view_num)) 
        pf.write("MEASURMENT_LEN={}\n".format(args.exr_num//2))
        pf.write("IF_USE_CC={}\n".format(args.if_use_cc))
        pf.write('python gather_measurements.py $DATA_ROOT $VIEW_NUM $MEASURMENT_LEN $IF_USE_CC')