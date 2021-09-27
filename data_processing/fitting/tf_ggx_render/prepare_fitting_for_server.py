import numpy as np
import argparse
import sys
import os
import shutil
sys.path.append("../utils/")
from dir_folder_and_files import make_dir,safely_recursively_copy_folder
from parser_related import get_bool_type
import glob

parser = argparse.ArgumentParser(usage="prepare data for server and fitting")
parser.add_argument("data_root")
parser.add_argument("sub_folder_name")
parser.add_argument("model_path")
parser.add_argument("model_name")
parser.add_argument("diff_sample_num",type=int)
parser.add_argument("spec_sample_num",type=int)
parser.add_argument("sample_view_num",type=int)

parser.add_argument("m_len",type=int)
parser.add_argument("lighting_pattern_num",type=int)
parser.add_argument("thread_num",type=int)
parser.add_argument("fix_normal")
parser.add_argument("texture_map_size",type=int)
parser.add_argument("--add_normal",action="store_true")

args = parser.parse_args()

if __name__ == "__main__":
    data_root = args.data_root+"undistort_feature/texture_{}/".format(args.texture_map_size)# TODO

    target_root = args.data_root+"images_{}/data_for_server/".format(args.sub_folder_name)
    make_dir(target_root)
    
    diff_slice_len = args.diff_sample_num * args.diff_sample_num * 6
    spec_slice_len = args.spec_sample_num * args.spec_sample_num * 6

    #####################
    ###1 measurements and others
    #####################
    print("copying measurements....")
    data_file_root = target_root+"data/"
    make_dir(data_file_root)
    data_file_root = data_file_root+"images/"
    make_dir(data_file_root)
    data_file_names = [
        "selected_measurements.bin",
        "selected_positions.bin",
        "selected_normals.bin",
        "selected_tangents.bin"
    ]
    for data_file_name in data_file_names:
        shutil.copyfile(data_root+args.sub_folder_name+"/"+data_file_name,data_file_root+data_file_name)
    print("done.")
    #####################
    ###2 models
    #####################
    print("copying model....")
    model_file_root = target_root+"model/"
    make_dir(model_file_root)
    shutil.copyfile(args.model_path+"/models/"+args.model_name,model_file_root+args.model_name)
    print("done.")
    ####################
    ###3 python files
    ###################
    print("copying python files....")
    python_file_root = target_root+"python_files/"
    make_dir(python_file_root)

    #-----------part1 ircheckers
    tmp_root = python_file_root+"PointNet/"
    safely_recursively_copy_folder("../PointNet",tmp_root)
    # shutil.copyfile(args.model_path+"/training_files/*.py",tmp_root)
    # current_list = glob.glob(os.path.join(args.model_path+"/training_files/",'*'))
 
    # for x in current_list:
    #     shutil.copy(x,tmp_root)

    #-----------part2 torch_renderer
    tmp_root = python_file_root+"torch_renderer/"
    safely_recursively_copy_folder("../torch_renderer",tmp_root)

    ##-----------part2 fittinger
    tmp_root = python_file_root+"tf_ggx_render/"
    safely_recursively_copy_folder("./",tmp_root)
   
    ##-----------part3 utils
    utils_file_dir = python_file_root+"utils/"
    safely_recursively_copy_folder("../utils/",utils_file_dir)

    print("done.")

##############three checker##############
    name = "run.sh"
    with open(python_file_root+"PointNet/test_files/"+name,"w",newline='\n') as pf:
        pf.write("#!/bin/bash\n")
        pf.write("DATA_ROOT=../../../data/images/\n")
        pf.write("MODEL_ROOT=../../../model/\n")
        pf.write("MODEL_FILE_NAME={}\n".format(args.model_name))
        pf.write("MEASUREMENT_LEN={}\n".format(args.m_len))
        pf.write("LIGHTING_PATTERN_NUM={}\n".format(args.lighting_pattern_num))
        pf.write("SPEC_SAMPLE_NUM={}\n".format(args.spec_sample_num))
        pf.write("DIFF_SAMPLE_NUM={}\n".format(args.diff_sample_num))
        pf.write("SAMPLE_VIEW_NUM={}\n".format(args.sample_view_num))

        pf.write("python infer_slice.py $DATA_ROOT $MODEL_ROOT $MODEL_FILE_NAME $DIFF_SAMPLE_NUM $SPEC_SAMPLE_NUM $MEASUREMENT_LEN $LIGHTING_PATTERN_NUM $SAMPLE_VIEW_NUM --batch_size 500")
        pf.write(" --add_normal\n") if args.add_normal else pf.write("\n")
##############splitter##############
    name="split.sh"
    with open(python_file_root+"tf_ggx_render/"+name,"w",newline='\n') as pf:
        pf.write("#!/bin/bash\n")
        pf.write("DATA_ROOT=\"../\"\n")
        pf.write("SAMPLE_VIEW_NUM={}\n".format(args.sample_view_num))
        pf.write("SPEC_SAMPLE_NUM={}\n".format(args.spec_sample_num))
        pf.write("DIFF_SAMPLE_NUM={}\n".format(args.diff_sample_num))
        pf.write("THREAD_NUM={}\n".format(args.thread_num))
        pf.write("IS_FOR_SERVER=True\n")
        pf.write("M_LEN_PERVIEW={}\n".format(args.m_len*3))

        pf.write("IF_FIX_NORMAL={}\n".format(args.fix_normal))

        pf.write("python split_data_and_prepare_for_server.py ../../data/ $DIFF_SAMPLE_NUM $SPEC_SAMPLE_NUM $SAMPLE_VIEW_NUM $THREAD_NUM $IS_FOR_SERVER $IF_FIX_NORMAL $M_LEN_PERVIEW")
##############splitter##############

    name = "run.bat"
    with open(python_file_root+"PointNet/test_files/"+name,"w",newline='\n') as pf:
        pf.write("SET DATA_ROOT=../../../data/images/\n")
        pf.write("SET MODEL_ROOT=../../../model/\n")
        pf.write("SET MODEL_FILE_NAME={}\n".format(args.model_name))
        pf.write("SET MEASUREMENT_LEN={}\n".format(args.m_len))
        pf.write("SET LIGHTING_PATTERN_NUM={}\n".format(args.lighting_pattern_num))
        pf.write("SET SPEC_SAMPLE_NUM={}\n".format(args.spec_sample_num))
        pf.write("SET DIFF_SAMPLE_NUM={}\n".format(args.diff_sample_num))
        pf.write("SET SAMPLE_VIEW_NUM={}\n".format(args.sample_view_num))

        pf.write("python infer_slice.py %DATA_ROOT% %MODEL_ROOT% %MODEL_FILE_NAME% %DIFF_SAMPLE_NUM% %SPEC_SAMPLE_NUM% %MEASUREMENT_LEN% %LIGHTING_PATTERN_NUM% %SAMPLE_VIEW_NUM%")
        pf.write(" --add_normal\n") if args.add_normal else pf.write("\n")

##############splitter##############
    name="split.bat"
    with open(python_file_root+"tf_ggx_render/"+name,"w",newline='\n') as pf:
        pf.write("SET DATA_ROOT=\"../\"\n")
        pf.write("SET SAMPLE_VIEW_NUM={}\n".format(args.sample_view_num))
        pf.write("SET SPEC_SAMPLE_NUM={}\n".format(args.spec_sample_num))
        pf.write("SET DIFF_SAMPLE_NUM={}\n".format(args.diff_sample_num))
        pf.write("SET THREAD_NUM={}\n".format(args.thread_num))
        pf.write("SET IS_FOR_SERVER=False\n")

        pf.write("IF_FIX_NORMAL={}\n".format(args.fix_normal))

        pf.write("python split_data_and_prepare_for_server.py ../../data/ %DIFF_SAMPLE_NUM% %SPEC_SAMPLE_NUM% %SAMPLE_VIEW_NUM% %THREAD_NUM% %IS_FOR_SERVER% %IF_FIX_NORMAL%")

