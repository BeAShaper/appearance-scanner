import numpy as np
import os
import sys
sys.path.append("../utils/")
from dir_folder_and_files import make_dir,safely_recursively_copy_folder
from parser_related import get_bool_type
import argparse
from subprocess import Popen
import shutil

parser = argparse.ArgumentParser(usage="split relighting brdf not slice")
get_bool_type(parser)
parser.add_argument("data_root")
parser.add_argument("diff_sample_num",type=int)
parser.add_argument("spec_sample_num",type=int)
parser.add_argument("sample_view_num",type=int)
parser.add_argument("thread_num",type=int)
parser.add_argument("server_num",type=int)
parser.add_argument("which_server",type=int)
parser.add_argument('is_for_server',type='bool')
parser.add_argument("fix_normal",type="bool")
parser.add_argument("m_len_perview",type=int)
parser.add_argument("tex_resolution",type=int)
parser.add_argument("--just_local_check_file",action="store_true")

args = parser.parse_args()

if __name__ == "__main__":
    data_root = args.data_root+"images/"# TODO
    if args.is_for_server:
        target_root = data_root+"data_for_server/"
    else:
        target_root = data_root+"data_for_local/"

    total_thread_num = args.thread_num*args.server_num

    if not args.just_local_check_file:
        if os.path.exists(target_root):
            shutil.rmtree(target_root)

        make_dir(target_root)

        shutil.copyfile(data_root+"texturemap_uv.bin",target_root+"texturemap_uv.bin")
        shutil.copyfile(data_root+"tangents_geo_global.bin",target_root+"tangents_geo_global.bin")
        shutil.copyfile(data_root+"normals_geo_global.bin",target_root+"normals_geo_global.bin")
        shutil.copyfile(data_root+"W.bin",target_root+"W.bin")

        # positions_data = np.fromfile(data_root+"selected_positions.bin",np.float32).reshape([-1,args.sample_view_num,3])
        # normals_data = np.fromfile(data_root+"selected_normals.bin",np.float32).reshape([-1,args.sample_view_num,3])
        # tangents_data = np.fromfile(data_root+"selected_tangents.bin",np.float32).reshape([-1,args.sample_view_num,3])
        # measurement_data = np.fromfile(data_root+"selected_measurements.bin",np.float32).reshape([-1,args.sample_view_num,args.m_len_perview])
        visible_view_num = np.fromfile(data_root+"visible_views_num.bin",np.int32).reshape([-1])
        pixel_num = visible_view_num.shape[0]
        print("[SPLITTER]pixel num:",pixel_num)
        
        pf_pd_slice_f = open(data_root+"diff_slice.bin")
        pf_ps_slice_f = open(data_root+"spec_slice.bin")
        pf_pos_f = open(data_root+"selected_positions.bin")
        pf_normal_f = open(data_root+"selected_normals.bin")
        pf_tangent_f = open(data_root+"selected_tangents.bin")
        pf_measurement_f = open(data_root+"selected_measurements.bin")

        num_per_thread = int(pixel_num//total_thread_num)
        
        pd_record_len = args.diff_sample_num * args.diff_sample_num * 6
        ps_record_len = args.spec_sample_num * args.spec_sample_num * 6

        ptr = 0
        for thread_id in range(total_thread_num):
            tmp_dir = target_root+"{}/".format(thread_id)
            make_dir(tmp_dir)

            cur_batchsize = num_per_thread if (not thread_id == total_thread_num-1) else (pixel_num-(total_thread_num-1)*num_per_thread)

            tmp_visible_view_num = visible_view_num[ptr:ptr+cur_batchsize]
            tmp_visible_view_num_batch = np.sum(tmp_visible_view_num)

            if not (thread_id == total_thread_num-1):
                tmp_pd_slices = np.fromfile(pf_pd_slice_f,np.float32,num_per_thread*pd_record_len).reshape([-1,pd_record_len])
                tmp_ps_slices = np.fromfile(pf_ps_slice_f,np.float32,num_per_thread*ps_record_len).reshape([-1,ps_record_len])
            else:
                tmp_pd_slices = np.fromfile(pf_pd_slice_f,np.float32).reshape([-1,pd_record_len])
                tmp_ps_slices = np.fromfile(pf_ps_slice_f,np.float32).reshape([-1,ps_record_len])
            
            assert tmp_pd_slices.shape[0] == tmp_ps_slices.shape[0]
            assert tmp_ps_slices.shape[0] == cur_batchsize
            
            
            tmp_positions = np.fromfile(pf_pos_f,np.float32,tmp_visible_view_num_batch*3)
            tmp_normals = np.fromfile(pf_normal_f,np.float32,tmp_visible_view_num_batch*3)
            tmp_tangents = np.fromfile(pf_tangent_f,np.float32,tmp_visible_view_num_batch*3)
            tmp_measurements = np.fromfile(pf_measurement_f,np.float32,tmp_visible_view_num_batch*args.m_len_perview)

            tmp_positions.astype(np.float32).tofile(tmp_dir+"position.bin")#(pointnum,sampleview_num,3)
            tmp_normals.astype(np.float32).tofile(tmp_dir+"normal_geo.bin")#(pointnum,sampleview_num,3)
            tmp_tangents.astype(np.float32).tofile(tmp_dir+"tangent_geo.bin")#(pointnum,sampleview_num,3)
            tmp_pd_slices.astype(np.float32).tofile(tmp_dir+"pd_slice.bin")
            tmp_ps_slices.astype(np.float32).tofile(tmp_dir+"ps_slice_normalized.bin")
            tmp_measurements.astype(np.float32).tofile(tmp_dir+"measurements.bin")#(pointnum,sampleview_num,m_lenperview)
            tmp_visible_view_num.astype(np.int32).tofile(tmp_dir+"visible_view_num.bin")#(pointnum,)
        
            print("thread:",thread_id," num:",tmp_ps_slices.shape[0])

            ptr += tmp_ps_slices.shape[0]
        for which_file,tmp_pf in enumerate([pf_pd_slice_f,pf_ps_slice_f,pf_pos_f,pf_normal_f,pf_tangent_f,pf_measurement_f]):
            remain_data = np.fromfile(tmp_pf,np.uint8)
            print("{} {}".format(which_file,len(remain_data)))
            if len(remain_data) > 0:
                print("this file is not at the end!",i)
                exit()
        pf_pd_slice_f.close()
        pf_ps_slice_f.close()
        pf_pos_f.close()
        pf_normal_f.close()
        pf_tangent_f.close()
        pf_measurement_f.close()
    ########################################
    #####prepare python files
    ########################################
    python_root = target_root+"python_files/"
    make_dir(python_root)

    if not args.just_local_check_file:
        for a_file in os.listdir(python_root):
            if os.path.isfile(python_root+a_file):
                os.remove(python_root+a_file)

        make_dir(python_root)


    shutil.copy("./tf_ggx_fittinger.py",  python_root+"tf_ggx_fittinger.py")
    shutil.copy("./tf_ggx_render.py",  python_root+"tf_ggx_render.py")
    shutil.copy("./tf_ggx_render_utils.py",  python_root+"tf_ggx_render_utils.py")
    shutil.copy("./fitted_result_gather.py",  python_root+"fitted_result_gather.py")
    shutil.copy("./fitting_master.py",  python_root+"fitting_master.py")
    shutil.copy("./ps_normalized_fitting_result_process.py",  python_root+"ps_normalized_fitting_result_process.py")
    shutil.copy("./fitting_measurement.py",  python_root+"fitting_measurement.py")
    shutil.copy("./trans_back_normal.py",  python_root+"trans_back_normal.py")

    ps_config_file_name = "tf_ggx_render_configs_cube_slice_{}x{}".format(args.spec_sample_num,args.spec_sample_num)
    pd_config_file_name = "tf_ggx_render_configs_cube_slice_{}x{}".format(args.diff_sample_num,args.diff_sample_num)

    if not args.just_local_check_file:
        utils_file_dir = target_root+"utils/"
        safely_recursively_copy_folder("../utils/",utils_file_dir)

        ps_config_file_dir = python_root+ps_config_file_name+"/"
        safely_recursively_copy_folder(ps_config_file_name+"/",ps_config_file_dir)

        pd_config_file_dir = python_root+pd_config_file_name+"/"
        safely_recursively_copy_folder(pd_config_file_name+"/",pd_config_file_dir)

        torchrenderer_file_dir = target_root+"torch_renderer/"
        safely_recursively_copy_folder("../torch_renderer/",torchrenderer_file_dir)

    name = "run.sh" if args.is_for_server else "run.bat"
    with open(python_root+name,"w",newline='\n') as pf:
        if args.is_for_server:
            pf.write("#!/bin/bash\n")
            pf.write("DATA_ROOT=\"../\"\n")
            pf.write("IF_DUMP=0\n") 
            pf.write("PS_CONFIG_FILE_NAME={}\n".format(ps_config_file_name))
            pf.write("PD_CONFIG_FILE_NAME={}\n".format(pd_config_file_name))
            pf.write("SPEC_SAMPLE_NUM={}\n".format(args.spec_sample_num))
            pf.write("DIFF_SAMPLE_NUM={}\n".format(args.diff_sample_num))
            pf.write("THREAD_NUM={}\n".format(args.thread_num))
            pf.write("SERVER_NUM={}\n".format(args.server_num))
            pf.write("WHICH_SERVER={}\n".format(args.which_server))
            pf.write("FIX_NORMAL={}\n".format(args.fix_normal))
            pf.write("IS_FOR_REFINE=False\n")
            pf.write("SAMPLE_VIEW_NUM={}\n".format(args.sample_view_num)) 
            pf.write("M_LEN_PERVIEW={}\n".format(args.m_len_perview)) 
            pf.write("TEX_RESOLUTION={}\n".format(args.tex_resolution)) 

            pf.write("python fitting_master.py $DATA_ROOT $THREAD_NUM $SERVER_NUM $WHICH_SERVER $IF_DUMP $PS_CONFIG_FILE_NAME $PD_CONFIG_FILE_NAME $FIX_NORMAL $IS_FOR_REFINE $DIFF_SAMPLE_NUM $SPEC_SAMPLE_NUM $SAMPLE_VIEW_NUM $M_LEN_PERVIEW $TEX_RESOLUTION\n")
        else:
            test_thread_id = 0
            pf.write("SET DATA_ROOT=\"../\"\n")
            pf.write("SET IF_DUMP=1\n") 
            pf.write("SET PS_CONFIG_FILE_NAME={}\n".format(ps_config_file_name))   
            pf.write("SET PD_CONFIG_FILE_NAME={}\n".format(pd_config_file_name))   
            pf.write("SET SPEC_SAMPLE_NUM={}\n".format(args.spec_sample_num))
            pf.write("SET DIFF_SAMPLE_NUM={}\n".format(args.diff_sample_num)) 
            pf.write("SET THREAD_NUM={}\n".format(args.thread_num))
            pf.write("SET SERVER_NUM={}\n".format(args.server_num))
            pf.write("SET WHICH_SERVER={}\n".format(args.which_server))   
            pf.write("SET FIX_NORMAL={}\n".format(args.fix_normal))
            pf.write("SET IS_FOR_REFINE=False\n") 
            pf.write("SET SAMPLE_VIEW_NUM={}\n".format(args.sample_view_num)) 
            pf.write("SET M_LEN_PERVIEW={}\n".format(args.m_len_perview)) 
            pf.write("SET TEX_RESOLUTION={}\n".format(args.tex_resolution)) 
            pf.write("python fitting_master.py %DATA_ROOT% %THREAD_NUM% %SERVER_NUM% %WHICH_SERVER% %IF_DUMP% %PS_CONFIG_FILE_NAME% %PD_CONFIG_FILE_NAME% %FIX_NORMAL% %IS_FOR_REFINE% %DIFF_SAMPLE_NUM% %SPEC_SAMPLE_NUM% %SAMPLE_VIEW_NUM% %M_LEN_PERVIEW% %TEX_RESOLUTION%\n")

        
