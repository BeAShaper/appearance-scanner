import numpy as np
import sys
import os
import time
sys.path.append("../utils/")
from subprocess import Popen
from dir_folder_and_files import make_dir
from parser_related import get_bool_type
import argparse
import GPUtil
import queue
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import warnings
warnings.filterwarnings('ignore')

exclude_gpu_list = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="fitting using multi card")
    get_bool_type(parser)
    parser.add_argument("data_root")
    parser.add_argument("thread_num",type=int)
    parser.add_argument("server_num",type=int)
    parser.add_argument("which_server",type=int)
    parser.add_argument("if_dump")
    parser.add_argument("ps_config_name")
    parser.add_argument("pd_config_name")
    parser.add_argument("fix_normal",type="bool")
    parser.add_argument("is_for_refine",type="bool")
    parser.add_argument("diff_sample_num",type=int)
    parser.add_argument("spec_sample_num",type=int)
    parser.add_argument("sample_view_num",type=int)
    parser.add_argument("m_len_perview",type=int,choices=[3])
    parser.add_argument("tex_resolution",type=int)

    args = parser.parse_args()

    deviceIDs = GPUtil.getAvailable(limit = 5,excludeID=exclude_gpu_list,maxLoad=1.0,maxMemory=1.0)
    gpu_num = len(deviceIDs)
    print("available gpu num:",gpu_num)
    # time.sleep(2)
    pool = []
    p_log_f = open(args.data_root+"fitting_log_all.txt","w",buffering=1)
    ##################################
    #step 1 fitting grey lumitexel
    ##################################
    for which_thread in range(args.which_server*args.thread_num,(args.which_server+1)*args.thread_num):
        print("starting thread:{}".format(which_thread))
        my_env = os.environ.copy()
        my_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        my_env["CUDA_VISIBLE_DEVICES"] = "{}".format(deviceIDs[which_thread%gpu_num])

        this_args = [
                "python",
                "tf_ggx_fittinger.py",
                "{}".format(which_thread),
                args.data_root,
                args.pd_config_name,
                args.ps_config_name
        ]
        if int(args.if_dump) == 1:
            this_args.append("--need_dump")
        if not args.fix_normal:
            this_args.append("--free_spec_normal")

        theProcess = Popen(
            this_args,
            stdout = open("thread_log_{}.txt".format(which_thread),"a"),
            env=my_env
        )
        # tmp_exit_code = theProcess.wait()
        # print("thread:{} exit code:{}".format(which_thread,tmp_exit_code))
        # exit()
        pool.append(theProcess)
    exit_codes = [ p.wait() for p in pool ]
    print("fitting exit codes:",exit_codes)
    p_log_f.write("fitting exit codes:{}\n".format(exit_codes))

    ##################################
    #step 2 fitting rhod rhos
    ##################################
    thread_per_gpu = [0]*len(deviceIDs)
    q = queue.Queue()
    [q.put(i) for i in range(args.which_server*args.thread_num,(args.which_server+1)*args.thread_num)]
    try_num = [0]*(args.thread_num*args.server_num)
    pool = []
    while not q.empty():
        which_thread = q.get()
        try_num[which_thread] = try_num[which_thread]+1
        print("starting thread:{}".format(which_thread))
        which_gpu = which_thread%gpu_num
        my_env = os.environ.copy()

        this_args = [
                "python",
                "fitting_measurement.py",
                "--data_for_server_root",
                "../",
                "--sample_view_num",
                "{}".format(args.sample_view_num),
                "--m_len_perview",
                "{}".format(args.m_len_perview),
                "--thread_ids",
                "{}".format(which_thread),
                "--gpu_id",
                "{}".format(deviceIDs[which_gpu]),
                "--tex_resolution",
                "{}".format(args.tex_resolution),
                "--total_thread_num",
                "{}".format(args.thread_num*args.server_num)
        ]
        if int(args.if_dump) == 1:
            this_args.append("--need_dump")
        theProcess = Popen(
            this_args,
            stdout = open("thread_log_{}.txt".format(which_thread),"a"),
            env=my_env
        )
        # tmp_exit_code = theProcess.wait()
        # print("thread:{} exit code:{}".format(which_thread,tmp_exit_code))
        # exit()
        pool.append(theProcess)
        thread_per_gpu[which_gpu] += 1 
        if thread_per_gpu.count(2) == len(thread_per_gpu):
            exit_codes = [p.wait() for p in pool]
            print("fitting rhod rhos exit codes:",exit_codes)
            for i in range(len(exit_codes)):
                crash_thread_id = int(pool[i].args[9])
                if exit_codes[i] != 0 and try_num[crash_thread_id] < 2:
                    print("thread:{} crashed!".format(crash_thread_id))
                    q.put(crash_thread_id)
            pool = []
            thread_per_gpu = [0]*len(deviceIDs)
    exit_codes = [ p.wait() for p in pool ]
    print("fitting rhod rhos exit codes:",exit_codes)
    p_log_f.write("fitting rhod rhos exit codes:{}\n".format(exit_codes))

    ##################################
    #step 3 transback normals
    ##################################
    if args.which_server == 0:
        with open("gather.sh","w") as pf:
            pf.write("#!/bin/bash\n")
            this_args = [
                "python",
                "trans_back_normal.py",
                "--data_root",
                "../",
                "--thread_num",
                "{}".format(args.thread_num*args.server_num),
                "--tex_resolution",
                "{}".format(args.tex_resolution)
            ]
            pf.write(" ".join(this_args))

    if args.server_num == 1:
        this_args = [
            "python",
            "trans_back_normal.py",
            "--data_root",
            "../",
            "--thread_num",
            "{}".format(args.thread_num),
            "--tex_resolution",
            "{}".format(args.tex_resolution)
        ]
        theProcess = Popen(
            this_args
        )
        tmp_exit_code = theProcess.wait()
        print("thread:{} exit code:{}".format(which_thread,tmp_exit_code))

    p_log_f.close()