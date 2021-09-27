import numpy as np
import argparse
import os
from subprocess import Popen

data_roots = [
    "F:/Turbot_freshmeat/main_results/1_18/cloth3_1024/",
    "F:/Turbot_freshmeat/main_results/1_18/amiibo_1024/",
    "F:/Turbot_freshmeat/main_results/1_18/bottle2_1024/",
    "F:/Turbot_freshmeat/main_results/1_18/egypt2_1024/",
    "F:/Turbot_freshmeat/main_results/1_18/iron_1024/",
    "F:/Turbot_freshmeat/main_results/1_18/rabbit2_1024/",
    "F:/Turbot_freshmeat/main_results/1_18/mask2_1024/"
]

for a_data_root in data_roots:
    for which_axis in [0,1,2]:
        tmp_args = [
            "python",
            "sorting_tangent.py",
            a_data_root,
            "{}".format(which_axis)
        ]
        the_process = Popen(tmp_args)
        exit_code = the_process.wait()
        if exit_code != 0:
            print("error")
            exit()
        
        tmp_args.append("--if_neg")
        the_process = Popen(tmp_args)
        exit_code = the_process.wait()
        if exit_code != 0:
            print("error")
            exit()