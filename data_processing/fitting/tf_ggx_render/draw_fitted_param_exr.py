import numpy as np
import cv2
import sys
sys.path.append("../utils/")
import argparse
from parser_related import get_bool_type
from subprocess import Popen

parser = argparse.ArgumentParser(usage="draw_fitted_param exr")
get_bool_type(parser)
parser.add_argument("data_root")
parser.add_argument("is_for_server",type="bool")
parser.add_argument("pd_only",type="bool")
parser.add_argument("texture_map_size",type=int)
args = parser.parse_args()

param_len = 12

if __name__ == "__main__":
    root = "data_for_server" if args.is_for_server else "data_for_local"
    
    valids = np.fromfile(args.data_root+"images/texturemap_uv.bin",np.int32).reshape([-1,2])

    
    total_num = valids.shape[0]
    # wanted_pos = valids>0

    param_ptr = 0
    if not args.pd_only:
    #normal
        fitted_params = np.fromfile(args.data_root+"images/{}/gathered_all/params_gathered_gloabal_normal.bin".format(root),np.float32).reshape([-1,param_len])
        # assert np.count_nonzero(valids>0) == fitted_params.shape[0]
        param_num = 3
        param_name = "normal"
        tmp_normal = np.zeros([args.texture_map_size,args.texture_map_size,3],np.float32)
        fitted_normals = fitted_params[:,param_ptr:param_ptr+param_num]
        fitted_normals = fitted_normals*0.5+0.5

        tmp_normal[valids[:,1],valids[:,0]] = fitted_normals

        tmp_normal.tofile(args.data_root+"images/{}.bin".format(param_name))
        tmp_normal = tmp_normal[:,:,::-1]
        cv2.imwrite(args.data_root+"images/{}.exr".format(param_name),tmp_normal)

        # theProcess = Popen(
        #     [
        #         "../exe/float_img_to_exr.exe",
        #         args.data_root+"images/",
        #         "{}.bin".format(param_name),
        #         "{}_fitted.exr".format(param_name),
        #         "{}".format(args.texture_map_size),
        #         "{}".format(args.texture_map_size),
        #         "3"
        #     ]
        # )
        # print("exit code:",theProcess.wait())
        # param_ptr+=param_num

    #t
        param_num = 1
        param_name = "t"
        tmp_normal = np.zeros([args.texture_map_size,args.texture_map_size,3],np.float32)
        fitted_normals = fitted_params[:,param_ptr:param_ptr+param_num]
        fitted_normals = np.concatenate([
            fitted_normals,
            np.zeros([fitted_normals.shape[0],3-param_num])
        ],axis=1)
        tmp_normal[wanted_pos] = fitted_normals

        tmp_normal.tofile(args.data_root+"images/{}.bin".format(param_name))

        theProcess = Popen(
            [
                "../exe/float_img_to_exr.exe",
                args.data_root+"images/",
                "{}.bin".format(param_name),
                "{}_fitted.exr".format(param_name),
                "{}".format(args.texture_map_size),
                "{}".format(args.texture_map_size),
                "3"
            ]
        )
        print("exit code:",theProcess.wait())
        param_ptr+=param_num

    # axay
        param_num = 2
        param_name = "axay"
        tmp_normal = np.zeros([args.texture_map_size,args.texture_map_size,3],np.float32)
        fitted_normals = fitted_params[:,param_ptr:param_ptr+param_num]
        log_min = 0.006
        log_max = 0.503
        fitted_axay_forvisualize = fitted_normals.copy()
        fitted_axay_forvisualize = (fitted_axay_forvisualize-log_min)/(log_max-log_min)
        # fitted_axay_forvisualize = np.concatenate([fitted_axay_forvisualize,fitted_normals[:,[1]]],axis=-1)
        fitted_normals = np.concatenate([
            fitted_normals,
            np.zeros([fitted_normals.shape[0],3-param_num])
        ],axis=1)
        tmp_normal[wanted_pos] = fitted_normals

        fitted_axay_forvisualize = np.concatenate([
            fitted_axay_forvisualize,
            np.zeros([fitted_normals.shape[0],3-param_num])
        ],axis=1)
        

        tmp_normal.tofile(args.data_root+"images/{}.bin".format(param_name))
        tmp_normal[wanted_pos] = fitted_axay_forvisualize
        tmp_normal.astype(np.float32).tofile(args.data_root+"images/axay_visualize.bin")

        theProcess = Popen(
            [
                "../exe/float_img_to_exr.exe",
                args.data_root+"images/",
                "{}.bin".format(param_name),
                "{}_fitted.exr".format(param_name),
                "{}".format(args.texture_map_size),
                "{}".format(args.texture_map_size),
                "3"
            ]
        )
        print("exit code:",theProcess.wait())
        theProcess = Popen(
            [
                "../exe/float_img_to_exr.exe",
                args.data_root+"images/",
                "axay_visualize.bin",
                "axay_visualize_fitted.exr",
                "{}".format(args.texture_map_size),
                "{}".format(args.texture_map_size),
                "3"
            ]
        )
        print("exit code:",theProcess.wait())
        param_ptr+=param_num

    #pd
        param_num = 3
        param_name = "pd"
        tmp_normal = np.zeros([total_num,3],np.float32)
        fitted_normals = fitted_params[:,param_ptr:param_ptr+param_num]
        # fitted_normals = np.fromfile(args.data_root+"images/cam00_data.bin",np.float32).reshape([-1,3])
        fitted_normals = np.concatenate([
            fitted_normals,
            np.zeros([fitted_normals.shape[0],3-param_num])
        ],axis=1)
        tmp_normal[wanted_pos] = fitted_normals

        tmp_normal.tofile(args.data_root+"images/{}.bin".format(param_name))

        theProcess = Popen(
            [
                "../exe/float_img_to_exr.exe",
                args.data_root+"images/",
                "{}.bin".format(param_name),
                "{}_fitted.exr".format(param_name),
                "{}".format(args.texture_map_size),
                "{}".format(args.texture_map_size),
                "3"
            ]
        )
        print("exit code:",theProcess.wait())
        param_ptr+=param_num

    #ps
        param_num = 3
        param_name = "ps"
        tmp_normal = np.zeros([total_num,3],np.float32)
        fitted_normals = fitted_params[:,param_ptr:param_ptr+param_num]
        fitted_normals = np.concatenate([
            fitted_normals,
            np.zeros([fitted_normals.shape[0],3-param_num])
        ],axis=1)
        tmp_normal[wanted_pos] = fitted_normals

        tmp_normal.tofile(args.data_root+"images/{}.bin".format(param_name))

        theProcess = Popen(
            [
                "../exe/float_img_to_exr.exe",
                args.data_root+"images/",
                "{}.bin".format(param_name),
                "{}_fitted.exr".format(param_name),
                "{}".format(args.texture_map_size),
                "{}".format(args.texture_map_size),
                "3"
            ]
        )
        print("exit code:",theProcess.wait())
        param_ptr+=param_num
#ps_length
        # tmp_normal = np.zeros([total_num,3],np.float32)
        # param_name = "ps_length"
        # fitted_normals = np.fromfile(args.data_root+"images/ps_length_predicted.bin",np.float32).reshape([-1,3])
        # tmp_normal[wanted_pos] = fitted_normals

        # tmp_normal.tofile(args.data_root+"images/{}.bin".format(param_name))

        # theProcess = Popen(
        #     [
        #         "../exe/float_img_to_exr.exe",
        #         args.data_root+"images/",
        #         "{}.bin".format(param_name),
        #         "{}_fitted.exr".format(param_name),
        #         "{}".format(args.texture_map_size),
        #         "{}".format(args.texture_map_size),
        #         "3"
        #     ]
        # )
        # print("exit code:",theProcess.wait())

#global tangent
        tmp_normal = np.zeros([total_num,3],np.float32)
        param_name = "tangent_predicted"
        fitted_normals = np.fromfile(args.data_root+"images/data_for_server/gathered_all/gathered_global_tangent.bin",np.float32).reshape([-1,3])
        tmp_normal[wanted_pos] = fitted_normals
        fitted_normals = fitted_normals*0.5+0.5

        tmp_normal.tofile(args.data_root+"images/{}.bin".format(param_name))

        theProcess = Popen(
            [
                "../exe/float_img_to_exr.exe",
                args.data_root+"images/",
                "{}.bin".format(param_name),
                "{}_fitted.exr".format(param_name),
                "{}".format(args.texture_map_size),
                "{}".format(args.texture_map_size),
                "3"
            ]
        )
        print("exit code:",theProcess.wait())

    else:
        param_num = 3
        param_name = "pd"
        tmp_normal = np.zeros([total_num,3],np.float32)
        fitted_normals = np.fromfile(args.data_root+"images/pd_predicted.bin",np.float32).reshape([-1,3])
        fitted_normals = np.concatenate([
            fitted_normals,
            np.zeros([fitted_normals.shape[0],3-param_num])
        ],axis=1)
        tmp_normal[wanted_pos] = fitted_normals

        tmp_normal.tofile(args.data_root+"images/{}.bin".format(param_name))

        theProcess = Popen(
            [
                "../exe/float_img_to_exr.exe",
                args.data_root+"images/",
                "{}.bin".format(param_name),
                "{}_fitted_special.exr".format(param_name),
                "{}".format(args.texture_map_size),
                "{}".format(args.texture_map_size),
                "3"
            ]
        )
        print("exit code:",theProcess.wait())
        param_ptr+=param_num


    index = np.fromfile(args.data_root+"images/cam_indexes_full.bin",np.float32).reshape([-1,3])
    index = np.around(index).astype(np.int32)
    index = index[:,0]
    tmp_normal = np.zeros([total_num,3],np.float32)
    total_view_num = 24

    def get_random_color():
        return np.random.rand(3)

    for which_view in range(total_view_num):
        tmp_normal[index == which_view] = get_random_color()
    
    tmp_normal.tofile(args.data_root+"images/cam_index_random.bin")

    theProcess = Popen(
        [
            "../exe/float_img_to_exr.exe",
            args.data_root+"images/",
            "cam_index_random.bin",
            "cam_index_random.exr",
            "{}".format(args.texture_map_size),
            "{}".format(args.texture_map_size),
            "3"
        ]
    )
    print("exit code:",theProcess.wait())
