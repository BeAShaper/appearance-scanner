import numpy as np
import sys
sys.path.append('../utils/')
from parser_related import get_bool_type
import argparse

parser = argparse.ArgumentParser(usage="gather measurements from views")
get_bool_type(parser)
parser.add_argument('data_root',help="data root for test")
parser.add_argument('view_num',type=int,help="view num")
parser.add_argument('measurement_len',type=int,help="length of measurement")
parser.add_argument('if_use_cc',type='bool')
# parser.add_argument('folder_name',help="for ae or pd")

args = parser.parse_args()

if __name__ == "__main__":
    data_root = args.data_root
    cam_indexes = np.fromfile(data_root+"images/cam_indexes.bin",np.int32).reshape([-1,1]).astype(np.int32)
    uvs = np.fromfile(data_root+"images/uvs.bin",np.int32).reshape([-1,2]).astype(np.int32)
    # print(cam_indexes[259])
    # print(uvs[259])
    assert uvs.shape[0] == cam_indexes.shape[0]

    pixel_keys = np.concatenate([cam_indexes,uvs],axis=1)#[-1,3]
    # a = np.where((pixel_keys == (0, 1013,736)).all(axis=1))
    # print(a)
    # b = a[0][0]
    # print(b)
    # print(cam_indexes[b])
    # print(uvs[b])
    # exit()

    pixel_keys = tuple(map(tuple, pixel_keys))#((3))



    print("[GATHER] point num:",uvs.shape[0])

    measurements_collector = {}
    # uvs_collector = []

    print("loading measurements...")
    uvs_map = {}#{(cam,u,v):(cam,i)}
    # folder_name = args.folder_name
    # print("folder name:",folder_name)
    file_name = "cam00_data_32_cc_compacted.bin" if args.if_use_cc else "cam00_data_32_nocc_compacted.bin"
    idx_name = "cam00_index_cc.bin" if args.if_use_cc else "cam00_index_nocc.bin"
    for view_no in range(args.view_num):
        print("{}/{}".format(view_no+1,args.view_num))
        tmp_measurements = np.fromfile(data_root+"{}/{}".format(view_no,file_name),np.float32).reshape([-1,3,args.measurement_len])
        tmp_measurements = {(view_no,i):tmp_measurements[i] for i in range(tmp_measurements.shape[0])}
        measurements_collector={**measurements_collector,**tmp_measurements}

        with open(data_root+"{}/{}".format(view_no,idx_name),"rb") as pf:
            _ = np.fromfile(pf,np.int32,1)
            tmp_uvs = np.fromfile(pf,np.int32).reshape([-1,2])
        assert tmp_uvs.shape[0] == len(tmp_measurements)
        # uvs_collector.append(tmp_uvs)

        tmp_map = {(view_no,tmp_uvs[i][0],tmp_uvs[i][1]) : (view_no,i) for i in range(tmp_uvs.shape[0])}
        uvs_map = {**uvs_map,**tmp_map}
    
    print("computing wanted idxes...")
    # print(pixel_keys[0])
    # print(uvs_map[pixel_keys[0]])
    wanted_idxes = [uvs_map[key] for key in pixel_keys]#[(2)]
    # print(wanted_idxes[0])

    print("gathering...")
    measurements_collected = []
    for idx,key in enumerate(wanted_idxes):
        measurements_collected.append(measurements_collector[key])

    measurements_collected = np.asarray(measurements_collected)

    print(measurements_collected.shape)
    print("outputting...")
    measurements_collected.tofile(data_root+"images/measurements.bin")

    measurements_collected = np.mean(measurements_collected,axis=1)
    measurements_collected.tofile(data_root+"images/measurements_grey.bin")
