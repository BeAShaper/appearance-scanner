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
    cam_indexes = np.fromfile(data_root+"images/cam_ids.bin",np.int32).reshape([-1,1]).astype(np.int32)
    uvs = np.fromfile(data_root+"images/uvs.bin",np.int32).reshape([-1,2]).astype(np.int32)

    # print(cam_indexes[6:16])
    # print(uvs[6:16])
    # exit()

    assert uvs.shape[0] == cam_indexes.shape[0]

    pixel_keys = np.concatenate([cam_indexes,uvs],axis=1)#[-1,3]

    unique_row = np.unique(pixel_keys,axis=0)
    print(unique_row.shape)

    print(pixel_keys.shape[0])
    uv_view_collector=[]
    for which_view in range(args.view_num):
        tmp_idx = pixel_keys[:,0] == which_view
        uv_view_collector.append(tuple(map(tuple,uvs[tmp_idx])))

    pixel_keys = tuple(map(tuple, pixel_keys))#((3))
    print(len(pixel_keys))
    texture_idxes = {pixel_keys[i]: [] for i in range(len(pixel_keys))}
    for i in range(len(pixel_keys)):
        texture_idxes[pixel_keys[i]].append(i)
    print(len(texture_idxes))
    print("[GATHER] point num:",uvs.shape[0])

    measurements_collector = {}
    # uvs_collector = []

    print("loading measurements...")
    file_name = "cam00_data_32_cc_compacted.bin" if args.if_use_cc else "cam00_data_32_nocc_compacted.bin"
    collector = np.ones([uvs.shape[0],args.measurement_len,3],np.float32)
    idx_name = "cam00_index_cc.bin" if args.if_use_cc else "cam00_index_nocc.bin"
    
    for view_no in range(args.view_num):
        print("{}/{}".format(view_no+1,args.view_num))
        tmp_measurements = np.fromfile(data_root+"{}/{}".format(view_no,file_name),np.float32).reshape([-1,args.measurement_len,3])

        with open(data_root+"{}/{}".format(view_no,idx_name),"rb") as pf:
            _ = np.fromfile(pf,np.int32,1)
            tmp_uvs = np.fromfile(pf,np.int32).reshape([-1,2])
        assert tmp_uvs.shape[0] == len(tmp_measurements)
        # uvs_collector.append(tmp_uvs)

        tmp_map = {(tmp_uvs[i][0],tmp_uvs[i][1]) : i for i in range(tmp_uvs.shape[0])}

        for a_uv in uv_view_collector[view_no]:
            collector[texture_idxes[(view_no,a_uv[0],a_uv[1])]] = tmp_measurements[tmp_map[a_uv]]
    
    print("outputting...")
    collector.astype(np.float32).tofile(data_root+"images/measurements.bin")

    collector = np.mean(collector,axis=1)
    collector.tofile(data_root+"images/measurements_grey.bin")
    

    # old = np.fromfile(data_root+"images/measurements_oldmethod.bin",np.float32).reshape([-1,3,16])
    # new = np.fromfile(data_root+"images/measurements.bin",np.float32).reshape([-1,3,16])
    # assert old.shape[0] == new.shape[0]

    # want_to_check_idx = np.random.randint(0,old.shape[0],size=20)

    # for idx in want_to_check_idx:
    #     print("---------------------\nidx:{}\nold:".format(idx))
    #     print(old[idx])
    #     print("new:")
    #     print(new[idx])
    #     input()