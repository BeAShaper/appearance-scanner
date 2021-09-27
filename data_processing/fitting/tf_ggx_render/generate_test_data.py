import numpy as np
import os
import cv2
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_root",default="F:/Turbot_freshmeat/main_results/1_18/box_check/data/images/")
    parser.add_argument("--new_root",default="F:/Turbot_freshmeat/main_results/1_18/box_check/data/images_check/")
    parser.add_argument("--ignore_slice",action="store_true")

    args = parser.parse_args()

    origin_root = args.origin_root
    new_root = args.new_root
    os.makedirs(new_root,exist_ok=True)

    check_map = cv2.imread(origin_root+"check_map.exr",cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)[:,:,0]

    check_ids = np.stack(np.where(check_map > 0.0),axis=1)[:,::-1]#(pixelnum,2) (x,y)
    print("filtered point num:",check_ids.shape[0])

    texturemap_uv = np.fromfile(origin_root+"texturemap_uv.bin",np.int32).reshape((-1,2))

    ids_collector = []
    for which_check_sample in range(check_ids.shape[0]):
        if which_check_sample % 1000 == 0:
            print("{}/{}".format(which_check_sample,check_ids.shape[0]))
        tmp_id = np.where((texturemap_uv == check_ids[which_check_sample]).all(axis=1))[0]
        if len(tmp_id)>0:
            ids_collector.append(tmp_id[0] )
    
    ids_collector = np.array(ids_collector)
    
    #################
    ###
    #################
    if not args.ignore_slice:
        file_name = "diff_slice.bin"
        data = np.fromfile(origin_root+file_name,np.float32).reshape((-1,384))
        data = data[ids_collector]
        data.tofile(new_root+file_name)

        file_name = "spec_slice.bin"
        data = np.fromfile(origin_root+file_name,np.float32).reshape((-1,6144))
        data = data[ids_collector]
        data.tofile(new_root+file_name)
    
    file_name = "normals_geo_global.bin"
    data = np.fromfile(origin_root+file_name,np.float32).reshape((-1,3))
    data = data[ids_collector]
    data.tofile(new_root+file_name)
    
    file_name = "tangents_geo_global.bin"
    data = np.fromfile(origin_root+file_name,np.float32).reshape((-1,3))
    data = data[ids_collector]
    data.tofile(new_root+file_name)

    file_name = "texturemap_uv.bin"
    data = np.fromfile(origin_root+file_name,np.int32).reshape((-1,2))
    data = data[ids_collector]
    data.tofile(new_root+file_name)


    file_name = "visible_views_num.bin"
    data = np.fromfile(origin_root+file_name,np.int32).reshape((-1,1))
    data = data[ids_collector]
    data.tofile(new_root+file_name)

    file_name = "W.bin"
    data = np.fromfile(origin_root+file_name,np.float32).reshape((-1))
    data.tofile(new_root+file_name)
    
    file_name = "selected_measurements.bin"
    pf_origin = open(origin_root+file_name,"rb")
    pf_new = open(new_root+file_name,"wb")
    file_name_n = "selected_normals.bin"
    pf_origin_n = open(origin_root+file_name_n,"rb")
    pf_new_n = open(new_root+file_name_n,"wb")
    file_name_t = "selected_tangents.bin"
    pf_origin_t = open(origin_root+file_name_t,"rb")
    pf_new_t = open(new_root+file_name_t,"wb")
    file_name_p = "selected_positions.bin"
    pf_origin_p = open(origin_root+file_name_p,"rb")
    pf_new_p = open(new_root+file_name_p,"wb")
    visible_views_num = np.fromfile(origin_root+"visible_views_num.bin",np.int32)
    for which_sample in ids_collector:
        all_visible_num_before = np.sum(visible_views_num[:which_sample])
        this_visible_num = visible_views_num[which_sample]
        #m
        pf_origin.seek(all_visible_num_before*3*4,0)
        tmp_data = np.fromfile(pf_origin,np.float32,this_visible_num*3)
        tmp_data.tofile(pf_new)
        #n
        pf_origin_n.seek(all_visible_num_before*3*4,0)
        tmp_data = np.fromfile(pf_origin_n,np.float32,this_visible_num*3)
        tmp_data.tofile(pf_new_n)
        #t
        pf_origin_t.seek(all_visible_num_before*3*4,0)
        tmp_data = np.fromfile(pf_origin_t,np.float32,this_visible_num*3)
        tmp_data.tofile(pf_new_t)
        #p
        pf_origin_p.seek(all_visible_num_before*3*4,0)
        tmp_data = np.fromfile(pf_origin_p,np.float32,this_visible_num*3)
        tmp_data.tofile(pf_new_p)

    pf_origin.close()
    pf_new.close()
    pf_origin_n.close()
    pf_new_n.close()
    pf_origin_t.close()
    pf_new_t.close()
    pf_origin_p.close()
    pf_new_p.close()
