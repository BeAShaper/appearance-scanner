import numpy as np
import os
import cv2
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_root",default="F:/Turbot_freshmeat/main_results/Cheongsam_1024_origin/")
    parser.add_argument("--new_root",default="F:/Turbot_freshmeat/main_results/Cheongsam_1024/")
    # parser.add_argument("--origin_root",default="F:/Turbot_freshmeat/egypt_korean_1024_full/")
    # parser.add_argument("--new_root",default="F:/Turbot_freshmeat/egypt_korean_1024/")
    parser.add_argument("--ignore_slice",action="store_true")

    args = parser.parse_args()

    origin_root = args.origin_root
    new_root = args.new_root
    os.makedirs(new_root,exist_ok=True)

    check_map = cv2.imread(origin_root+"check_map.exr",cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)[:,:,0]
    normal_map = cv2.imread(origin_root+"normal_global.exr",cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
    valid_norm = np.linalg.norm(normal_map,axis=2)
    check_map = np.where(valid_norm > 0.0,check_map,np.zeros_like(check_map))

    check_ids = np.stack(np.where(check_map > 0.0),axis=1)[:,::-1]#(pixelnum,2) (x,y)
    print("filtered point num:",check_ids.shape[0])

    texturemap_uv = np.fromfile(origin_root+"texturemap_uv.bin",np.int32).reshape((-1,2))

    print("building uv to id mapping...")
    uv_id_map = {tuple(texturemap_uv[i]):i for i in range(texturemap_uv.shape[0])}
    # print(uv_id_map)
    print("done.")

    print("selecting id...")
    ids_collector = [uv_id_map[tuple(check_ids[i])] for i in range(check_ids.shape[0])]

    print("done")
    
    #################
    ###
    #################
    
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

    file_name = "W.bin"
    data = np.fromfile(origin_root+file_name,np.float32).reshape((-1))
    data.tofile(new_root+file_name)

    tmp_file_names = os.listdir(origin_root)
    for a_file_name in tmp_file_names:
        if "images" in a_file_name:
            img_folder_name = a_file_name
            break
    
    nn_origin_root = origin_root+img_folder_name+"/data_for_server/"
    nn_new_root = new_root+img_folder_name+"/data_for_server/"
    if os.path.exists(nn_new_root):
        shutil.rmtree(nn_new_root)
    os.makedirs(nn_new_root,exist_ok=True)

    foldername = "model"
    shutil.copytree(nn_origin_root+foldername,nn_new_root+foldername)
    foldername = "python_files"
    shutil.copytree(nn_origin_root+foldername,nn_new_root+foldername)

    nn_origin_root_data = nn_origin_root+"data/images/"
    nn_new_root_data = nn_new_root+"data/images/"
    os.makedirs(nn_new_root_data,exist_ok=True)

    file_name = "visible_views_num.bin"
    data = np.fromfile(nn_origin_root_data+file_name,np.int32).reshape((-1,1))
    data = data[ids_collector]
    data.tofile(nn_new_root_data+file_name)

    
    
    file_name = "selected_measurements.bin"
    pf_origin = open(nn_origin_root_data+file_name,"rb")
    pf_new = open(nn_new_root_data+file_name,"wb")
    file_name_n = "selected_normals.bin"
    pf_origin_n = open(nn_origin_root_data+file_name_n,"rb")
    pf_new_n = open(nn_new_root_data+file_name_n,"wb")
    file_name_t = "selected_tangents.bin"
    pf_origin_t = open(nn_origin_root_data+file_name_t,"rb")
    pf_new_t = open(nn_new_root_data+file_name_t,"wb")
    file_name_p = "selected_positions.bin"
    pf_origin_p = open(nn_origin_root_data+file_name_p,"rb")
    pf_new_p = open(nn_new_root_data+file_name_p,"wb")
    visible_views_num = np.fromfile(nn_origin_root_data+"visible_views_num.bin",np.int32).astype(np.int64)
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

    fitting_temp_root = new_root+"fitting_temp/"
    if os.path.exists(fitting_temp_root):
        shutil.rmtree(fitting_temp_root)
    shutil.copytree("../",fitting_temp_root,ignore=shutil.ignore_patterns(".git",".vscode"))