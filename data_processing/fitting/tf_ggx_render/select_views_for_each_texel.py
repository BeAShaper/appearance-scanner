import numpy as np
import argparse
import sys
import os
import shutil
sys.path.append("../utils/")
from dir_folder_and_files import make_dir,safely_recursively_copy_folder
from parser_related import get_bool_type
sys.path.append("../torch_renderer/")
import torch_render
import cv2
import open3d as o3d
def farthest_point_sample(xyz, npoint):
    N,_ = xyz.shape
    
    centroids = np.ones(npoint,np.int32) *-1
    distance = np.ones( N,np.float32) * 1e10
    farthest = int(np.random.rand() * N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return centroids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="select N views")
    parser.add_argument("data_root")
    parser.add_argument("sub_root")
    parser.add_argument("lighting_pattern_num",type=int)
    parser.add_argument("measurement_len",type=int)
    parser.add_argument("scalar",type=float)
    parser.add_argument("select_view_num",type=int)
    parser.add_argument("texture_map_size",type=int)
    parser.add_argument("config_file",type=str)
    parser.add_argument("model_path",type=str)

    args = parser.parse_args()

    np.random.seed(666)
    data_root = args.data_root+"undistort_feature/texture_{}/".format(args.texture_map_size)
    save_root = data_root + args.sub_root + "/"
    make_dir(save_root)

    file = open(args.data_root+"undistort_feature/images.txt")
    for i in range(4):
        line = file.readline()
    s = [i for i in line if str.isdigit(i)][:3]
    sample_view_num = int(''.join(s))
    print(sample_view_num)
    
    # images = os.listdir(args.data_root + "full_on_obj_udt/images_0/")
    # regi_cam_id_goodboy = []
    # for img in images:
    #     regi_cam_id_goodboy.append(int(img[13:-6]))
    # print(regi_cam_id_goodboy)
    

    all_positions = np.fromfile(data_root+"positions_fulllocalview.bin",np.float32).reshape([-1,sample_view_num,3])
    all_normals = np.fromfile(data_root+"normals_geo_fulllocalview.bin",np.float32).reshape([-1,sample_view_num,3])
    all_tangents = np.fromfile(data_root+"tangents_geo_fulllocalview.bin",np.float32).reshape([-1,sample_view_num,3])
    all_ndotv = np.fromfile(data_root+"ndotv_fulllocalview.bin",np.float32).reshape([-1,sample_view_num])
    
    cam_indexes = np.fromfile(data_root+"cam_ids.bin",np.int32).reshape([-1,sample_view_num,1]).astype(np.int32)
    regi_cam_id = np.fromfile(data_root+"regi_cam_id.bin",np.int32).astype(np.int32)
    regi_cam_id = [regi_cam_id[i] for i in range(regi_cam_id.shape[0])]

    uvs = np.fromfile(data_root+"uvs.bin",np.int32).reshape([-1,sample_view_num,2]).astype(np.int32)
    tex_uvs = np.fromfile(data_root+"texturemap_uv.bin",np.int32).reshape([-1,2])
    
    with open(args.model_path+"/models/0/maxs.bin","rb") as pf:
        maxs = np.fromfile(pf,np.float32)

    cam_positions = np.fromfile(data_root+"cam_pos.bin",np.float32).reshape([sample_view_num,3])
    camera_extrinsic = np.fromfile(args.config_file,np.float32)
    cameraR = camera_extrinsic[:9].reshape(3,3)
    cameraT = camera_extrinsic[9:].reshape(1,3)

    texel_num = all_positions.shape[0]
    # texel_num = 1

    


    model_view_num = args.select_view_num // args.lighting_pattern_num
    surplus_view_num = model_view_num + 8
    
    reselect_views = np.zeros([texel_num, args.select_view_num])
    reselected_positions = np.zeros([texel_num,args.select_view_num, 3])
    reselected_measurements = np.zeros([texel_num,args.select_view_num, 3])
    reselected_normals = np.zeros([texel_num, args.select_view_num, 3])
    reselected_tangents = np.zeros([texel_num, args.select_view_num, 3])

    for which_pattern in range(args.lighting_pattern_num):
        select_views = np.zeros([texel_num, surplus_view_num])
        selected_positions = np.zeros([texel_num,surplus_view_num, 3])
        selected_normals = np.zeros([texel_num, surplus_view_num, 3])
        selected_tangents = np.zeros([texel_num, surplus_view_num, 3])
        select_views = np.zeros([texel_num, surplus_view_num])
        select_cams = np.ones([texel_num, surplus_view_num]) * -1
        select_uvs = np.zeros([texel_num, surplus_view_num,2],np.int32)
        visible_view_num = np.zeros([texel_num],np.uint8)

        tmp_cam_indexes = cam_indexes.copy()
        # print(tmp_cam_indexes[0].reshape([1,-1]))
        for i in range(tmp_cam_indexes.shape[1]):
            if regi_cam_id[i] % args.lighting_pattern_num != which_pattern:
                tmp_cam_indexes[:,i] = -1

        for which_point in range(texel_num): 
            if which_point % 10000 == 0:
                print(which_point)
            tmp_visible_cam = np.where(tmp_cam_indexes[which_point] != -1)[0]  # all visible views 
            tmp_visible_cam_ndotv = np.where(all_ndotv[which_point] < 0.3)[0]
            tmp_visible_cam = np.array(list(set(tmp_visible_cam)-set(tmp_visible_cam_ndotv)))
            
            visible_view_num[which_point] = tmp_visible_cam.shape[0]
            
            if tmp_visible_cam.shape[0] == 0:
                continue
            tmp_selected_cam_idx = farthest_point_sample(cam_positions[tmp_visible_cam],surplus_view_num)
            tmp_selected_cam = tmp_visible_cam[tmp_selected_cam_idx]
            
            # for which_selected_cam in list(tmp_selected_cam):
            #     if np.array(regi_cam_id)[which_selected_cam] in regi_cam_id_goodboy:
            #         first_goodboy = which_selected_cam
            #         break
                
            # for idx,which_selected_cam in enumerate(list(tmp_selected_cam)):
            #     if np.array(regi_cam_id)[which_selected_cam] not in regi_cam_id_goodboy:
            #         tmp_selected_cam[idx] = first_goodboy


            tmp_selected_cam = np.sort(tmp_selected_cam)
            select_views[which_point] = tmp_selected_cam
            # print(tmp_selected_cam.reshape([1,-1]))
            selected_positions[which_point] = all_positions[which_point,tmp_selected_cam]
            selected_normals[which_point] = all_normals[which_point,tmp_selected_cam]
            selected_tangents[which_point] = all_tangents[which_point,tmp_selected_cam]
            select_cams[which_point] = np.array(regi_cam_id)[tmp_selected_cam]
            select_uvs[which_point] = uvs[which_point,tmp_selected_cam]


           
        
        select_cams = select_cams.reshape([-1,1])
        select_uvs = select_uvs.reshape([-1,2])
        pixel_keys = np.concatenate([select_cams,select_uvs],axis=1)

        uv_view_collector=[]
        for which_cam in regi_cam_id:
            tmp_idx = pixel_keys[:,0] == which_cam
            uv_view_collector.append(tuple(map(tuple,select_uvs[tmp_idx])))   

        pixel_keys = tuple(map(tuple, pixel_keys))
        texture_idxes = {pixel_keys[i]: [] for i in range(len(pixel_keys))} 
        for i in range(len(pixel_keys)):
            texture_idxes[pixel_keys[i]].append(i)

        collector = np.zeros([select_uvs.shape[0],args.measurement_len,3],np.float32) 
        idx_name = "cam00_index_nocc.bin"
        
        for which_view, which_cam in enumerate(regi_cam_id):
            # idx_name = "cam00_index_nocc_{}.bin".format(which_cam)

            # if which_cam not in regi_cam_id_goodboy:
            #     continue
            if which_cam % args.lighting_pattern_num == which_pattern:
                file_name = "cam00_data_{}_nocc_compacted.bin".format(which_cam)

                print("[{}] REGISTED CAM ID : ".format(len(regi_cam_id)), which_cam)

                tmp_measurements = np.fromfile(args.data_root+"0/{}".format(file_name),np.float32).reshape([-1,args.measurement_len,3])

                with open(args.data_root+"0/{}".format(idx_name),"rb") as pf:
                    _ = np.fromfile(pf,np.int32,1)
                    tmp_uvs = np.fromfile(pf,np.int32).reshape([-1,2])
                    tmp_uvs = tmp_uvs[:,[1,0]]
                assert tmp_uvs.shape[0] == len(tmp_measurements)

                tmp_map = {(tmp_uvs[i][0],tmp_uvs[i][1]) : i for i in range(tmp_uvs.shape[0])}
                
                for a_uv in uv_view_collector[which_view]:
                    collector[texture_idxes[(which_cam,a_uv[0],a_uv[1])]] = tmp_measurements[tmp_map[a_uv]]

        visible_view_num = np.reshape(visible_view_num,[-1,1])
        img = np.ones([512,512,3],np.uint8)*100
        img[tex_uvs[:,1],tex_uvs[:,0]] = np.repeat(visible_view_num,3,axis=1)
        cv2.imwrite(save_root+"visible_num_pattern_{}.png".format(which_pattern),img)

        # model_view_num = 64 // args.lighting_pattern_num

        selected_measurements = collector.reshape([-1,surplus_view_num,3])
        visible_measurements = np.max(selected_measurements,axis=-1).reshape(-1,surplus_view_num) 
        visible_measurements = np.where(visible_measurements/maxs[which_pattern]*255.0 > 220, np.zeros_like(visible_measurements), visible_measurements).reshape(-1,surplus_view_num) 
        
        for which_point in range(texel_num): 
            tmp_visible_view = np.argsort(visible_measurements[which_point])
            tmp_max_view = surplus_view_num
            tmp_min_view = (visible_measurements[which_point,tmp_visible_view]/maxs[which_pattern]*255.0 != 0.0).argmax()
            tmp_visible_view = tmp_visible_view[tmp_min_view:tmp_max_view]
            valid_view_num = len(tmp_visible_view)
            
            if valid_view_num == 0:
                tmp_visible_view = np.zeros(model_view_num,np.int32)
            elif valid_view_num < model_view_num:
                tmp_visible_view = np.concatenate([tmp_visible_view,np.ones(model_view_num-valid_view_num,np.int32)*tmp_visible_view[valid_view_num//2]],axis=-1)
            else:
                tmp_visible_view = tmp_visible_view[-model_view_num:]

            tmp_visible_view = np.sort(tmp_visible_view)
            
            reselected_measurements[which_point,which_pattern::args.lighting_pattern_num] = selected_measurements[which_point,tmp_visible_view] * args.scalar
            reselected_positions[which_point,which_pattern::args.lighting_pattern_num] = selected_positions[which_point,tmp_visible_view]
            reselected_normals[which_point,which_pattern::args.lighting_pattern_num] = selected_normals[which_point,tmp_visible_view]
            reselected_tangents[which_point,which_pattern::args.lighting_pattern_num] = selected_tangents[which_point,tmp_visible_view]
            reselect_views[which_point,which_pattern::args.lighting_pattern_num] = select_views[which_point,tmp_visible_view]

            
    reselected_positions = np.matmul(reselected_positions,cameraR) + cameraT
    reselected_normals = np.matmul(reselected_normals,cameraR)
    reselected_tangents = np.matmul(reselected_tangents,cameraR)
    
    reselected_measurements.astype(np.float32).tofile(save_root+"selected_measurements.bin")
    reselected_positions.astype(np.float32).tofile(save_root+"selected_positions.bin")
    reselected_normals.astype(np.float32).tofile(save_root+"selected_normals.bin")
    reselected_tangents.astype(np.float32).tofile(save_root+"selected_tangents.bin")
    reselect_views.astype(np.int32).tofile(save_root+"selected_views.bin")