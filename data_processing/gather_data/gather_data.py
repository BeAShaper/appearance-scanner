import argparse
import numpy as np
import sys
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Select valid views.')
    parser.add_argument("data_root")
    parser.add_argument("sub_root")
    parser.add_argument("scalar",type=float)

    parser.add_argument("texture_map_size",type=int)
    parser.add_argument("config_file",type=str)
    parser.add_argument("model_path",type=str)
    parser.add_argument("--batch_size",type=int,default=200000)

    args = parser.parse_args()

    
    data_root = args.data_root+"undistort_feature/texture_{}/".format(args.texture_map_size)
    save_root = data_root + args.sub_root + "/"
    if(os.path.exists(save_root) == False):
        os.makedirs(save_root)

    
    tex_uvs = np.fromfile(data_root+"texturemap_uv.bin",np.int32).reshape([-1,2])
    regi_cam_id = np.fromfile(data_root+"regi_cam_id.bin",np.int32).astype(np.int32)
    sample_view_num = regi_cam_id.shape[0]
    print("{} images have been successfully registered.".format(sample_view_num))

    regi_cam_id = [regi_cam_id[i] for i in range(regi_cam_id.shape[0])]

    try:
        with open(args.model_path+"/0/maxs.bin","rb") as pf:
            maxs = np.fromfile(pf,np.float32)

    except FileNotFoundError as identifier:
        print("maxs.bin doesn't exist, run appearance_scanner/test_files/prepare_patterns.bat first.")
        

    camera_extrinsic = np.fromfile(args.config_file,np.float32)
    cameraR = camera_extrinsic[:9].reshape(3,3)
    cameraT = camera_extrinsic[9:].reshape(1,3)
    
    pf_all_positions = open(data_root+"positions_fulllocalview.bin","rb")
    pf_all_normals = open(data_root+"normals_geo_fulllocalview.bin","rb")
    pf_all_tangents = open(data_root+"tangents_geo_fulllocalview.bin","rb")
    pf_all_ndotv = open(data_root+"ndotv_fulllocalview.bin","rb")
    pf_cam_indexes = open(data_root+"cam_ids.bin","rb")
    pf_uvs = open(data_root+"uvs.bin","rb")


    data_size = {}
    data_size["positions"] = 3 * sample_view_num
    data_size["normals"] = 3 * sample_view_num
    data_size["tangents"] = 3 * sample_view_num
    data_size["ndotv"] = 1 * sample_view_num
    data_size["cam_indexs"] = 1 * sample_view_num
    data_size["uvs"] = 2 * sample_view_num

    pf_all_positions.seek(0,2)
    texel_num = pf_all_positions.tell()//4//data_size["positions"]
    pf_all_positions.seek(0,0)
    
    pf_map = {}
    pf_map["views"] = open(save_root+"selected_views.bin","wb")
    pf_map["measurements"] = open(save_root+"selected_measurements.bin","wb")
    pf_map["positions"] = open(save_root+"selected_positions.bin","wb")
    pf_map["normals"] = open(save_root+"selected_normals.bin","wb")
    pf_map["tangents"] = open(save_root+"selected_tangents.bin","wb")
    pf_map["visible_views_num"] = open(save_root+"visible_views_num.bin","wb")

    texel_sequence = np.arange(texel_num)
    start_ptr = 0
    ptr = start_ptr
    batch_size = args.batch_size
    
    while True:
        tmp_sequence = texel_sequence[ptr:ptr+batch_size]
        if tmp_sequence.shape[0] == 0:
            break
        tmp_seq_size = tmp_sequence.shape[0]

        tmp_cam_indexes = np.fromfile(pf_cam_indexes,np.int32,count=data_size["cam_indexs"]*tmp_seq_size).reshape([-1,sample_view_num,1]).astype(np.int32)
        
        tmp_ndotv = np.fromfile(pf_all_ndotv,np.float32,count=data_size["ndotv"]*tmp_seq_size).reshape([-1,sample_view_num])
        tmp_positions = np.fromfile(pf_all_positions,np.float32,count=data_size["positions"]*tmp_seq_size).reshape([-1,sample_view_num,3])
        tmp_normals = np.fromfile(pf_all_normals,np.float32,count=data_size["normals"]*tmp_seq_size).reshape([-1,sample_view_num,3])
        tmp_tangents = np.fromfile(pf_all_tangents,np.float32,count=data_size["tangents"]*tmp_seq_size).reshape([-1,sample_view_num,3])
        tmp_uvs = np.fromfile(pf_uvs,np.int32,count=data_size["uvs"]*tmp_seq_size).reshape([-1,sample_view_num,2]).astype(np.int32)
        

        select_views = [] 
        selected_positions = [] 
        selected_normals = [] 
        selected_tangents = [] 
        select_views = []
        select_cams = [] 
        select_uvs = [] 
        visible_view_num = np.zeros([tmp_seq_size],np.int32)

        valid_cam_indexes = tmp_cam_indexes.copy()


        for which_point in range(tmp_seq_size): 
        
            tmp_visible_cam = np.where(valid_cam_indexes[which_point] != -1)[0]  # all visible views 
            tmp_visible_cam_ndotv = np.where(tmp_ndotv[which_point] < 0.3)[0]
            tmp_visible_cam = np.array(list(set(tmp_visible_cam)-set(tmp_visible_cam_ndotv)))
            
            visible_view_num[which_point] = tmp_visible_cam.shape[0]
            

            if tmp_visible_cam.shape[0] == 0:
                select_views.append([])
                selected_positions.append([])
                selected_normals.append([])
                selected_tangents.append([])
                continue

            tmp_visible_cam = np.sort(tmp_visible_cam)
            select_views.append(tmp_visible_cam)
            selected_positions.append(np.matmul(tmp_positions[which_point,tmp_visible_cam],cameraR)+cameraT)
            selected_normals.append(np.matmul(tmp_normals[which_point,tmp_visible_cam],cameraR))
            selected_tangents.append(np.matmul(tmp_tangents[which_point,tmp_visible_cam],cameraR))
            select_cams.append(np.array(regi_cam_id)[tmp_visible_cam])
            select_uvs.append(tmp_uvs[which_point,tmp_visible_cam])

        select_cams = np.concatenate(select_cams,axis=0).reshape([-1,1])
        select_uvs = np.concatenate(select_uvs,axis=0).reshape([-1,2])
        pixel_keys = np.concatenate([select_cams,select_uvs],axis=1)

        uv_view_collector=[]
        for which_cam in regi_cam_id:
            tmp_idx = pixel_keys[:,0] == which_cam
            uv_view_collector.append(tuple(map(tuple,select_uvs[tmp_idx])))   

        pixel_keys = tuple(map(tuple, pixel_keys))
        texture_idxes = {pixel_keys[i]: [] for i in range(len(pixel_keys))} 
        for i in range(len(pixel_keys)):
            texture_idxes[pixel_keys[i]].append(i)

        collector = np.zeros([select_uvs.shape[0],1,3],np.float32) 
        idx_name = "cam00_index_nocc.bin"


        for which_view, which_cam in enumerate(regi_cam_id):
            file_name = "cam00_data_{}_nocc_compacted.bin".format(which_cam)

            if not os.path.isfile(args.data_root+"raw_images/{}".format(file_name)):
                file_name = "cam00_data_0_nocc_compacted.bin"

            tmp_measurements = np.fromfile(args.data_root+"raw_images/{}".format(file_name),np.float32).reshape([-1,1,3])

            pf = open(args.data_root+"raw_images/{}".format(idx_name),"rb")
            
            _ = np.fromfile(pf,np.int32,1)
            tmp_uvs = np.fromfile(pf,np.int32).reshape([-1,2])
            tmp_uvs = tmp_uvs[:,[1,0]]

            assert tmp_uvs.shape[0] == len(tmp_measurements)

            tmp_map = {(tmp_uvs[i][0],tmp_uvs[i][1]) : i for i in range(tmp_uvs.shape[0])}
            
            for a_uv in uv_view_collector[which_view]:
                collector[texture_idxes[(which_cam,a_uv[0],a_uv[1])]] = tmp_measurements[tmp_map[a_uv]] 
            
            print("\rProcess : ", "%.2f" % (min((ptr+batch_size)/texel_num*100, 100)), "%  - ", "%.2f" % (which_view/sample_view_num*100), "%",end="")


        point_ptr = 0
        for which_point in range(tmp_seq_size): 
            tmp_visible_view_num = visible_view_num[which_point]

            tmp_selected_measurements = collector[point_ptr:point_ptr+tmp_visible_view_num].reshape([tmp_visible_view_num,3])
            
            point_ptr += tmp_visible_view_num

            if tmp_visible_view_num == 0:
                np.array([0]).astype(np.int32).tofile(pf_map["visible_views_num"])
                continue
        
            visible_measurements = np.max(tmp_selected_measurements,axis=-1)

            visible_measurements = np.where(visible_measurements/maxs[0]*255.0 > 220, np.zeros_like(visible_measurements), visible_measurements)


            tmp_visible_view = np.argsort(visible_measurements)

            tmp_min_view = (visible_measurements[tmp_visible_view]/maxs[0]*255.0 != 0.0).argmax()
            tmp_visible_view = tmp_visible_view[tmp_min_view:] 

            valid_view_num = len(tmp_visible_view)
            
            np.array([valid_view_num]).astype(np.int32).tofile(pf_map["visible_views_num"])
            
            if valid_view_num == 0:
                continue
            
            tmp_visible_view = np.sort(tmp_visible_view)

            (tmp_selected_measurements[tmp_visible_view]* args.scalar).astype(np.float32).tofile(pf_map["measurements"])
            selected_positions[which_point][tmp_visible_view].astype(np.float32).tofile(pf_map["positions"])
            selected_normals[which_point][tmp_visible_view].astype(np.float32).tofile(pf_map["normals"])
            selected_tangents[which_point][tmp_visible_view].astype(np.float32).tofile(pf_map["tangents"])
            select_views[which_point][tmp_visible_view].astype(np.int32).tofile(pf_map["views"])


        ptr+=batch_size

    
    for a_key in pf_map:
        pf_map[a_key].close()

    print("\n[SELECT VALID VIEWS] done.")



        
    