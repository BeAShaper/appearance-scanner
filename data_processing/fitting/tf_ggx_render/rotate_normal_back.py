import numpy as np
import os
import shutil
import sys
import cv2
import math
import argparse
from pathlib import Path
import time
import matplotlib.pyplot as plt
TORCH_RENDER_PATH="../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
sys.path.append("../")
import torch_render
from torch_render import Setup_Config
import torch
import torchvision
sys.path.append("../val_files/inference/")

if __name__ == "__main__":
    batch_size = 1000
    parser = argparse.ArgumentParser()

    parser.add_argument("shot_root")
    parser.add_argument("feature_task")
    parser.add_argument("material_task")
    parser.add_argument("udt_folder_name")
    parser.add_argument("texture_folder_name")
    parser.add_argument("texture_resolution",type=int)
    parser.add_argument("sub_folder_name",type=str)
    args = parser.parse_args()

    tex_folder_root = args.shot_root+args.feature_task+"/"+args.udt_folder_name+"/"+args.texture_folder_name+"/"
    material_root = args.shot_root+args.material_task+"/images_{}/".format(args.sub_folder_name)
    sub_folder_name = args.sub_folder_name + "/"
    os.makedirs(tex_folder_root+sub_folder_name,exist_ok=True)

    device = "cuda:0"
    
    normals_geo_global = torch.from_numpy(np.fromfile(tex_folder_root+"normals_geo_global.bin",np.float32).reshape([-1,3])).to(device)
    tangents_geo_global = torch.from_numpy(np.fromfile(tex_folder_root+"tangents_geo_global.bin",np.float32).reshape([-1,3])).to(device)
    
    # geo = torch.from_numpy(np.fromfile(tex_folder_root+"/camera_sigmoid_geo/global_geo_frame.bin",np.float32).reshape([-1,9])).to(device)
    # normals_geo_global = geo[:,:3]
    # tangents_geo_global = geo[:,3:6]

    binormls_geo_global = torch.cross(normals_geo_global,tangents_geo_global)
    global_geo_frame = torch.stack([normals_geo_global,tangents_geo_global,binormls_geo_global],dim=1)

    # normals_geo_local = torch.from_numpy(np.fromfile(tex_folder_root+"normals_geo_local.bin",np.float32).reshape([-1,3])).to(device)
    # tangents_geo_local = torch.from_numpy(np.fromfile(tex_folder_root+"tangents_geo_local.bin",np.float32).reshape([-1,3])).to(device)
    # binormls_geo_local = torch.cross(normals_geo_local,tangents_geo_local)
    # local_geo_frame = torch.stack([normals_geo_local,tangents_geo_local,binormls_geo_local],dim=1)

    nn_normals = np.fromfile(material_root+"data_for_server/gathered_all/nn_normals.bin",np.float32).reshape([-1,3,3])
    nn_normals = np.mean(nn_normals,axis=1)
    nn_normals = torch.from_numpy(nn_normals).to(device)

    fitted_spec_params = np.fromfile(material_root+"data_for_server/gathered_all/params_gathered_gloabal_normal.bin",np.float32).reshape([-1,12])
    fitted_normals = torch.from_numpy(fitted_spec_params[:,:3]).to(device)
    fitted_tangents = np.fromfile(material_root+"data_for_server/gathered_all/gathered_global_tangent.bin",np.float32).reshape([-1,3])
    fitted_tangents = torch.from_numpy(fitted_tangents).to(device)
    
    print("fitted_normals ",fitted_normals.shape)
    print("fitted_tangents ",fitted_tangents.shape)
    print("global_geo_frame ",global_geo_frame.shape)
    print("nn_normals ",nn_normals.shape)

    pf_normal_global = open(tex_folder_root+sub_folder_name+"/normal_fitted_global.bin","wb")
    pf_tangent_global = open(tex_folder_root+sub_folder_name+"/tangent_fitted_global.bin","wb")
    pf_nn_normal_global = open(tex_folder_root+sub_folder_name+"/nn_normal_global.bin","wb")

    ptr = 0
    while True:
        
        tmp_normal = fitted_normals[ptr:ptr+batch_size]
        tmp_tangent = fitted_tangents[ptr:ptr+batch_size]
        tmp_global_geo_frame = global_geo_frame[ptr:ptr+batch_size]
        # tmp_local_geo_frame = local_geo_frame[ptr:ptr+batch_size]
        tmp_nn_normal = nn_normals[ptr:ptr+batch_size]

        valid_num = tmp_normal.shape[0]
        
        if valid_num == 0:
            break

        label_geo_frame = torch.from_numpy(np.array([[0,0,1],[1,0,0],[0,1,0]],np.float32)).unsqueeze(dim=0).repeat(valid_num,1,1).to(device)
        tmp_batch_size = tmp_normal.shape[0]
        tmp_binormal = torch.cross(tmp_normal,tmp_tangent)

        global_inv_R = torch.matmul(tmp_global_geo_frame.permute(0,2,1),torch.inverse(label_geo_frame.permute(0,2,1)))
        tmp_normal = torch.matmul(global_inv_R,tmp_normal.unsqueeze(dim=2)).squeeze(dim=2)
        tmp_tangent = torch.matmul(global_inv_R,tmp_tangent.unsqueeze(dim=2)).squeeze(dim=2)*-1
        tmp_nn_normal = torch.matmul(global_inv_R,tmp_nn_normal.unsqueeze(dim=2)).squeeze(dim=2)

        # local_inv_R = torch.matmul(tmp_local_geo_frame.permute(0,2,1),torch.inverse(label_geo_frame.permute(0,2,1)))
        # tmp_normal = torch.matmul(local_inv_R,tmp_normal.unsqueeze(dim=2)).squeeze(dim=2)
        # tmp_tangent = torch.matmul(local_inv_R,tmp_tangent.unsqueeze(dim=2)).squeeze(dim=2)*-1
        # tmp_nn_normal = torch.matmul(local_inv_R,tmp_nn_normal.unsqueeze(dim=2)).squeeze(dim=2)

        # global_inv_R = torch.matmul(tmp_global_geo_frame.permute(0,2,1),torch.inverse(tmp_local_geo_frame.permute(0,2,1)))
        # tmp_normal = torch.matmul(global_inv_R,tmp_normal.unsqueeze(dim=2)).squeeze(dim=2)
        # tmp_tangent = torch.matmul(global_inv_R,tmp_tangent.unsqueeze(dim=2)).squeeze(dim=2)*-1
        # tmp_nn_normal = torch.matmul(global_inv_R,tmp_nn_normal.unsqueeze(dim=2)).squeeze(dim=2)

        tmp_normal.cpu().numpy().astype(np.float32).tofile(pf_normal_global)
        tmp_tangent.cpu().numpy().astype(np.float32).tofile(pf_tangent_global)
        tmp_nn_normal.cpu().numpy().astype(np.float32).tofile(pf_nn_normal_global)
        
        ptr+=valid_num
        print(ptr)
    
    pf_normal_global.close()
    pf_tangent_global.close()
    pf_nn_normal_global.close()



# data_root = "F:/Turbot_freshmeat/7_11_16_8/blue_cloth/undistort_feature/texture_512/"
# sub_dir = "shot_GN_SD/"
# slice_len = 512
# standard_rendering_parameters = {
#     "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_plane_render_configs/",
#     "slice_height":16,
#     "slice_width":32
# }
# setup_input = Setup_Config(standard_rendering_parameters)

# param_bounds={}
# epsilon = 1e-3
# param_bounds["box"] = (-75.0,75.0)
# param_bounds["n"] = (epsilon,1.0-epsilon)
# param_bounds["theta"] = (0.0,math.pi)


# batch_size = 5000
# device = "cuda:0"
# # batch_indices = torch.arange(texel_num)[:,None] 
# random_rotate = np.fromfile(data_root+"random_rotate.bin",np.float32).reshape([-1,60,3])
# random_rotate = torch.from_numpy(random_rotate).to(device)
# best_cam_id = np.fromfile(data_root+sub_dir+"best_cam_id_ps.bin",np.int32).reshape([-1,1])
# best_view_id = np.fromfile(data_root+sub_dir+"selected_views.bin",np.int32).reshape([-1,16])
# best_rotate_angle = np.fromfile(data_root+"best_rotate_angle.bin",np.float32).reshape([-1,1])
# texel_num = best_cam_id.shape[0]
# texel_indices = np.arange(texel_num)[:,None]
# best_cam_id = best_view_id[texel_indices,best_cam_id]

# best_cam_id = np.fromfile(data_root+"best_cam_id.bin",np.int32).reshape([-1,1])

# def rotate_normal(data_root,input_normal,kind,idx=None,clock=True):
#     # random_rotate = np.fromfile(data_root+"random_rotate.bin",np.float32).reshape([-1,60,3])
#     # random_rotate = torch.from_numpy(random_rotate).to(device)
#     global random_rotate
#     if not clock:
#         random_rotate = random_rotate * -1

#     best_cam_id = np.fromfile(data_root+"best_cam_id_{}.bin".format(kind),np.int32).reshape([-1,16])
#     best_cam_id = np.expand_dims(best_cam_id[:,15-idx],axis=-1)

#     # best_cam_id = np.fromfile(data_root+"best_cam_id_{}.bin".format(kind),np.int32).reshape([-1,1])
#     # best_cam_id = np.expand_dims(best_cam_id[:,15-idx],axis=-1)

#     best_view_id = np.fromfile(data_root+"selected_views.bin",np.int32).reshape([-1,16])
    
#     texel_num = best_cam_id.shape[0]
#     texel_indices = np.arange(texel_num)[:,None]
#     best_cam_id = best_view_id[texel_indices,best_cam_id]

#     input_normal = torch.from_numpy(input_normal).to(device)
#     sequence = np.arange(input_normal.shape[0])
#     ptr = 0
#     while True:
#         tmp_sequence = sequence[ptr:ptr+batch_size]
        
#         if tmp_sequence.shape[0] == 0:
#             break
#         tmp_normal = input_normal[tmp_sequence]
#         tmp_best_cam_id = best_cam_id[tmp_sequence]
#         tmp_batch_indices = texel_indices[tmp_sequence]
        
#         tmp_rotate_angles = torch.squeeze(random_rotate[tmp_batch_indices,tmp_best_cam_id],dim=1)
        
#         AXIS = [2,1,0] 
#         for which_axis in AXIS:
#             setup_input.set_rot_axis_torch(which_axis)
#             tmp_normal = torch_render.rotate_vector_along_axis(setup_input,-tmp_rotate_angles[:,[which_axis]],tmp_normal,is_list_input=False)

#         input_normal[tmp_sequence] = tmp_normal

#         print("[NORMAL] {}/{}".format(ptr,texel_num))
#         ptr+=batch_size


#     return input_normal.cpu().numpy()


# def rotate_normal_global(data_root,input_normal,clock=True):
#     best_cam_id = np.fromfile(data_root+"best_cam_id.bin",np.int32).reshape([-1,1])
#     # best_view_id = np.fromfile(data_root+"selected_views.bin",np.int32).reshape([-1,16])
    
#     # texel_num = best_cam_id.shape[0]
#     # texel_indices = np.arange(texel_num)[:,None]
#     # best_cam_id = best_view_id[texel_indices,best_cam_id]

    

#     input_normal = torch.from_numpy(input_normal).to(device)
#     sequence = np.arange(input_normal.shape[0])
#     ptr = 0
#     while True:
#         tmp_sequence = sequence[ptr:ptr+batch_size]
        
#         if tmp_sequence.shape[0] == 0:
#             break
#         tmp_normal = input_normal[tmp_sequence]
#         tmp_best_cam_id = best_cam_id[tmp_sequence]
        
#         tmp_rotate_angles = torch.ones([tmp_sequence.shape[0],1])*math.pi*2/60
#         tmp_rotate_angles = tmp_rotate_angles.to(device) * torch.from_numpy(tmp_best_cam_id).float().to(device)
        
#         tmp_normal = torch_render.rotate_vector_along_axis(setup_input,-tmp_rotate_angles,tmp_normal,is_list_input=False)

#         input_normal[tmp_sequence] = tmp_normal

#         print("[NORMAL] {}/{}".format(ptr,texel_num))
#         ptr+=batch_size


#     return input_normal.cpu().numpy()

# def rotate_normal_global_siga19(data_root,input_normal,clock=True):
#     best_rotate_angle = np.fromfile(data_root+"best_rotate_angle.bin",np.float32).reshape([-1,1])
#     best_cam_id = np.fromfile(data_root+"best_cam_id.bin",np.int32).reshape([-1,1])
#     texel_num = best_cam_id.shape[0]
#     texel_indices = np.arange(texel_num)[:,None]
#     input_normal = torch.from_numpy(input_normal).to(device)
#     sequence = np.arange(input_normal.shape[0])
#     ptr = 0
#     while True:
#         tmp_sequence = sequence[ptr:ptr+batch_size]
        
#         if tmp_sequence.shape[0] == 0:
#             break
#         tmp_normal = input_normal[tmp_sequence]
#         tmp_best_cam_id = best_cam_id[tmp_sequence]
#         # print(tmp_best_cam_id)
#         # exit()
#         tmp_batch_indices = texel_indices[tmp_sequence]
#         # tmp_best_rotate_angle = best_rotate_angle[tmp_sequence]
#         # print(tmp_best_rotate_angle)
#         tmp_rotate_angles = torch.ones([tmp_sequence.shape[0],1])*math.pi*2/60
#         tmp_rotate_angles = tmp_rotate_angles.to(device) * torch.from_numpy(tmp_best_cam_id).float().to(device)

#         tmp_normal = torch_render.rotate_vector_along_axis(setup_input,-tmp_rotate_angles,tmp_normal,is_list_input=False)

#         input_normal[tmp_sequence] = tmp_normal

#         print("[NORMAL] {}/{}".format(ptr,texel_num))
#         ptr+=batch_size


#     return input_normal.cpu().numpy()

# def rotate_centre_normal_to_global(geo_normal,fitting_normal):
    
#     batch_size = geo_normal.shape[0]
#     geo_normal = torch.from_numpy(geo_normal)
#     fitting_normal = torch.from_numpy(fitting_normal)
#     device = geo_normal.device

#     cam_pos = setup_input.get_cam_pos_torch(device)
#     choose_light = (torch.ones(batch_size) * 240).long()
#     light_positions = setup_input.get_light_poses_torch(device)

#     wi = light_positions[choose_light] # - input_positions[0]
#     wo = cam_pos # - input_positions[0]
#     fix_normal = torch.nn.functional.normalize((wi+wo)/2, dim=1)
    
#     geo_normal = torch.nn.functional.normalize(geo_normal,dim=1)

#     u = torch.cross(fix_normal, geo_normal)
#     u = torch.nn.functional.normalize(u,dim=1)  # [batch,3]

#     tmp = fix_normal.mul(geo_normal)
#     tmp = torch.sum(tmp,dim=1,keepdim=True)
#     theta = torch.acos(tmp)
#     # theta = torch.acos(torch.einsum('ik, ik -> i', fix_normal, geo_normal)) / 2  # [batch,1]
    
#     rotate_matrix = torch_render.rotation_axis(theta,u)
#     view_mat_for_normal =torch.transpose(torch.inverse(rotate_matrix),1,2)
#     view_mat_for_normal_t = torch.transpose(view_mat_for_normal,1,2)
    
#     pn = torch.unsqueeze(torch.cat([fitting_normal,torch.ones(batch_size,1,dtype=fitting_normal.dtype,device=device)],dim=1),1)#[batch,1,4]
        
#     vector = torch.squeeze(torch.matmul(pn,view_mat_for_normal_t),1)[:,:3]
        
#     return vector.cpu().numpy()

