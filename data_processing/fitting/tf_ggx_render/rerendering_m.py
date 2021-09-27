import numpy as np
import sys
import torch
import math
TORCH_RENDER_PATH = "../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render
import matplotlib.pyplot as plt
import shutil
from torch_render import Setup_Config
import cv2
import os
import argparse

RENDER_SCALAR = 2*1e3/math.pi
RENDER_SCALAR_OUTPUT = 5*1e3/math.pi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",default="F:/Turbot_freshmeat/1_11/egypt/data/")

    args = parser.parse_args()

    #################
    ##
    #################
    standard_rendering_parameters = {
            "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/handheld_device_render_config_16x32/"
        }
    setup = Setup_Config(standard_rendering_parameters)

    standard_rendering_parameters["config_dir"] = TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_cube_slice_8x8/"
    setup_output_diff = Setup_Config(standard_rendering_parameters)
    setup_output_diff.set_cam_pos(np.array([0,0,356.20557],np.float32))

    standard_rendering_parameters["config_dir"] = TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_cube_slice_64x64/"
    setup_output_spec = Setup_Config(standard_rendering_parameters)
    setup_output_spec.set_cam_pos(np.array([0,0,356.20557],np.float32))


    src_root = args.data_root+"images_check/"
    save_root = args.data_root+"images_check_rerendered/"
    os.makedirs(save_root,exist_ok=True)

    ################
    ##
    ################
    rendering_device = torch.device("cuda:0")

    local_normals = torch.from_numpy(np.fromfile(src_root+"selected_normals.bin",np.float32).reshape((-1,3))).to(rendering_device)#(pointnum,3)
    local_tangents = torch.from_numpy(np.fromfile(src_root+"selected_tangents.bin",np.float32).reshape((-1,3))).to(rendering_device)#(pointnum,3)
    local_positions = torch.from_numpy(np.fromfile(src_root+"selected_positions.bin",np.float32).reshape((-1,3))).to(rendering_device)#(pointnum,3)
    local_binormals = torch.cross(local_normals,local_tangents)

    pointnum = local_normals.shape[0]

    shape_list = [local_positions.shape[0],local_tangents.shape[0]]
    assert shape_list.count(pointnum) == len(shape_list)

    input_params = np.zeros((pointnum,11),np.float32)
    input_params[:,3:5] = 0.3
    input_params[:,5:8] = 1e-5
    input_params[:,8:11] = (19.36,9.758,3.484)

    disturb_stddev = {
        "rhod":0.22,
        "rhos":0.22
    }

    # input_params[:,3:5] = 0.05
    # input_params[:,5:8] = 0.4
    # input_params[:,8:11] = (7.36,4.758,3.484)

    output_params = np.concatenate(
        [
            input_params[:,:5],
            np.mean(input_params[:,5:8],axis=1,keepdims=True),
            np.mean(input_params[:,8:11],axis=1,keepdims=True)
        ],axis=1
    )
    output_params = torch.from_numpy(output_params).to(rendering_device)


    lighting_pattern = np.fromfile(src_root+"W.bin",np.float32).reshape((-1,setup.get_light_num(),3))#(1,lightnum,3)
    lighting_pattern = torch.from_numpy(lighting_pattern).to(rendering_device)
    lighting_pattern = torch.unsqueeze(lighting_pattern,dim=3).repeat(pointnum,1,1,1)#(pointnum,lightnum,3,1)

    ################
    ##
    ################
    input_params_tc = torch.from_numpy(input_params).to(rendering_device)
    input_params_tc_noised = input_params_tc.clone()
    input_params_tc_noised[:,5:8] = torch.clamp(input_params_tc_noised[:,5:8]*torch.from_numpy(np.random.normal(1.0,disturb_stddev["rhod"],[pointnum,3]).astype(np.float32)).to(rendering_device),0.0,1.0)
    input_params_tc_noised[:,8:11] = torch.clamp(input_params_tc_noised[:,8:11]*torch.from_numpy(np.random.normal(1.0,disturb_stddev["rhos"],[pointnum,3]).astype(np.float32)).to(rendering_device),0.0,30.0)


    rotate_zeros = torch.zeros(pointnum,1,dtype=torch.float32,device=rendering_device)

    rendered_lumi,_ = torch_render.draw_rendering_net(setup,input_params_tc,local_positions,rotate_zeros,"lala",global_custom_frame=[local_normals,local_tangents,local_binormals],use_custom_frame="ntb")#(pointnum,lightnum,3)
    rendered_lumi = rendered_lumi*RENDER_SCALAR
    rendered_lumi = torch.unsqueeze(rendered_lumi,dim=2)#(pointnum,lightnum,1,3)

    rendered_lumi_noised,_ = torch_render.draw_rendering_net(setup,input_params_tc_noised,local_positions,rotate_zeros,"lala",global_custom_frame=[local_normals,local_tangents,local_binormals],use_custom_frame="ntb")#(pointnum,lightnum,3)
    rendered_lumi_noised = rendered_lumi_noised*RENDER_SCALAR
    rendered_lumi_noised = torch.unsqueeze(rendered_lumi_noised,dim=2)#(pointnum,lightnum,1,3)

    color_tensor = setup.get_color_tensor(rendering_device)#(3,3,3)
    color_tensor =color_tensor.reshape(1,9,3)/color_tensor.max()
    
    light_obj = lighting_pattern*rendered_lumi#(pointnum,lightnum,3,3)
    light_obj = torch.sum(light_obj,dim=1)#(pointnum,3,3)
    light_obj = torch.reshape(light_obj,(pointnum,1,9))

    light_obj_noised = lighting_pattern*rendered_lumi_noised#(pointnum,lightnum,3,3)
    light_obj_noised = torch.sum(light_obj_noised,dim=1)#(pointnum,3,3)
    light_obj_noised = torch.reshape(light_obj_noised,(pointnum,1,9))

    measurement = torch.matmul(light_obj,color_tensor).reshape(pointnum,3)#(pointnum,3)
    measurement_noised = torch.matmul(light_obj_noised,color_tensor).reshape(pointnum,3)#(pointnum,3)

    tmp_noise = torch.randn_like(measurement)*0.15+1.
    measurement_noised = measurement_noised*tmp_noise
    measurement_noised = measurement_noised.cpu().numpy()
    measurement_noised.tofile(save_root+"selected_measurements.bin")
    measurement = measurement.cpu().numpy()
    # measurement.tofile(save_root+"selected_measurements.bin")

    visible_num = np.fromfile(src_root+"visible_views_num.bin",np.int32)

    m_log_root = save_root+"rerendered_m/"
    os.makedirs(m_log_root,exist_ok=True)
    ptr = 0
    fig = plt.figure()
    for which_point in range(visible_num.shape[0]):
        tmp_measurement = measurement_noised[ptr:ptr+visible_num[which_point]]
        np.savetxt(m_log_root+"{}.csv".format(which_point),tmp_measurement,delimiter=',')
        
        xs = np.arange(visible_num[which_point])
        for which_channel in range(3):
            m_nonoise = measurement[ptr:ptr+visible_num[which_point],which_channel]
            m_noised = measurement_noised[ptr:ptr+visible_num[which_point],which_channel]
            plt.ylim(0,120)
            plt.plot(xs,m_nonoise,color="r")
            plt.plot(xs,m_noised,color="b")
            plt.savefig(m_log_root+"{}_{}.png".format(which_point,["r","g","b"][which_channel]))
            plt.clf()

        ptr+=visible_num[which_point]

    
    for a_file in ["normals_geo_global.bin","selected_normals.bin","selected_positions.bin","selected_tangents.bin","tangents_geo_global.bin","texturemap_uv.bin","visible_views_num.bin","W.bin"]:
        shutil.copyfile(src_root+a_file,save_root+a_file)

    
    ##################
    ##
    ##################
    pointnum = 1
    output_params = output_params[[0]]
    print(output_params)

    output_pos = torch.zeros(pointnum,3,dtype=torch.float32,device=rendering_device)
    output_normal = torch.zeros(pointnum,3,dtype=torch.float32,device=rendering_device)
    output_normal[:,2]=1.0
    output_tangent = torch.zeros(pointnum,3,dtype=torch.float32,device=rendering_device)
    output_tangent[:,0]=1.0
    output_binormal = torch.zeros(pointnum,3,dtype=torch.float32,device=rendering_device)
    output_binormal[:,1]=1.0

    print(output_params.size())

    rendered_diff_gt,_ = torch_render.draw_rendering_net(setup_output_diff,output_params,output_pos,rotate_zeros[[0]],"sdkfl",global_custom_frame=[output_normal,output_tangent,output_binormal],use_custom_frame="ntb",pd_ps_wanted="pd_only")
    rendered_spec_gt,_ = torch_render.draw_rendering_net(setup_output_spec,output_params,output_pos,rotate_zeros[[0]],"sdkfl",global_custom_frame=[output_normal,output_tangent,output_binormal],use_custom_frame="ntb",pd_ps_wanted="ps_only")
    rendered_diff_gt = rendered_diff_gt*RENDER_SCALAR_OUTPUT
    rendered_spec_gt = rendered_spec_gt*RENDER_SCALAR_OUTPUT

    gt_lumi_root = save_root+"gt_lumi/"
    os.makedirs(gt_lumi_root,exist_ok=True)
    lumi_diff_gt = torch_render.visualize_lumi(rendered_diff_gt.cpu().numpy(),setup_output_diff,resize=True)
    lumi_spec_gt = torch_render.visualize_lumi(rendered_spec_gt.cpu().numpy(),setup_output_spec,resize=True)

    for which_point in range(pointnum):
        cv2.imwrite(gt_lumi_root+"{}_diff.png".format(which_point),lumi_diff_gt[which_point]*255.0)
        cv2.imwrite(gt_lumi_root+"{}_spec.png".format(which_point),lumi_spec_gt[which_point]*255.0)
