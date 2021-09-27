import numpy as np
import os
import shutil
import sys
import cv2
import argparse
import torch
import math
TORCH_RENDER_PATH="../../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render
from torch_render import Setup_Config
import torchvision

from scanner_net_inuse import ScannerNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("model_root")
    parser.add_argument("model_file_name")
    parser.add_argument("diff_sample_num",type=int)
    parser.add_argument("spec_sample_num",type=int)

    parser.add_argument("m_len",type=int)
    parser.add_argument("lighting_pattern_num",type=int)

    parser.add_argument("--m_scalar",type=float,default=1.0)
    parser.add_argument("--batch_size",type=int,default=1000)
    parser.add_argument("--log_lumi",action="store_true",default=False)

    args = parser.parse_args()

    spec_slice_len = args.spec_sample_num * args.spec_sample_num * 6
    diff_slice_len = args.diff_sample_num * args.diff_sample_num * 6
    
    rendering_parameters = {
        "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_cube_slice_{}x{}/".format(args.spec_sample_num,args.spec_sample_num),
        "slice_width":256,
        "slice_height":192
    }
    spec_fitting_setup = Setup_Config(rendering_parameters)

    rendering_parameters["config_dir"]=TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_cube_slice_{}x{}/".format(args.diff_sample_num,args.diff_sample_num)
    diff_fitting_setup = Setup_Config(rendering_parameters)

    rendering_parameters["config_dir"]=TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_cube_slice_64x64/"
    setup_output = Setup_Config(rendering_parameters)

    #####################################
    ###load model
    #####################################
    RENDER_SCALAR = 5*1e3/math.pi
    inference_device = torch.device("cuda:0")
    
    # np.set_printoptions(threshold=np.inf)
    nn_model = ScannerNet(args)
    for param in nn_model.parameters():
        param.requires_grad = False
    pretrained_dict = torch.load(args.model_root + args.model_file_name, map_location='cuda:0')
    print("loading trained model...")
    something_not_found = False
    model_dict = nn_model.state_dict()
    for k,_ in model_dict.items():
        if k not in pretrained_dict and "linear_projection" not in k and "weight_layers" not in k:
            print("not found:",k)
            something_not_found = True
    if something_not_found:
        exit()
    model_dict = nn_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    nn_model.load_state_dict(model_dict)
    nn_model.to(inference_device)
    nn_model.eval()

    #####################################
    ###about data
    #####################################
    byte_num_per_record = args.m_len*4
    batch_size = args.batch_size
    
    #####################################
    ###process here
    #####################################
    
    ##about input files
    
    selected_measurements = open(args.data_root+"selected_measurements.bin","rb")
    selected_positions = open(args.data_root+"selected_positions.bin","rb")
    selected_normals = open(args.data_root+"selected_normals.bin","rb")
    selected_tangents = open(args.data_root+"selected_tangents.bin","rb")
    visible_views_num = np.fromfile(args.data_root+"visible_views_num.bin",np.int32)
    
    texel_num = visible_views_num.shape[0]
    data_record_size = 3 
    ##about output files
    pf_map = {}
    pf_map["spec_slice"] = open(args.data_root+"spec_slice.bin","wb")
    pf_map["diff_slice"] = open(args.data_root+"diff_slice.bin","wb")
    pf_map["nn_normals"] = open(args.data_root+"nn_normals.bin","wb")

    ##infer here
    fitting_sequence = np.arange(texel_num)
    start_ptr = 0
    ptr = start_ptr

    if args.log_lumi:
        log_path = args.data_root+"lumi_img/"
        os.makedirs(log_path,exist_ok=True)

    while True:
        tmp_sequence = fitting_sequence[ptr:ptr+batch_size]
        tmp_sequence_size = tmp_sequence.shape[0]
        if tmp_sequence_size == 0:
            break
        tmp_batch_max_visible_num = np.max(visible_views_num[ptr:ptr+batch_size])
        tmp_batch_max_visible_num = tmp_batch_max_visible_num if tmp_batch_max_visible_num > 0 else 1
        tmp_measurements = np.zeros([tmp_sequence_size,tmp_batch_max_visible_num,3],np.float32)
        tmp_positions = np.zeros([tmp_sequence_size,tmp_batch_max_visible_num,3],np.float32)
        tmp_normals = np.zeros([tmp_sequence_size,tmp_batch_max_visible_num,3],np.float32)
        tmp_tangents = np.zeros([tmp_sequence_size,tmp_batch_max_visible_num,3],np.float32)

        for which_point in range(tmp_sequence_size):
            tmp_visible_view_num = visible_views_num[ptr+which_point]
            if tmp_visible_view_num == 0:
                tmp_point_measurements = np.zeros([1,1,3])
                tmp_point_positions = np.zeros([1,1,3])
                tmp_point_normals = np.zeros([1,1,3])
                tmp_point_tangents = np.zeros([1,1,3])
            else:
                tmp_point_measurements = np.fromfile(selected_measurements,np.float32,count=data_record_size*tmp_visible_view_num).reshape([1,tmp_visible_view_num,3])
                tmp_point_positions = np.fromfile(selected_positions,np.float32,count=data_record_size*tmp_visible_view_num).reshape([1,tmp_visible_view_num,3])
                tmp_point_normals = np.fromfile(selected_normals,np.float32,count=data_record_size*tmp_visible_view_num).reshape([1,tmp_visible_view_num,3])
                tmp_point_tangents = np.fromfile(selected_tangents,np.float32,count=data_record_size*tmp_visible_view_num).reshape([1,tmp_visible_view_num,3])
            
            this_batch_sample_size = tmp_measurements.shape[0]
            tmp_measurements[which_point,:tmp_visible_view_num,:] = tmp_point_measurements
            tmp_positions[which_point,:tmp_visible_view_num,:] = tmp_point_positions
            tmp_normals[which_point,:tmp_visible_view_num,:] = tmp_point_normals
            tmp_tangents[which_point,:tmp_visible_view_num,:] = tmp_point_tangents

            tmp_measurements[which_point,tmp_visible_view_num:,:] = tmp_point_measurements[0,[0]]
            tmp_positions[which_point,tmp_visible_view_num:,:] = tmp_point_positions[0,[0]]
            tmp_normals[which_point,tmp_visible_view_num:,:] = tmp_point_normals[0,[0]]
            tmp_tangents[which_point,tmp_visible_view_num:,:] = tmp_point_tangents[0,[0]]


        tmp_measurements = torch.from_numpy(tmp_measurements).type(torch.FloatTensor).to(inference_device)
        tmp_positions = torch.from_numpy(tmp_positions).type(torch.FloatTensor).to(inference_device)
        tmp_normals = torch.from_numpy(tmp_normals).type(torch.FloatTensor).to(inference_device)
        tmp_tangents = torch.from_numpy(tmp_tangents).type(torch.FloatTensor).to(inference_device)
        n_t_xyz = torch.cat([tmp_normals,tmp_tangents],dim=-1)

        batch_data = {
            "input_positions" : tmp_positions,
            "n_t_xyz" : n_t_xyz,
            "measurements" : tmp_measurements
        }
        
        tmp_result_map = nn_model(batch_data)
    
        spec_tensor = tmp_result_map["spec_slice"]#(batch,sample_view_num,spec_slice_len,1)
        diff_tensor = tmp_result_map["diff_slice"]#(batch,sample_view_num,diff_slice_len,1)
        
        spec_tensor = spec_tensor.reshape([this_batch_sample_size,1,spec_slice_len])
        diff_tensor = diff_tensor.reshape([this_batch_sample_size,1,diff_slice_len])
        nn_normal = tmp_result_map["nn_normals"]
        
        if args.log_lumi:
            spec_tensor_tmp = spec_tensor.clone().permute(0,2,1).reshape(this_batch_sample_size,-1,1)
            diff_tensor_tmp = diff_tensor.clone().permute(0,2,1).reshape(this_batch_sample_size,-1,1)
            
            spec_lumi_imgs = torch_render.visualize_lumi(spec_tensor_tmp.cpu().numpy(),spec_fitting_setup,resize=True)
            diff_lumi_imgs = torch_render.visualize_lumi(diff_tensor_tmp.cpu().numpy(),diff_fitting_setup,resize=True)
            lumi_imgs = spec_lumi_imgs+diff_lumi_imgs

            img_stack = np.stack([lumi_imgs,diff_lumi_imgs,spec_lumi_imgs],axis=1) #(batchsize,3,imgheight,imgwidth,channel)
            img_stack_list = torch.from_numpy(img_stack)

            for which_sample in range(this_batch_sample_size):
                tmp_lumi_img = torchvision.utils.make_grid(img_stack_list[which_sample].permute(0,3,1,2),nrow=3, pad_value=0.5) 
                tmp_lumi_img = tmp_lumi_img.permute(1,2,0).numpy()
                cv2.imwrite(log_path+"{}_lumi.png".format(ptr+which_sample),tmp_lumi_img[:,:,::-1]*255.0)

        for a_key in pf_map:
            tmp_data = tmp_result_map[a_key].cpu().detach().numpy()
            tmp_data.astype(np.float32).tofile(pf_map[a_key])

        if ptr % 10000 == 0:
            print("[MAIN] {}/{}".format(ptr,texel_num))
        
        ptr += batch_size
        

    for a_key in pf_map:
        pf_map[a_key].close()
    
    

    