import numpy as np
import cv2
import argparse
import sys
import torch
import os
TORCH_RENDER_PATH = "../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render
from torch_render import Setup_Config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",default="F:/Turbot_freshmeat/12_28/rabbit/data/images/data_for_server/")
    parser.add_argument("--thread_num",type=int,default=12)
    parser.add_argument("--batchsize",type=int,default=10000)
    parser.add_argument("--tex_resolution",type=int,default=512)
    
    args = parser.parse_args()

    compute_device = torch.device("cuda:0")
    data_for_server_root = args.data_root
    cam_pos_centre = torch.from_numpy(np.array([0,0,356.20557],np.float32)).to(compute_device)#(3,)
    print("cam_pos_centre:",cam_pos_centre)

    ##read in fitted params
    fitted_collector = []
    for which_thread in range(args.thread_num):
        cur_root = data_for_server_root+"{}/".format(which_thread)
        tmp_fitted = np.fromfile(cur_root+"fitted.bin",np.float32).reshape((-1,7))
        fitted_collector.append(tmp_fitted)

    fitted_params_grey = np.concatenate(fitted_collector,axis=0)#(samplenum,)
    fitted_params_grey = torch.from_numpy(fitted_params_grey).to(compute_device)

    ##read in positions geo frames
    normals_geo = np.fromfile(data_for_server_root+"normals_geo_global.bin",np.float32).reshape((-1,3))
    tangents_geo = np.fromfile(data_for_server_root+"tangents_geo_global.bin",np.float32).reshape((-1,3))
    binormals_geo = np.cross(normals_geo.reshape((-1,3)),tangents_geo.reshape((-1,3)))

    num_list = [normals_geo.shape[0],tangents_geo.shape[0],binormals_geo.shape[0]]
    assert num_list.count(fitted_params_grey.shape[0]) == len(num_list),"some bins are corrupted,num list:{} wanted:{}".format(num_list,fitted_params_grey.shape[0])
    sample_num = fitted_params_grey.shape[0]
    
    frame_geo = np.stack([normals_geo,tangents_geo,binormals_geo],axis=2)#(framenum,3,3 axis)
    frame_geo_001 = np.zeros_like(frame_geo)#(framenum,3,3 axis)
    frame_geo_001[:,2,0] = 1.0#n,z
    frame_geo_001[:,0,1] = 1.0#t,x
    frame_geo_001[:,1,2] = 1.0#b,y
        
    frame_geo = torch.from_numpy(frame_geo).to(compute_device)
    frame_geo_001 = torch.from_numpy(frame_geo_001).to(compute_device)
    R_inv = torch.matmul(frame_geo,torch.inverse(frame_geo_001))
    
    normal_collector = np.zeros((0,3),np.float32)
    tangent_collector = np.zeros((0,3),np.float32)
    binormal_collector = np.zeros((0,3),np.float32)
    axay_collector = np.zeros((0,2),np.float32)
    pd_grey_collector = np.zeros((0,1),np.float32)
    ps_grey_collector = np.zeros((0,1),np.float32)

    ptr = 0
    while True:
        if ptr % 1000 == 0:
            print("{}/{}".format(ptr,sample_num))

        tmp_R_inv = R_inv[ptr:ptr+args.batchsize]#(batchsize,3,3)
        tmp_fitted_params_grey = fitted_params_grey[ptr:ptr+args.batchsize]#(batchsize,7)
        
        cur_batchsize = tmp_fitted_params_grey.shape[0]
        if cur_batchsize == 0:
            print("break because all done.")
            break
        ptr = ptr+cur_batchsize

        #reconstruct fitted normal,tangent
        positions_centre = torch.zeros((cur_batchsize,3),dtype=torch.float32,device=compute_device)
        view_dir = cam_pos_centre -  positions_centre#shape=[batch,3]
        view_dir = torch.nn.functional.normalize(view_dir,dim=1)#shape=[batch,3]
        
        #build local frame
        frame_t,frame_b = torch_render.build_frame_f_z(view_dir,None,with_theta=False)#[batch,3]
        frame_n = view_dir

        n_2d = tmp_fitted_params_grey[:,:2]#(batchsize,2)
        n_local = torch_render.back_hemi_octa_map(n_2d)#(batch,3)
        theta = tmp_fitted_params_grey[:,[2]]
        t_local,_ = torch_render.build_frame_f_z(n_local,theta,with_theta=True)
        n_local_x,n_local_y,n_local_z = torch.split(n_local,[1,1,1],dim=1)#[batch,1],[batch,1],[batch,1]
        n = n_local_x*frame_t+n_local_y*frame_b+n_local_z*frame_n#[batch,3]
        t_local_x,t_local_y,t_local_z = torch.split(t_local,[1,1,1],dim=1)#[batch,1],[batch,1],[batch,1]
        t = t_local_x*frame_t+t_local_y*frame_b+t_local_z*frame_n#[batch,3]
        b = torch.cross(n,t)#[batch,3]

        frame_shading_centre = torch.stack([n,t,b],dim=2)#(batchsize,3 axis,3vector)

        frame_shading_globalview = torch.matmul(tmp_R_inv,frame_shading_centre)#(batchsize,3 axis,3 vector)
        frame_shading_globalview = [
            frame_shading_globalview[:,:,0],
            frame_shading_globalview[:,:,1],
            frame_shading_globalview[:,:,2],
        ]

        normal_collector = np.concatenate(
            [
                normal_collector,
                frame_shading_globalview[0].cpu().numpy().astype(np.float32)
            ],axis=0
        )
        tangent_collector = np.concatenate(
            [
                tangent_collector,
                frame_shading_globalview[1].cpu().numpy().astype(np.float32)
            ],axis=0
        )
        binormal_collector = np.concatenate(
            [
                binormal_collector,
                frame_shading_globalview[2].cpu().numpy().astype(np.float32)
            ],axis=0
        )

        axay_collector = np.concatenate(
            [
                axay_collector,
                tmp_fitted_params_grey[:,3:5].cpu().numpy().astype(np.float32)
            ],
            axis=0
        )

    #######
    # tangent ambiguity removing
    tangent_collector = np.where(axay_collector[:,[0]] > axay_collector[:,[1]],tangent_collector,binormal_collector)
    dot_result = np.sum(tangent_collector*np.array([0.0,0.0,1.0]),axis=1,keepdims=True)
    tangent_collector = np.where(dot_result>0.0,tangent_collector,-tangent_collector)
    axay_collector = np.where(axay_collector[:,[0]]>axay_collector[:,[1]],axay_collector,axay_collector[:,::-1])


    ##########################
    #output fitted texture maps
    tex_uv = np.fromfile(data_for_server_root+"texturemap_uv.bin",np.int32).reshape((-1,2))
    save_root = data_for_server_root+"fitted_grey/"
    os.makedirs(save_root,exist_ok=True)

    image = np.zeros((args.tex_resolution,args.tex_resolution,3),np.float32)
    image[tex_uv[:,1],tex_uv[:,0]] = normal_collector*0.5+0.5
    cv2.imwrite(save_root+"normal_fitted_global.exr",image[:,:,::-1])

    image = np.zeros((args.tex_resolution,args.tex_resolution,3),np.float32)
    image[tex_uv[:,1],tex_uv[:,0]] = tangent_collector*0.5+0.5
    cv2.imwrite(save_root+"tangent_fitted_global.exr",image[:,:,::-1])

    image = np.zeros((args.tex_resolution,args.tex_resolution,3),np.float32)
    image[tex_uv[:,1],tex_uv[:,0],:2] = axay_collector
    cv2.imwrite(save_root+"axay_fitted.exr",image[:,:,::-1])

    pd_collector = np.zeros((0,3),np.float32)
    ps_collector = np.zeros((0,3),np.float32)
    for which_thread in range(args.thread_num):
        fitted_params = np.fromfile(data_for_server_root+"{}/fitted_pd_ps.bin".format(which_thread),np.float32).reshape((-1,2,3))
        fitted_params_pd = fitted_params[:,0]
        fitted_params_ps = fitted_params[:,1]
        pd_collector = np.concatenate([pd_collector,fitted_params_pd],axis=0)
        ps_collector = np.concatenate([ps_collector,fitted_params_ps],axis=0)
    
    img = np.zeros((args.tex_resolution,args.tex_resolution,3),np.float32)
    img[tex_uv[:,1],tex_uv[:,0]] = pd_collector
    cv2.imwrite(save_root+"pd_fitted.exr",img[:,:,::-1])
    img[tex_uv[:,1],tex_uv[:,0]] = ps_collector
    cv2.imwrite(save_root+"ps_fitted.exr",img[:,:,::-1])