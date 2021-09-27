import numpy as np
import argparse
import torch
import sys
import math
import os
import cv2
from scipy.optimize import least_squares
from scipy.optimize import nnls
from scipy.optimize import lsq_linear
TORCH_RENDER_PATH = "../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render
from torch_render import Setup_Config

RENDER_SCALAR = 2*1e3/math.pi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_for_server_root",default="F:/Turbot_freshmeat/12_28/egypt2/data/images/data_for_server/")
    parser.add_argument("--sample_view_num",type=int,default="128")
    parser.add_argument("--m_len_perview",type=int,default=3)
    parser.add_argument("--batchsize",type=int,default=100)
    parser.add_argument('--thread_ids', nargs='+', type=int,default=[5])
    parser.add_argument('--gpu_id', type=int,default=0)
    parser.add_argument("--need_dump",action="store_true")
    parser.add_argument("--total_thread_num",type=int,default=24)
    parser.add_argument("--tex_resolution",type=int,default=512)


    args = parser.parse_args()

    for which_test_thread in args.thread_ids:
        data_root = args.data_for_server_root+"{}/".format(which_test_thread)

        ###############
        ###init
        ###############
        compute_device = torch.device("cuda:{}".format(args.gpu_id))

        ###############
        ### read in fitting device data  ###############
        standard_rendering_parameters = {
            "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/handheld_device_render_config_16x32/"
        }
        setup_input = Setup_Config(standard_rendering_parameters)
        color_tensor = setup_input.get_color_tensor(compute_device)#(3,3,3)#(light,obj,cam)
        color_tensor = color_tensor.permute(2,0,1)#(cam,light,obj)
        # print(color_tensor)

        cam_pos_centre = torch.from_numpy(np.array([0,0,356.20557],np.float32)).to(compute_device)
        print("cam_pos_centre:",cam_pos_centre)
        ###############
        ###read in data
        ###############
        measurements = torch.from_numpy(np.fromfile(data_root+"measurements.bin",np.float32)).to(compute_device).reshape((-1,args.m_len_perview))
        positions = torch.from_numpy(np.fromfile(data_root+"position.bin",np.float32)).to(compute_device).reshape((-1,3))
        normals_geo = np.fromfile(data_root+"normal_geo.bin",np.float32).reshape((-1,3))
        tangents_geo = np.fromfile(data_root+"tangent_geo.bin",np.float32).reshape((-1,3))
        binormals_geo = np.cross(normals_geo.reshape((-1,3)),tangents_geo.reshape((-1,3)))
        fitted_params_grey = torch.from_numpy(np.fromfile(data_root+"fitted.bin",np.float32).reshape((-1,7))).to(compute_device)
        visible_view_num = np.fromfile(data_root+"visible_view_num.bin",np.int32)

        sample_num = fitted_params_grey.shape[0]

        shape_list = [measurements.shape[0],positions.shape[0],normals_geo.shape[0],tangents_geo.shape[0],binormals_geo.shape[0]]
        assert shape_list.count(binormals_geo.shape[0]) == len(shape_list),"some data are corrupted,shape list:{}".format(shape_list)

        print("sample num:",sample_num)


        lighting_patterns_np = np.fromfile(args.data_for_server_root+"W.bin",np.float32).reshape((-1,setup_input.get_light_num(),3))
        assert lighting_patterns_np.shape[0] == 1,"I cannot handle multipatterns now!"
        lighting_patterns = torch.from_numpy(lighting_patterns_np[0]).to(compute_device)#(lightnum,channel_num)
        lighting_pattern_num = 1#lighting_patterns.shape[0]
        print("lighting pattern num:",lighting_pattern_num)

        ###############=
        ### calculate R back 
        ### R_inv * 001_frame = geo_frame 
        ### 001_frame = R * geo_frame 
        ### 001_fram,geo_frame = (3,1)
        ###############
        print("Building R_inv matrix....")
        frame_geo = np.stack([normals_geo.reshape((-1,3)),tangents_geo.reshape((-1,3)),binormals_geo.reshape((-1,3))],axis=2)#(framenum,3,3 axis)
        
        frame_geo_001 = np.zeros_like(frame_geo)#(framenum,3,3 axis)
        frame_geo_001[:,2,0] = 1.0#n,z
        frame_geo_001[:,0,1] = 1.0#t,x
        frame_geo_001[:,1,2] = 1.0#b,y
        frame_geo = torch.from_numpy(frame_geo).to(compute_device)
        frame_geo_001 = torch.from_numpy(frame_geo_001).to(compute_device)
        R_inv = torch.matmul(frame_geo,torch.inverse(frame_geo_001))
        # R_inv = R_inv.reshape(sample_num,args.sample_view_num,3,3)#(samplenum,sample_view_num,3,3)
        print("Done.")
        
        ##########################################
        ### fitting rhod rhos with measurements
        ##########################################
        uvs = np.fromfile(args.data_for_server_root+"texturemap_uv.bin",np.int32).reshape((-1,2))
        sample_num_normal = uvs.shape[0]//12
        if which_test_thread == 11:
            uvs = uvs[which_test_thread*sample_num_normal:]
        else:
            uvs = uvs[which_test_thread*sample_num_normal:(which_test_thread+1)*sample_num_normal]
        # ids = np.where((uvs==(406,460)).all(axis=1))[0][0]
        ids = 0
        print(ids)
        # exit()
        log_path = data_root+"check/"
        os.makedirs(log_path,exist_ok=True)



        pf_result = open(data_root+"fitted_pd_ps.bin","wb")
        ptr_start = ids
        ptr = ptr_start
        ptr_multiview = 0
        print("fitting pd ps...")
        while True:
            if ptr % 1000 == 0:
                print("{}/{}".format(ptr,sample_num))

            tmp_visible_view_num = visible_view_num[ptr:ptr+args.batchsize]
            tmp_fitted_params_grey = fitted_params_grey[ptr:ptr+args.batchsize]#(batchsize,7)
            

            cur_batchsize = tmp_visible_view_num.shape[0]
            if cur_batchsize == 0:
                print("break because all done.")
                break
            ptr = ptr+cur_batchsize

            tmp_visible_view_num_all = np.sum(tmp_visible_view_num)
            if tmp_visible_view_num_all == 0:
                R = np.zeros((cur_batchsize,6),np.float32)#(batchsize,6)
                R.tofile(pf_result)
                continue

            tmp_positions = positions[ptr_multiview:ptr_multiview+tmp_visible_view_num_all]#(batchsize*sampleviewnum,3)
            tmp_measurements = measurements[ptr_multiview:ptr_multiview+tmp_visible_view_num_all]#(batchsize*sampleviewnum,m_len_perview)
            tmp_R_inv = R_inv[ptr_multiview:ptr_multiview+tmp_visible_view_num_all]#(batchsize*sampleviewnum,3,3)
            tmp_frame_geo = frame_geo[ptr_multiview:ptr_multiview+tmp_visible_view_num_all]#(batchsize*sampleviewnum,3,3)
            # tmp_fitted_params_grey_multiview = torch.unsqueeze(tmp_fitted_params_grey,dim=1).repeat(1,args.sample_view_num,1)#(batchsize,,sampleviewnum,7)

            ptr_multiview = ptr_multiview+tmp_visible_view_num_all

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

            tmp_fitted_params_grey_multiview = torch.zeros(0,7,dtype = torch.float32,device=tmp_fitted_params_grey.device)
            frame_shading_centre_multiview = torch.zeros(0,3,3,dtype = torch.float32,device=frame_shading_centre.device)
            for which_sample,this_visible_num in enumerate(tmp_visible_view_num):
                tmp_fitted_params_grey_multiview = torch.cat(
                    [
                        tmp_fitted_params_grey_multiview,
                        tmp_fitted_params_grey[[which_sample]].repeat(this_visible_num,1)
                    ],dim=0
                )
                frame_shading_centre_multiview = torch.cat(
                    [
                        frame_shading_centre_multiview,
                        frame_shading_centre[[which_sample]].repeat(this_visible_num,1,1)
                    ]
                )
            frame_shading_localview = torch.matmul(tmp_R_inv,frame_shading_centre_multiview)#(batchsize*sampleviewnum,3 axis,3 vector)
            frame_shading_localview = [
                frame_shading_localview[:,:,0],
                frame_shading_localview[:,:,1],
                frame_shading_localview[:,:,2],
            ]
            # frame_shading_localview = [
            #     tmp_frame_geo[:,:,0],
            #     tmp_frame_geo[:,:,1],
            #     tmp_frame_geo[:,:,2],
            # ]

            #rendering lumi and get measurement
            tmp_fitted_params_grey_pd = tmp_fitted_params_grey_multiview.clone()#(batchsize*sampleviewnum,7)
            tmp_fitted_params_grey_pd[:,5] = 1.0
            tmp_fitted_params_grey_pd[:,6] = 0.0
            
            tmp_fitted_params_grey_ps = tmp_fitted_params_grey_multiview.clone()#(batchsize*sampleviewnum,7)
            tmp_fitted_params_grey_ps[:,5] = 0.0
            tmp_fitted_params_grey_ps[:,6] = 1.0

            tmp_rotate_theta = torch.zeros(tmp_visible_view_num_all,1,device=compute_device,dtype=torch.float32)

            rendered_pd_slice_1,_ = torch_render.draw_rendering_net(
                setup_input,
                tmp_fitted_params_grey_pd,
                tmp_positions,
                tmp_rotate_theta,
                "diff_1",
                global_custom_frame=frame_shading_localview,
                use_custom_frame="ntb",
                pd_ps_wanted="pd_only"
            )
            rendered_pd_slice_1 = rendered_pd_slice_1.reshape(tmp_visible_view_num_all,setup_input.get_light_num(),1)*RENDER_SCALAR#(batchsize*sampleviewnum,lightnum,1)
            
            rendered_ps_slice_1,end_points = torch_render.draw_rendering_net(
                setup_input,
                tmp_fitted_params_grey_ps,
                tmp_positions,
                tmp_rotate_theta,
                "diff_1",
                global_custom_frame=frame_shading_localview,
                use_custom_frame="ntb",
                pd_ps_wanted="ps_only"
            )
            rendered_ps_slice_1 = rendered_ps_slice_1.reshape(tmp_visible_view_num_all,setup_input.get_light_num(),1)*RENDER_SCALAR#(batchsize,sampleviewnum,lightnum,1)

            # visibility = end_points["n_dot_view_dir"].reshape(cur_batchsize,args.sample_view_num)#(batchsize,sampleviewnum)
            # valid_flag = torch.where(visibility > 0.0,torch.ones_like(visibility,dtype=torch.int32),torch.zeros_like(visibility,dtype=torch.int32))#(batchsize,1)
            # visible_view_num = torch.sum(valid_flag,dim=1)#(batchsize,)
            # invalid_view_idxes = torch.where(visible_view_num==0)[0]#(invalid view num,)

            tmp_lighting_patterns = torch.unsqueeze(lighting_patterns.clone(),dim=0)#(1,lgihtnum,3)
            P = tmp_lighting_patterns.permute(0,2,1)#(1,3,lightnum)

            #TODO light transport 3x3x3 tensor should be considered here!!!

            T_d = torch.matmul(P,rendered_pd_slice_1)#(batchsize*sampleviewnum,3,1)
            T_s = torch.matmul(P,rendered_ps_slice_1)#(batchsize*sampleviewnum,3,1)
            T = torch.cat((T_d,T_s),dim=2)#(batchsize*sampleviewnum,3,2)
            T = torch.unsqueeze(T,dim=1).repeat(1,3,1,1)#(bathsize*sampleviewnum,3 cam ,3 light,2 diff spec)
            C = color_tensor.clone().unsqueeze(dim=0)#(1,3 cam ,3 light,3 obj)
            
            A = torch.matmul(T.permute(0,1,3,2),C.permute(0,1,3,2)).reshape(tmp_visible_view_num_all,3,6)#reshape(cur_batchsize*args.sample_view_num,3,6)

            M = tmp_measurements.reshape((tmp_visible_view_num_all,args.m_len_perview))
            
            if True:
                def shoting_sim(x,T,C,M,this_visible_num):

                    x_reshaped = np.repeat(np.repeat(np.expand_dims(np.expand_dims(x.reshape((2,3)),axis=0),axis=0),this_visible_num,axis=0),3,axis=1)#(sampleviewnum,3 cam ,2 diff spec,3 rgb)
                    M_rendered = np.matmul(T,x_reshaped)#(sampleviewnum,3cam,3 light, 3 obj)
                    M_rendered = np.sum((M_rendered*C).reshape(this_visible_num*3,9),axis=1)

                    residual = M_rendered - M#np.linalg.norm()
                    residual = residual**2
                    residual = np.concatenate((residual,0.0*(x**2)),axis=0)

                    return residual


                batch_collector = []
                minibatch_ptr = 0
                for i,this_visible_num in enumerate(tmp_visible_view_num):
                    tmpT = T[minibatch_ptr:minibatch_ptr+this_visible_num].cpu().numpy().astype(np.float64)#(sampleviewnum,3 cam, 3 light, 2 diff spec)
                    tmpC = C.cpu().numpy().astype(np.float64)#(1,3 cam, 3 light, 3 obj)
                    tmpM = M[minibatch_ptr:minibatch_ptr+this_visible_num].cpu().numpy().astype(np.float64).reshape((-1))#(sample_view_num*lightingpatternnum,3)
                    try:
                        x0_rosenbrock = np.array((0.5,0.5,0.5,1.0,1.0,1.0),np.float32)
                        tmpX = least_squares(shoting_sim, x0_rosenbrock,args=(tmpT, tmpC,tmpM,this_visible_num),bounds=(np.array((0.0,0.0,0.0,0.0,0.0,0.0)),np.array((np.inf,np.inf,np.inf,np.inf,np.inf,np.inf)))).x
                        # tmpX = lsq_linear(tmpA, tmpM,bounds=(np.array((0.0,0.0,0.0,0.0,0.0,0.0)),np.array((1.0,1.0,1.0,20.0,20.0,20.0)))).x
                        # tmpX,_ = nnls(tmpA, tmpM)
                    except Exception as identifier:
                        print(identifier)
                        tmpX = np.zeros(6,)
                    
                    batch_collector.append(tmpX)
                    minibatch_ptr = minibatch_ptr+this_visible_num
            else:
                batch_collector = []
                minibatch_ptr = 0
                for i,this_visible_num in enumerate(tmp_visible_view_num):
                    if this_visible_num == 0:
                        tmpX = np.zeros(6,)
                    else:
                        tmpA = A[minibatch_ptr:minibatch_ptr+this_visible_num].cpu().numpy().astype(np.float64)#(sampleviewnum,3,6)
                        tmpM = M[minibatch_ptr:minibatch_ptr+this_visible_num].cpu().numpy().astype(np.float64)#(sampleviewnum,3)
                        try:
                            tmpX = lsq_linear(tmpA.reshape((this_visible_num*3,6)), tmpM.reshape((this_visible_num*3)),bounds=(np.array((0.0,0.0,0.0,0.0,0.0,0.0)),np.array((np.inf,np.inf,np.inf,np.inf,np.inf,np.inf)))).x
                            # tmpX,_ = nnls(tmpA, tmpM)
                        except Exception as identifier:
                            tmpX = np.zeros(6,)
                    
                    batch_collector.append(tmpX)
                    minibatch_ptr = minibatch_ptr+this_visible_num
            R = np.stack(batch_collector,axis=0)#(batchsize,6)
            R.astype(np.float32).tofile(pf_result)
            
            
            

            # MTM = torch.matmul(M.permute(0,2,1),M)#(batchsize,2,2)
            # MTM[invalid_view_idxes] = torch.eye(2,dtype=torch.float32,device=MTM.device)
            
            # MTM_minus1 = torch.inverse(MTM+1000.0*torch.eye(2).to(compute_device))
            
            # MTMp = torch.matmul(M.permute(0,2,1),Mp)#(batchsize,2,3)

            # A = torch.matmul(MTM_minus1,MTMp)#(batchssize,2,3)
            # A[invalid_view_idxes] = torch.zeros_like(A[invalid_view_idxes])

            # A.cpu().numpy().tofile(pf_result)

            # log here
            if args.need_dump:
                minibatch_ptr = 0
                col_num = 30
                for which_sample,this_visible_num in enumerate(tmp_visible_view_num):
                    tmp_R = np.repeat(np.repeat(R[which_sample].reshape(1,1,2,3),this_visible_num,axis=0),3,axis=1)#(sampleviewnum,3 cam,2,3)
                    tmp_T = T[minibatch_ptr:minibatch_ptr+this_visible_num].cpu().numpy()#pattern*lumi (sampleviewnum,3 cam, 3 light, 2 diff spec)
                    tmp_M = np.matmul(tmp_T,tmp_R)#pattern*lumi*albedo (sampleviewnum,3 cam, 3 light, 3 obj)
                    tmp_C = color_tensor.clone().unsqueeze(dim=0).cpu().numpy()#(1,3 cam ,3 light,3 obj)

                    tmpM_render = np.reshape(tmp_M*tmp_C,(this_visible_num,3,9))#(sampleviewnum,3 cam,9)
                    tmpM_render = np.sum(tmpM_render,axis=2)#(sampleviewnum,3)

                    tmpM_photo =M[minibatch_ptr:minibatch_ptr+this_visible_num].cpu().numpy()#(sampleviewnum,3) 

                    remain_num = col_num - tmpM_photo.shape[0] % col_num
                    tmpM_photo_cat = np.concatenate([tmpM_photo,np.zeros((remain_num,3),np.float32)],axis=0)
                    tmpM_photo_cat = np.reshape(tmpM_photo_cat,(-1,col_num,3))
                    tmpM_photo_cat = cv2.resize(tmpM_photo_cat,(tmpM_photo_cat.shape[1]*10,tmpM_photo_cat.shape[0]*10),interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(log_path+"{}_Mp.png".format(ptr-cur_batchsize+which_sample),tmpM_photo_cat[:,:,::-1]*10.0)


                    tmpM_all = np.concatenate((tmpM_render,tmpM_photo),axis=1)
                    np.savetxt(log_path+"{}_M.csv".format(ptr-cur_batchsize+which_sample),tmpM_all,delimiter=',')

                    minibatch_ptr = minibatch_ptr+this_visible_num

            # break
            
        pf_result.close()
        print("done.")

        ######################
        ###
        ######################
        fitted_params = np.fromfile(data_root+"fitted_pd_ps.bin",np.float32).reshape((-1,2,3))
        fitted_params_pd = fitted_params[:,0]
        print(fitted_params_pd.shape)
        fitted_params_ps = fitted_params[:,1]
        
        uvs = np.fromfile(args.data_for_server_root+"texturemap_uv.bin",np.int32).reshape((-1,2))
        sample_num = uvs.shape[0]//args.total_thread_num
        print("ptr:{} sample_num:{}".format(ptr,sample_num))
        if which_test_thread == (args.total_thread_num-1) and ptr == sample_num:
            uvs = uvs[which_test_thread*sample_num+ptr_start:]
        else:
            uvs = uvs[which_test_thread*sample_num+ptr_start:which_test_thread*sample_num+ptr]
        print(uvs.shape)
        # uvs = np.fromfile(data_root+"texturemap_uv.bin",np.int32).reshape((-1,2))[2*sample_num+ptr_start:2*sample_num+ptr]

        img = np.zeros((args.tex_resolution,args.tex_resolution,3),np.float32)
        img[uvs[:,1],uvs[:,0]] = fitted_params_pd
        cv2.imwrite(data_root+"pd.exr",img[:,:,::-1])
        # tex_root = "F:/Turbot_freshmeat/12_28/egypt2/undistort_feature/texture_512/"
        # cv2.imwrite(tex_root+"pd.exr",img[:,:,::-1])
        img[uvs[:,1],uvs[:,0]] = fitted_params_ps
        cv2.imwrite(data_root+"ps.exr",img[:,:,::-1])
        # cv2.imwrite(tex_root+"ps.exr",img[:,:,::-1])
        img[uvs[:,1],uvs[:,0]] = np.ones_like(fitted_params_ps)
        cv2.imwrite(data_root+"where.exr",img[:,:,::-1])

