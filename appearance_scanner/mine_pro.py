import torch
import numpy as np
import math
import torch
import threading
from mine import Mine,param_bounds,rejection_sampling_axay
import time
import sys
import random
TORCH_RENDER_PATH="../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render
from multiview_renderer_mt_pro import Multiview_Renderer
from torch_render import Setup_Config

def disturb_frame(normal,tangent,binormal,disturb_stddev):
    batch_size = normal.shape[0]
    device = normal.device
    tmp_x2 = torch.from_numpy(np.random.rand(batch_size,2).astype(np.float32)).to(device)
    disturb_dir = tangent*tmp_x2[:,[0]]+binormal*tmp_x2[:,[1]]
    disturb_len = torch.from_numpy(np.random.normal(0.0,disturb_stddev["normal"],[batch_size,1]).astype(np.float32)).to(device)
    disturbed_normal = normal+disturb_dir*disturb_len
    disturbed_normal = torch.nn.functional.normalize(disturbed_normal,dim=1)
    theta = torch.rand(batch_size,1)*(param_bounds["theta"][1]-param_bounds["theta"][0])+param_bounds["theta"][0]
    theta = theta.to(device)
    disturbed_tangent,disturbed_binormal = torch_render.build_frame_f_z(disturbed_normal,theta,with_theta=True)

    disturbed_tangent_neg = disturbed_tangent * -1
    
    pos_dot = torch.sum(tangent*disturbed_tangent,dim=-1,keepdim=True)
    neg_dot = torch.sum(tangent*disturbed_tangent_neg,dim=-1,keepdim=True)   
    
    disturbed_tangent = torch.where(pos_dot>neg_dot, disturbed_tangent, disturbed_tangent_neg)
    disturbed_binormal = torch.cross(disturbed_normal, disturbed_tangent)

    return disturbed_normal,disturbed_tangent,disturbed_binormal

def gen_geometry_parameters(setup_input,setup_output,disturb_stddev,batch_data,batch_size,device,sample_view_num):

    global_input_positions = np.random.uniform(param_bounds["box"][0],param_bounds["box"][1],[batch_size,3]).astype(np.float32)
    
    local_input_positions = np.random.uniform(param_bounds["box"][0],param_bounds["box"][1],[sample_view_num,batch_size,3]).astype(np.float32)
    local_input_positions = np.matmul(local_input_positions.reshape(-1,3),setup_input.cameraR) + setup_input.cameraT

    global_input_positions = torch.from_numpy(global_input_positions).reshape([-1,3]).to(device)
    local_input_positions = torch.from_numpy(local_input_positions).reshape([-1,3]).to(device)
    
    cam_pos = setup_input.get_cam_pos_torch(device)

    ######################################
    ##### gt shading normal
    ######################################
    ns = torch.rand(batch_size,2).to(device)
    theta = torch.rand(batch_size,1,device=device)*(param_bounds["theta"][1]-param_bounds["theta"][0])+param_bounds["theta"][0]
    global_shading_normal = torch_render.back_full_octa_map(ns)       
    global_shading_tangent,global_shading_binormal = torch_render.build_frame_f_z(global_shading_normal,theta,with_theta=True)
    global_shading_frame = [global_shading_normal,global_shading_tangent,global_shading_binormal]

    global_geo_normal,global_geo_tangent,global_geo_binormal = disturb_frame(global_shading_normal,global_shading_tangent,global_shading_binormal,disturb_stddev)
    global_geo_frame = torch.stack([global_geo_normal,global_geo_tangent,global_geo_binormal],dim=1) #[batch_size,3,3]  
    
    ##### global shading frame ==> label shading frame
    label_geo_frame = torch.from_numpy(np.array([[0,0,1],[1,0,0],[0,1,0]],np.float32)).unsqueeze(dim=0).repeat(batch_size,1,1).to(device) #[batch_size,3,3]  
    global_R = torch.matmul(label_geo_frame.permute(0,2,1),torch.inverse(global_geo_frame.permute(0,2,1)))   # rotation matrix from global to local
    label_shading_normal = torch.matmul(global_R,global_shading_normal.unsqueeze(dim=2)).squeeze(dim=2)
    label_shading_tangent = torch.matmul(global_R,global_shading_tangent.unsqueeze(dim=2)).squeeze(dim=2)
    label_shading_binormal = torch.matmul(global_R,global_shading_binormal.unsqueeze(dim=2)).squeeze(dim=2)
    label_shading_frame = [label_shading_normal,label_shading_tangent,label_shading_binormal]

    ######################################
    ##### sample local shading frame 
    ######################################
    theta = torch.rand(sample_view_num*batch_size,1,device=device)*(param_bounds["theta"][1]-param_bounds["theta"][0])+param_bounds["theta"][0]
    wo = torch.nn.functional.normalize(cam_pos - local_input_positions, dim=1) #[batch*sample_view_num,3]
    frame_t,frame_b = torch_render.build_frame_f_z(wo,None,with_theta=False) 
    frame_n = wo
    ns_random = torch.rand(sample_view_num*batch_size,2).to(device)
    n_local = torch_render.back_hemi_octa_map(ns_random)
    t_local,b_local = torch_render.build_frame_f_z(n_local,theta,with_theta=True)
    n_local_x,n_local_y,n_local_z = torch.split(n_local,[1,1,1],dim=1)
    shading_normal_random = n_local_x*frame_t+n_local_y*frame_b+n_local_z*frame_n
    t_local_x,t_local_y,t_local_z = torch.split(t_local,[1,1,1],dim=1)
    shading_tangent_random = t_local_x*frame_t+t_local_y*frame_b+t_local_z*frame_n
    shading_binormal_random = torch.cross(shading_normal_random,shading_tangent_random)

    while True:
        ndotv = torch.sum(shading_normal_random*wo,dim=-1)
        idx = torch.where(ndotv < 0.3)[0]  # delete grazing angle
        if idx.shape[0] == 0:
            break
        ns_random_new = torch.rand(idx.shape[0],2,device=device)
        ns_random[idx] = ns_random_new

        n_local = torch_render.back_hemi_octa_map(ns_random)
        n_local_x,n_local_y,n_local_z = torch.split(n_local,[1,1,1],dim=1)
        shading_normal_random = n_local_x*frame_t+n_local_y*frame_b+n_local_z*frame_n
        t_local,b_local = torch_render.build_frame_f_z(n_local,theta,with_theta=True)
        t_local_x,t_local_y,t_local_z = torch.split(t_local,[1,1,1],dim=1)
        shading_tangent_random = t_local_x*frame_t+t_local_y*frame_b+t_local_z*frame_n
        shading_binormal_random = torch.cross(shading_normal_random,shading_tangent_random)

    local_shading_normal = shading_normal_random.reshape([sample_view_num,batch_size,3])
    local_shading_tangent = shading_tangent_random.reshape([sample_view_num,batch_size,3])
    local_shading_binormal = shading_binormal_random.reshape([sample_view_num,batch_size,3])
    local_input_positions = local_input_positions.reshape([sample_view_num,batch_size,3])

    ################ confidence part ####################
    tmp_local_input_positions = local_input_positions.permute(1,0,2).reshape([-1,3])
    tmp_local_shading_normal = local_shading_normal.permute(1,0,2).reshape([-1,3])
    tmp_local_shading_tangent = local_shading_tangent.permute(1,0,2).reshape([-1,3])
    tmp_local_shading_binormal = local_shading_binormal.permute(1,0,2).reshape([-1,3])

    tmp_wo = torch.nn.functional.normalize(cam_pos - tmp_local_input_positions, dim=1)
    ndotv = torch.sum(tmp_local_shading_normal*tmp_wo,dim=-1,keepdim=True)
    
    tmp_wi = 2 * ndotv * tmp_local_shading_normal - tmp_wo
    tmp_wi = torch.nn.functional.normalize(tmp_wi,dim=-1)
    
    #########################################


    local_shading_normal = [torch.squeeze(n,dim=0) for n in torch.chunk(local_shading_normal,sample_view_num,dim=0)]
    local_shading_tangent = [torch.squeeze(t,dim=0) for t in torch.chunk(local_shading_tangent,sample_view_num,dim=0)]
    local_shading_binormal = [torch.squeeze(b,dim=0) for b in torch.chunk(local_shading_binormal,sample_view_num,dim=0)]
    local_input_positions = [torch.squeeze(pos,dim=0) for pos in torch.chunk(local_input_positions,sample_view_num,dim=0)]    

    global_points = torch.stack([global_shading_normal,global_shading_tangent,global_shading_binormal],dim=1) # [batch,3,3]

    input_params = []

    disturb_net = True
    if disturb_net:
        hogwild_view_num = 1
        disturbed_view_num = sample_view_num - hogwild_view_num 

        axay_hogwild = torch.from_numpy(rejection_sampling_axay(batch_size*hogwild_view_num).astype(np.float32)).to(device)
        pd_hogwild = torch.from_numpy(np.random.uniform(param_bounds["pd"][0],param_bounds["pd"][1],[batch_size*hogwild_view_num,3]).astype(np.float32)).to(device)
        ps_hogwild = torch.from_numpy(np.random.uniform(param_bounds["ps"][0],param_bounds["ps"][1],[batch_size*hogwild_view_num,3]).astype(np.float32)).to(device)

        axay_disturbed = torch.clamp(batch_data[:,3:5].repeat(disturbed_view_num,1)*torch.from_numpy(np.random.normal(1.0,disturb_stddev["axay_disturb"],[batch_size*disturbed_view_num,2]).astype(np.float32)).to(device),param_bounds["a"][0],param_bounds["a"][1])
        pd_disturbed = torch.clamp(batch_data[:,5:8].repeat(disturbed_view_num,1)*torch.from_numpy(np.random.normal(1.0,disturb_stddev["rhod"],[batch_size*disturbed_view_num,3]).astype(np.float32)).to(device),param_bounds["pd"][0],param_bounds["pd"][1])
        ps_disturbed = torch.clamp(batch_data[:,8:11].repeat(disturbed_view_num,1)*torch.from_numpy(np.random.normal(1.0,disturb_stddev["rhos"],[batch_size*disturbed_view_num,3]).astype(np.float32)).to(device),param_bounds["ps"][0],param_bounds["ps"][1])

        input_params_hogwild = torch.cat([batch_data[:,:3].repeat(hogwild_view_num,1),axay_hogwild,pd_hogwild,ps_hogwild],dim=1).unsqueeze(dim=1).reshape([hogwild_view_num,batch_size,11]).permute(1,0,2)
        input_params_disturbed = torch.cat([batch_data[:,:3].repeat(disturbed_view_num,1),axay_disturbed,pd_disturbed,ps_disturbed],dim=1).reshape([disturbed_view_num,batch_size,11]).permute(1,0,2)
        
        if sample_view_num-disturbed_view_num-hogwild_view_num > 0:
            input_params_gt = batch_data.unsqueeze(dim=1).repeat(1,sample_view_num-disturbed_view_num-hogwild_view_num,1)
        else:
            input_params_gt = torch.zeros([batch_size,0,11],device=device)

        input_params = torch.cat([input_params_gt,input_params_disturbed,input_params_hogwild],dim=1)
    else:
        input_params = batch_data.unsqueeze(dim=1).repeat(1,sample_view_num,1)

    batch_indices = torch.arange(batch_size)[:,None]
    view_indices = []
    for i in range(batch_size):
        indice = [i for i in range(sample_view_num)]
        random.shuffle(indice)
        view_indices.append(indice)
    view_indices = np.stack(view_indices,axis=0)
    view_indices = torch.from_numpy(view_indices).long().to(device)
    input_params = input_params[batch_indices,view_indices].permute(1,0,2) 

    ################ confidence part #######################
    tmp_input_params = input_params.permute(1,0,2).reshape([-1,11])
    tmp_normal,tmp_theta,tmp_ax,tmp_ay,tmp_pd3,tmp_ps3 = torch.split(tmp_input_params,[2,1,1,1,3,3],dim=1)

    wi_local = torch.cat([  torch.sum(tmp_wi*tmp_local_shading_tangent,dim=1,keepdim=True),
                            torch.sum(tmp_wi*tmp_local_shading_binormal,dim=1,keepdim=True),
                            torch.sum(tmp_wi*tmp_local_shading_normal,dim=1,keepdim=True)],dim=1)#shape is [batch_size*sample_view_num,3]
    
    wo_local = torch.cat([  torch.sum(tmp_wo*tmp_local_shading_tangent,dim=1,keepdim=True),
                            torch.sum(tmp_wo*tmp_local_shading_binormal,dim=1,keepdim=True),
                            torch.sum(tmp_wo*tmp_local_shading_normal,dim=1,keepdim=True)],dim=1)
    wi_local = wi_local.unsqueeze(dim=1)
    
    

    tmp_BRDF = torch_render.calc_light_brdf(wi_local,wo_local,tmp_ax,tmp_ay,tmp_pd3,tmp_ps3,pd_ps_wanted="both",specular_component="D_F_G_B")
    
    tmp_BRDF = tmp_BRDF.squeeze(dim=1).reshape([-1,sample_view_num,3])  #[batch_size,sample_view_num,3]
    tmp_BRDF = torch.mean(tmp_BRDF,dim=-1) #[batch_size,sample_view_num]

    ########################################################
    input_params = [torch.squeeze(p,dim=0) for p in torch.chunk(input_params,sample_view_num,dim=0)]

    local_shading_frames = []
    local_geo_normal = torch.zeros([0,batch_size,3]).to(device)
    local_geo_tangent = torch.zeros([0,batch_size,3]).to(device)
    local_geo_binormal = torch.zeros([0,batch_size,3]).to(device)
    for which_view in range(sample_view_num):
        local_points = torch.stack([local_shading_normal[which_view], local_shading_tangent[which_view], local_shading_binormal[which_view]],dim=1) # [batch,3,3]
        R = torch.matmul(local_points.permute(0,2,1),torch.inverse(global_points.permute(0,2,1)))
        tmp_local_geo_normal = torch.matmul(R,global_geo_normal.unsqueeze(dim=2)).squeeze(dim=2)
        tmp_local_geo_tangent = torch.matmul(R,global_geo_tangent.unsqueeze(dim=2)).squeeze(dim=2)
        
        local_shading_frames.append([local_shading_normal[which_view],local_shading_tangent[which_view],local_shading_binormal[which_view]])
        local_geo_normal = torch.cat([local_geo_normal, tmp_local_geo_normal.unsqueeze(dim=0)],dim=0)
        local_geo_tangent = torch.cat([local_geo_tangent, tmp_local_geo_tangent.unsqueeze(dim=0)],dim=0)
    
    n_t_xyz = torch.cat([local_geo_normal,local_geo_tangent],dim=-1).permute(1,0,2)  #[batch_size,sample_view_num,6]

    return input_params,local_input_positions, local_shading_frames, label_shading_frame, n_t_xyz, tmp_BRDF


def run(args,disturb_stddev,name,sample_view_num,setup,setup_slice_spec,setup_slice_diff,setup_output,sampled_rotate_angles_np_o,RENDER_SCALAR_CENTRE,RENDER_SCALAR,spec_loss_form,output_queue):
    mine = Mine(args,name)
    #######################################
    # define rendering module           ###
    #######################################
    multiview_render_args = {
        "available_devices":args["rendering_devices"],
        "torch_render_path":TORCH_RENDER_PATH,
        "rendering_view_num":sample_view_num,
        "setup":setup,
        "renderer_name_base":"multiview_renderer",
        "renderer_configs":["ntb","both"],
        "input_as_list":True
    }
    
    multiview_renderer = Multiview_Renderer(multiview_render_args,max_process_live_per_gpu=5)

    print("[MINE PRO PROCESS] Starting...{}".format(mine.name))
    while True:
        batch_data = mine.generate_training_data() #(batchsize,((1+self.sample_view_num)*(7+3+3+3+3)+3)) torch_renderer
        batch_size = batch_data.size()[0]
        device = batch_data.device

        input_params,input_positions, local_shading_frames, label_shading_frame, n_t_xyz, tmp_BRDF = gen_geometry_parameters(setup,setup_output,disturb_stddev,batch_data,batch_size,device,sample_view_num)

        

        center_positions = torch.zeros([batch_size,3]).to(device)
        centre_view_angles = torch.zeros([batch_size,1]).to(device)
        label_input_params = torch.cat([batch_data[:,:5],torch.mean(batch_data[:,5:8],dim=-1, keepdim=True),torch.mean(batch_data[:,8:11],dim=-1,keepdim=True)],dim=-1)
        output_diff_lumi,_ = torch_render.draw_rendering_net(setup_slice_diff,label_input_params,center_positions,centre_view_angles,"output_diff",
            global_custom_frame=label_shading_frame,
            use_custom_frame="ntb",
            pd_ps_wanted="pd_only"
        )
        output_diff_lumi *= RENDER_SCALAR_CENTRE
        output_spec_lumi,_ = torch_render.draw_rendering_net(setup_slice_spec,label_input_params,center_positions,centre_view_angles,"output_spec",
            global_custom_frame=label_shading_frame,
            use_custom_frame="ntb",
            pd_ps_wanted="ps_only"
        )
        output_spec_lumi *= RENDER_SCALAR_CENTRE


        sampled_view_angles = torch.zeros([batch_size, sample_view_num])
        multiview_lumitexel_list,end_points = multiview_renderer(input_params,input_positions,sampled_view_angles,global_frame=local_shading_frames,end_points_wanted_list=["position","form_factors"])

        
        multiview_lumitexel_list_tensor = torch.stack(multiview_lumitexel_list,dim=1)

        multiview_lumitexel_list = [a_lumi * RENDER_SCALAR for a_lumi in multiview_lumitexel_list] #a list item shape=(batchsize,lumilen,channel_num)
        
        
        # check_lumi = torch.stack(multiview_lumitexel_list,dim=1).squeeze(dim=-1) #[batch_size,64,512,1]
        # check_lumi = torch.sum(check_lumi,dim=-1)#[batch_size,64]
        # idx = torch.where(check_lumi < 1e-3)
        form_factors = [tmp_end_points["form_factors"] for tmp_end_points in end_points]
        form_factors = torch.stack(form_factors,dim=1)#.squeeze(dim=-1)  #[batch_size,sample_view_num,lumi_len,1]   
        
        
        tmp_BRDF = torch.log(tmp_BRDF*RENDER_SCALAR+1) #[batch_size,sample_view_num]
        # tmp_BRDF_max = torch.max(tmp_BRDF,dim=-1,keepdim=True)[0] #[batch_size,1]
        multiview_lumitexel_list_tensor = multiview_lumitexel_list_tensor / (form_factors+1e-6) #[batch_size,sample_view_num,lumi_len,3]
        multiview_lumitexel_list_tensor = torch.log(multiview_lumitexel_list_tensor* RENDER_SCALAR+1)
        multiview_lumitexel_list_tensor = torch.max(torch.mean(multiview_lumitexel_list_tensor,dim=-1),dim=-1)[0]    #[batch_size,sample_view_num]
        
        conf = multiview_lumitexel_list_tensor / (tmp_BRDF + 1e-6)
        
        conf_max = conf.max(dim=-1,keepdim=True)[0]#[batch_size,1]
        

        position_label_list = [tmp_end_points["position"] for tmp_end_points in end_points]

        training_data_map = {
            "normals":None,
            "tangents":None,
            "n_t_xyz":n_t_xyz,
            "input_positions":position_label_list,
            "multiview_lumitexel_list":multiview_lumitexel_list,
            "label_shading_normal":label_shading_frame[0],
            "label_shading_tangent":label_shading_frame[1],
            "centre_rendered_slice_diff_gt_list":output_diff_lumi,
            "centre_rendered_slice_spec_gt_list":output_spec_lumi,
            "confidence":conf_max
        }
        
        output_queue.put(training_data_map)

class Mine_Pro():
    def __init__(self,args,name,output_queue,output_sph):
        print("[MINE PRO {}] creating mine...".format(name))
        ##########
        ##parse arguments
        #########
        self.args = args
        self.name = name
        self.output_queue = output_queue
        self.sample_view_num = args["sample_view_num"]
        self.lighting_pattern_num = args["lighting_pattern_num"]
        self.spec_loss_form = args["spec_loss_form"]
        self.m_len = args["m_len"]
        self.lumitexel_length = args["lumitexel_length"]
        self.batch_size = args["batch_size"]
        self.training_device = args["training_device"]
        self.setup_input = args["setup_input"]
        self.setup_slice_diff = args["setup_slice_diff"]
        self.setup_slice_spec = args["setup_slice_spec"]
        self.setup_output = args["setup_output"]
        self.setup_slice_diff.set_cam_pos(np.array([0,0,356.20557],np.float32))
        self.setup_slice_spec.set_cam_pos(np.array([0,0,356.20557],np.float32))
        
        self.sampled_rotate_angles_np = np.linspace(0,math.pi*2.0/12,num=self.sample_view_num+2,endpoint=False)#[self.sample_view_num]
        self.sampled_rotate_angles_np = np.expand_dims(self.sampled_rotate_angles_np,axis=0)
        self.sampled_rotate_angles_np = self.sampled_rotate_angles_np.astype(np.float32)

        self.RENDER_SCALAR_CENTRE = 5*1e3/math.pi
        self.RENDER_SCALAR = 2*1e3/math.pi

        self.disturb_stddev = {}
        self.disturb_stddev["geo_normal"] = 0#0.5
        self.disturb_stddev["normal"] = 0.25
        self.disturb_stddev["tangent"] = 0.0#0.5
        self.disturb_stddev["axay_disturb"] = 0.15
        self.disturb_stddev["rhod"] = 0.05
        self.disturb_stddev["rhos"] = 0.05

        print("[MINE PRO {}] creating mine pro done.".format(name))

    def start(self):
        self.generator = threading.Thread(target=run, args=(
            self.args,
            self.disturb_stddev,
            self.name,
            self.sample_view_num,
            self.setup_input,
            self.setup_slice_spec,
            self.setup_slice_diff,
            self.setup_output,
            self.sampled_rotate_angles_np,
            self.RENDER_SCALAR_CENTRE,
            self.RENDER_SCALAR,
            self.spec_loss_form,
            self.output_queue
        ))
        
        self.generator.setDaemon(True)
        self.generator.start()
