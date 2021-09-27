import torch
import torch.nn as nn
import torch_render
import numpy as np
import threading

class Rendering_Thread(threading.Thread):
    def __init__(self,setup,name,rendering_configs,device,thread_ctr_map,use_global_frame):
        threading.Thread.__init__(self)
        print("forked rendering process:{}".format(name))
        self.setup = setup
        self.name = name
        self.rendering_configs = rendering_configs
        self.device = device
        self.use_global_frame = use_global_frame

        self.thread_ctr_map = thread_ctr_map
        self.device_sph = self.thread_ctr_map["device_sph"]
        self.io_list_id = self.thread_ctr_map["io_list_id"]
        self.input_list = self.thread_ctr_map["input_holder"]
        self.output_list = self.thread_ctr_map["output_holder"]
    
    def run(self):
        print("[RENDERING WORKER {}] Starting".format(self.name))
        while True:
            self.thread_ctr_map["input_sph"].acquire()
            self.device_sph.acquire()
            #data to worker's device
            tmp_input_params = self.input_list[self.io_list_id][0].to(self.device)
            tmp_input_positions = self.input_list[self.io_list_id][1].to(self.device)
            tmp_rotate_theta = self.input_list[self.io_list_id][2].to(self.device)
            tmp_shared_frame = [a.to(self.device) for a in self.input_list[self.io_list_id][3]] if self.use_global_frame else None
        
            #render here
            tmp_lumi,end_points = torch_render.draw_rendering_net(
                self.setup,
                tmp_input_params,
                tmp_input_positions,
                tmp_rotate_theta,
                self.name,
                tmp_shared_frame,
                *self.rendering_configs
            )

            self.output_list[self.io_list_id] = [tmp_lumi,end_points]
            self.thread_ctr_map["output_sph"].release()
            self.device_sph.release()

class Multiview_Renderer(nn.Module):
    def __init__(self,args,max_process_live_per_gpu=4):
        '''
        max_process_live_per_gpu:
            how many live process can run in one gpu
        '''
        super(Multiview_Renderer,self).__init__()
    
        ########################################
        ##parse configuration                ###
        ########################################
        self.available_devices = args["available_devices"]
        self.available_devices_num = len(self.available_devices)
        self.rendering_view_num = args["rendering_view_num"]
        if self.rendering_view_num < self.available_devices_num:
            self.available_devices = self.available_devices[:self.rendering_view_num]
            self.available_devices_num = len(self.available_devices)
        self.view_device_ratio = self.rendering_view_num // self.available_devices_num
        self.setup = args["setup"]
        self.use_global_frame = True if (len(args["renderer_configs"]) > 0) else False
        self.renderer_name_base = args["renderer_name_base"]
        self.renderer_configs = args["renderer_configs"]#rotate point rotate normal etc.
        self.input_as_list = args["input_as_list"]

        #######################################
        ## construct renderer               ###
        #######################################
        self.device_sph_list = []
        for which_device in range(self.available_devices_num):
            self.device_sph_list.append(threading.Semaphore(max_process_live_per_gpu))

        self.input_holder = [None]*self.available_devices_num
        self.output_holder = [None]*self.available_devices_num

        self.renderer_list = []
        self.input_sph_list = []
        self.output_sph = threading.Semaphore(0)
        for which_renderer in range(self.available_devices_num):
            input_sph = threading.Semaphore(0)
            self.input_sph_list.append(input_sph)

            cur_device_id = which_renderer%self.available_devices_num
            cur_device = self.available_devices[cur_device_id]
            cur_semaphore = self.device_sph_list[cur_device_id]


            thread_ctr_map = {
                "input_sph":input_sph,
                "output_sph":self.output_sph,
                "input_holder":self.input_holder,
                "output_holder":self.output_holder,
                "io_list_id":which_renderer,
                "device_sph":cur_semaphore
            }

            tmp_renderer = Rendering_Thread(
                self.setup,
                self.renderer_name_base+"_{}".format(which_renderer),
                self.renderer_configs,
                cur_device,
                thread_ctr_map,
                self.use_global_frame
            )
            tmp_renderer.setDaemon(True)
            tmp_renderer.start()
            self.renderer_list.append(tmp_renderer)
        

    def forward(self,input_params,input_positions,rotate_theta,global_frame = None,return_tensor = False,end_points_wanted_list=[]):
        '''
        input_params=(batch_size,7 or 11) torch tensor maybe a list if self.input_as_list 
        input_positions=(batch_size,3) torch tensor maybe a list if self.input_as_list
        rotate_theta=(batch_size,rendering_view_num)
        global_frame=[(batch_size,3),(batch_size,3),(batch_size,3)] torch tensor maybe a list if self.input_as_list

        return = 
            if return_tensor = True:
                (batch, rendering_view_num, lightnum, channel_num)
            else:
                list of (batch,lightnum,channel_num) each of them on the specific gpu
        if return_tensor:
            returned tensor will be placed on where input_params is
        else
            item of returned tensor list will be placed on where it rendered
        '''
        
        ############################################################################################################################
        ##step 0 unpack batch data
        ############################################################################################################################
        if self.input_as_list:
            batch_size = input_params[0].size()[0]
            origin_device = input_params[0].device
            all_param_dim = input_params[0].size()[1]
        else:
            batch_size = input_params.size()[0]
            origin_device = input_params.device
            assert input_positions.size()[0] == batch_size,"input_params shape:{} input_positions shape:{}".format(input_params.size(),input_positions.size())
            all_param_dim = input_params.size()[1]
        
        assert all_param_dim == 11 or all_param_dim == 7,"input param dim should be 11 or 7 now:{}".format(all_param_dim)
        channel_num = 3 if all_param_dim == 11 else 1
        
        ############################################################################################################################
        ##step 2 rendering
        ############################################################################################################################
        cur_view_num_list = []
        for which_renderer in range(self.available_devices_num):
            start_idx = which_renderer*self.view_device_ratio
            end_idx =  self.rendering_view_num if (which_renderer == self.available_devices_num-1) else (which_renderer+1)*self.view_device_ratio
            
            tmp_rotate_theta = rotate_theta[:,start_idx:end_idx]#(batchsize,curviewnum)
            tmp_rotate_theta = tmp_rotate_theta.permute(1,0)
            cur_view_num = tmp_rotate_theta.size()[0]
            cur_view_num_list.append(cur_view_num)
            tmp_rotate_theta = tmp_rotate_theta.reshape(cur_view_num*batch_size,1)
            
            if self.input_as_list:
                tmp_input_params = torch.cat(input_params[start_idx:end_idx],dim=0)
                tmp_input_positions = torch.cat(input_positions[start_idx:end_idx],dim=0)
                
                if self.use_global_frame:
                    tmp_global_frames = global_frame[start_idx:end_idx]
                    tmp_global_frame_n = torch.cat([a_frame[0] for a_frame in tmp_global_frames],dim=0)
                    tmp_global_frame_t = torch.cat([a_frame[1] for a_frame in tmp_global_frames],dim=0)
                    tmp_global_frame_b = torch.cat([a_frame[2] for a_frame in tmp_global_frames],dim=0)

                    tmp_global_frame = [tmp_global_frame_n,tmp_global_frame_t,tmp_global_frame_b]
            else:
                tmp_input_params = torch.unsqueeze(input_params,dim=0).repeat(cur_view_num,1,1).reshape(cur_view_num*batch_size,all_param_dim)
                tmp_input_positions = torch.unsqueeze(input_positions,dim=0).repeat(cur_view_num,1,1).reshape(cur_view_num*batch_size,3)
                if self.use_global_frame:
                    tmp_global_frame = [torch.unsqueeze(an_item,dim=0).repeat(cur_view_num,1,1).reshape(cur_view_num*batch_size,3) for an_item in global_frame]
            #tmp_input_params = (batchsize*cur_view_num,param_len)
            #tmp_input_positions = (batchsize*cur_view_num,3)
            
            self.input_holder[which_renderer] = [tmp_input_params,tmp_input_positions,tmp_rotate_theta.clone()]
            if self.use_global_frame:
                self.input_holder[which_renderer].append(tmp_global_frame)
                
            self.input_sph_list[which_renderer].release()

        for i in range(self.available_devices_num):
            self.output_sph.acquire()

        ############################################################################################################################
        ##step 3 grab_all_rendered_result
        ############################################################################################################################
        result_tensor = torch.empty(self.rendering_view_num,batch_size,self.setup.get_light_num(),channel_num,device=origin_device) if return_tensor else [None]*self.rendering_view_num
        result_end_points_list = []
        for i in range(self.rendering_view_num):
            result_end_points_list.append({})
        '''
        return_tensor shape: 
            list of#(batchsize,lumilen,channel_num) or
            #(rendering_view_num,batchsize,lumilen,channel_num)
        '''
        view_ptr = 0
        for which_renderer in range(self.available_devices_num):
            cur_view_num = cur_view_num_list[which_renderer]
            tmp_block_rendered_lumi = self.output_holder[which_renderer][0]#(batchsize*curview,lumilen,channel)
            tmp_block_rendered_lumi = tmp_block_rendered_lumi.reshape(cur_view_num,batch_size,tmp_block_rendered_lumi.size()[1],channel_num)

            if return_tensor:
                result_tensor[view_ptr:view_ptr+cur_view_num] = tmp_block_rendered_lumi.to(origin_device,copy=True)
            else:
                for local_id in range(cur_view_num):
                    result_tensor[view_ptr+local_id] = tmp_block_rendered_lumi[local_id]
            
            for a_key in end_points_wanted_list:
                wanted_block = self.output_holder[which_renderer][1][a_key].to(origin_device,copy=True)#(curview*batchsize,unkownsize)
                wanted_block = wanted_block.reshape(cur_view_num,batch_size,*(wanted_block.size()[1:]))
                for local_id in range(cur_view_num):
                    result_end_points_list[view_ptr+local_id][a_key] = wanted_block[local_id]

            view_ptr = view_ptr+cur_view_num
        if return_tensor:
            result_tensor = result_tensor.permute(1,0,2,3)#(batchsize,rendering_view_num,lumilen,channel_num)
        
        return result_tensor,result_end_points_list