import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import sys
import numpy as np
import math
import torchvision.utils as vutils
from torchvision import transforms
import torchvision

TORCH_RENDER_PATH="../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render

from torch_render import Setup_Config
from linear_projection import LinearProjection
from mine_pro import Mine_Pro
from BRDF_pointnet import BRDF_PointNet
from normal_net import NormalNet
from lumitexel_net import LumitexelNet
from confidence_net import ConfidenceNet

class ScannerNet(nn.Module):
    def __init__(self, args):
        super(ScannerNet, self).__init__()
        ########################################
        ##parse configuration                ###
        ########################################
        self.training_device = args["training_device"]
        self.rendering_device = args["rendering_devices"]
        self.sample_view_num = args["sample_view_num"]
        self.lighting_pattern_num = args["lighting_pattern_num"]
        self.m_len = args["m_len"]
        self.batch_size = args["batch_size"]
        self.lambdas = args["lambdas"]
        self.pointnet_global_feat_length = args["pointnet_global_feat_length"]
        self.keep_prob = args["keep_prob"]
        self.conf_thresh = args["conf_thresh"]

        ########################################
        ##loading setup configuration        ###
        ########################################
        self.setup_input = args["setup_input"]
        self.setup_slice_diff = args["setup_slice_diff"]
        self.setup_slice_spec = args["setup_slice_spec"]
        self.setup_output = args["setup_output"]
        
        ########################################
        ##define net modules                 ###
        ########################################
        self.l2_loss_fn = torch.nn.MSELoss(reduction='sum')
        channel = 3+3+3+self.m_len

        self.pretrained_model = args["pretrained_model"]
        self.linear_projection_pointnet_pipeline = BRDF_PointNet(args)
        self.normal_net = NormalNet(args)
        self.lumitexel_net = LumitexelNet(args)
        self.confidence_net = ConfidenceNet(args)

        
    def forward(self, batch_data,call_type="train"):
        '''
        batch_data = [batch,-1]
        '''
        ############################################################################################################################
        ## step 1 get training data
        ############################################################################################################################
        input_positions = [a.to(self.training_device) for a in batch_data["input_positions"]] 
        n_t_xyz = batch_data["n_t_xyz"].to(self.training_device)
        multiview_lumitexel_list = [a.to(self.training_device) for a in batch_data["multiview_lumitexel_list"]] # a list item shape=(batchsize,lumilen,channel_num)

        label_shading_normal =  batch_data["label_shading_normal"].to(self.training_device)
        label_shading_tangent = batch_data["label_shading_tangent"].to(self.training_device)

        confidence_label = batch_data["confidence"].to(self.training_device)
        
        confidence_label = confidence_label / self.conf_thresh
        
        confidence_label = torch.where(confidence_label > 1.0, torch.ones_like(confidence_label,device=self.training_device), confidence_label)
        
        centre_rendered_slice_diff_gt = batch_data["centre_rendered_slice_diff_gt_list"].to(self.training_device)
        centre_rendered_slice_spec_gt = batch_data["centre_rendered_slice_spec_gt_list"].to(self.training_device)

        ############################################################################################################################
        ## step 2 draw nn net
        ############################################################################################################################
        # 1 first we project every lumitexel to measurements

        measurements, local_feature, global_feature = self.linear_projection_pointnet_pipeline(input_positions,n_t_xyz,multiview_lumitexel_list)
    
        lumi_net_input = global_feature
        nn_normals = self.normal_net(lumi_net_input)
        nn_confidence = self.confidence_net(lumi_net_input)

        centre_slice_nn_spec,centre_slice_nn_diff = self.lumitexel_net(lumi_net_input)

        # nn_spec_list = []
        # nn_diff_list = []

        # nn_spec_list.append(centre_slice_nn_spec)
        # nn_diff_list.append(centre_slice_nn_diff)

        if call_type == "check_quality":
            #fetch every lumitexel from gpus
            multiview_lumitexel_list = [a_lumi.cpu() for a_lumi in multiview_lumitexel_list]
            multiview_lumitexel_tensor = torch.stack(multiview_lumitexel_list,dim=1) #(batch,sample_view_num,lumilen,channel_num) cpu
        
            rendered_slice_diff_gt_center_tensor = centre_rendered_slice_diff_gt.cpu().unsqueeze(dim=1)#(batch,1,lumilen,channel_num) cpu
            rendered_slice_spec_gt_center_tensor = centre_rendered_slice_spec_gt.cpu().unsqueeze(dim=1)#(batch,1,lumilen,channel_num) cpu
            
            nn_spec_tensor = torch.exp(centre_slice_nn_spec.detach().cpu())-1.0
            nn_spec_tensor = nn_spec_tensor.unsqueeze(dim=1) #(batch,1,lumilen,channel_num) cpu

            nn_diff_tensor = centre_slice_nn_diff.detach().cpu().unsqueeze(dim=1) #(batch,1,lumilen,channel_num) cpu
            
            lighting_pattern = [self.linear_projection_pointnet_pipeline.linear_projection.get_lighting_patterns(self.training_device,withclamp=False)]

            term_map = {
                "input_lumitexel":multiview_lumitexel_tensor,
                "lighting_pattern":lighting_pattern,
                "center_spec":[rendered_slice_spec_gt_center_tensor,nn_spec_tensor],
                "center_diff":[rendered_slice_diff_gt_center_tensor,nn_diff_tensor],
                "normals_list":[label_shading_normal,nn_normals]
            }
            term_map = self.visualize_quality_terms(term_map)
            return term_map

        ############################################################################################################################
        ## step 3 compute loss
        ############################################################################################################################
        ### !1 normal loss
        loss_normal = self.l2_loss_fn(label_shading_normal,nn_normals) 
        ### !2 diff loss
        loss_diff = self.l2_loss_fn(centre_slice_nn_diff,centre_rendered_slice_diff_gt)
        
        ### !3 spec loss
        
        gt_tensor_multi = torch.log(1.0+centre_rendered_slice_spec_gt)

        loss_spec = torch.sum(torch.sum(torch.square(centre_slice_nn_spec-gt_tensor_multi),dim=-1),dim=-1,keepdim=True)
        loss_spec = torch.sum(confidence_label*loss_spec)
        # print(loss_spec)

        # confidence_label= confidence_label.unsqueeze(dim=1)

        # loss_spec = self.l2_loss_fn(confidence_label*centre_slice_nn_spec,confidence_label*gt_tensor_multi)
        # confidence_label= confidence_label.squeeze(dim=1)

        # print(loss_spec)
        ###！4 confidence loss
        loss_confidence = self.l2_loss_fn(confidence_label,nn_confidence)


        total_loss =  loss_diff * self.lambdas["diff"] + loss_spec * self.lambdas["spec"] + loss_normal * self.lambdas["normal"] + self.lambdas["confidence"] * loss_confidence

        loss_log_map = {
            "diff":loss_diff.item(),
            "spec":loss_spec.item(),
            "normal":loss_normal.item(),
            "confidence":loss_confidence.item(),
            "total":total_loss.item()
        }
        return total_loss,loss_log_map

    def visualize_quality_terms(self,quality_map):

        result_map = {}
        
        input_lumi_tensor = quality_map["input_lumitexel"].numpy()

        gt_center_diff_tensor = quality_map["center_diff"][0].numpy()
        nn_center_diff_tensor = quality_map["center_diff"][1].numpy()

        gt_center_spec_tensor = quality_map["center_spec"][0].numpy()
        nn_center_spec_tensor = quality_map["center_spec"][1].numpy()

        gt_normal_list = quality_map["normals_list"][0].detach().cpu().numpy()
        nn_normal_list = quality_map["normals_list"][1].detach().cpu().numpy()

        position_label_list = np.zeros_like(gt_normal_list) 

        batch_size = input_lumi_tensor.shape[0]
        img_stack_list = []
        img_stack_center_list = []
        
        for which_sample in range(batch_size):
            # multiview_lumi
            img_input = torch_render.visualize_lumi(input_lumi_tensor[which_sample],self.setup_input)#(sample_view_num,imgheight,imgwidth,channel)
            
            img_center_diff_nn = torch_render.visualize_lumi(nn_center_diff_tensor[which_sample],self.setup_slice_diff,resize=True)#(1,imgheight,imgwidth,channel)
            img_center_diff_gt = torch_render.visualize_lumi(gt_center_diff_tensor[which_sample],self.setup_slice_diff,resize=True)#(1,imgheight,imgwidth,channel)

            img_center_spec_nn = torch_render.visualize_lumi(nn_center_spec_tensor[which_sample],self.setup_slice_spec,resize=True)#(1,imgheight,imgwidth,channel)
            img_center_spec_gt = torch_render.visualize_lumi(gt_center_spec_tensor[which_sample],self.setup_slice_spec,resize=True)#(1,imgheight,imgwidth,channel)
            img_center_spec_gt = torch_render.draw_vector_on_lumi(img_center_spec_gt,gt_normal_list[which_sample],position_label_list[which_sample],self.setup_output,is_batch_lumi=False,color=(1.0,0.0,0.0),resize=False,bold=1,length=5)
            img_center_spec_gt = torch_render.draw_vector_on_lumi(img_center_spec_gt,nn_normal_list[which_sample],position_label_list[which_sample],self.setup_output,is_batch_lumi=False,color=(0.0,1.0,0.0),resize=False,bold=1,length=5)

            img_center_lumi_nn = img_center_diff_nn + img_center_spec_nn
            img_center_lumi_gt = img_center_diff_gt + img_center_spec_gt
            
            img_stack_center = np.concatenate([img_center_lumi_gt,img_center_diff_gt,img_center_spec_gt,img_center_lumi_nn,img_center_diff_nn,img_center_spec_nn],axis=0)#(6,imgheight,imgwidth,channel)

            img_stack_list.append(np.expand_dims(img_input,axis=0))
            img_stack_center_list.append(img_stack_center)

        img_stack_list = np.stack(img_stack_list,axis=0)#(batchsize,6,sampleviewnum,imgheight,imgwidth,channel)
        img_stack_list = torch.from_numpy(img_stack_list)
        
        img_stack_center_list = np.stack(img_stack_center_list,axis=0)#(batchsize,6,imgheight,imgwidth,channel)
        img_stack_center_list = torch.from_numpy(img_stack_center_list)

        images_list = []
        for which_sample in range(batch_size):
            for which_group in range(img_stack_list.shape[1]):
                tmp_lumi_img = torchvision.utils.make_grid(img_stack_list[which_sample,which_group].permute(0,3,1,2),nrow=int(np.sqrt(self.sample_view_num)), pad_value=0.5)
                images_list.append(tmp_lumi_img)

        images = torch.stack(images_list,axis=0)#(batchsize*3,3,height(with padding),width(widthpadd))
        images = torch.clamp(images,0.0,1.0)
        result_map["multiview_lumi_img"] = images

        
        images_center_list = []
        for which_sample in range(batch_size):
            tmp_lumi_img = torchvision.utils.make_grid(img_stack_center_list[which_sample].permute(0,3,1,2),nrow=3, pad_value=0.5)
            images_center_list.append(tmp_lumi_img)

        images_center = torch.stack(images_center_list,axis=0)#(batchsize,3,height(with padding),width(widthpadd))
        images_center = torch.clamp(images_center,0.0,1.0)
        result_map["lumi_img_center"] = images_center

        #################################
        ###lighting patterns
        #################################

        lighting_pattern = [a_kernel.cpu().detach().numpy() for a_kernel in quality_map["lighting_pattern"]]
        
        lighting_pattern_collector = []
        for which_view in range(len(lighting_pattern)):
            lp = lighting_pattern[which_view]
            lp_pos = np.maximum(0.0,lighting_pattern[which_view])
            lp_pos_max = lp_pos.max()
            lp = torch_render.visualize_lumi(lp,self.setup_input,is_batch_lumi=False)
            lp = lp / (lp_pos_max+1e-6)
            lighting_pattern_collector.append(lp)

        lighting_pattern_collector = np.array(lighting_pattern_collector)#(lighting_pattern_num,img_height,img_width,3)
        
        result_map["lighting_patterns"] = lighting_pattern_collector

        return result_map


    def init_weight_layer(self,dim):
        W = np.eye(dim) / self.lighting_pattern_num
        W = torch.from_numpy(W.astype(np.float32))
        return W