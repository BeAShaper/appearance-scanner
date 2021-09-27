import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import sys
import numpy as np
import math
import torchvision.utils as vutils
from torchvision import transforms
import torchvision
import queue

sys.path.append("../")
TORCH_RENDER_PATH="../../torch_renderer/"
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
        self.lighting_pattern_num = args.lighting_pattern_num
        self.m_len = args.m_len
        self.batch_size = args.batch_size
        self.pointnet_global_feat_length = 512
        self.pointnet_local_feat_length = 64
        self.keep_prob = 0.9
        
        ########################################
        ##define net modules                 ###
        ########################################

        channel = 3+3+3+self.m_len
        
        training_configs = {
            "lighting_pattern_num" : args.lighting_pattern_num,
            "m_len" : args.m_len,
            "keep_prob" : self.keep_prob,
            "lumitexel_length" : 512,
            "pointnet_global_feat_length":self.pointnet_global_feat_length,
            "pointnet_local_feat_length":self.pointnet_local_feat_length,
            "noise_stddev":0.01,
            "train_lighting_pattern":False,
            "slice_sample_num_spec":args.spec_sample_num,
            "slice_sample_num_diff":args.diff_sample_num,
            "color_tensor":np.zeros([27,1])
        }

        self.linear_projection_pointnet_pipeline = BRDF_PointNet(training_configs)
        self.normal_net = NormalNet(training_configs)
        self.lumitexel_net = LumitexelNet(training_configs)
        self.confidence_net = ConfidenceNet(training_configs)


    def forward(self, batch_data,call_type="train"):
        '''
        batch_data = [batch,-1]
        '''
        
        result_map = {}

        input_positions = batch_data["input_positions"]   # [batch_size,sample_view_num,3]
        n_t_xyz = batch_data["n_t_xyz"]
        measurements = batch_data["measurements"]        # [batch_size,sample_view_num,measurement_len,1]
        batch_size = input_positions.shape[0]

        global_feature_tensor = self.linear_projection_pointnet_pipeline(input_positions,n_t_xyz,measurements,infer=True)     
        
        nn_normals = self.normal_net(global_feature_tensor)
        nn_confidence = self.confidence_net(global_feature_tensor)
        centre_slice_nn_spec,centre_slice_nn_diff = self.lumitexel_net(global_feature_tensor)
        centre_slice_nn_spec = torch.exp(centre_slice_nn_spec) - 1.0
        
        result_map["spec_slice"] = centre_slice_nn_spec
        result_map["diff_slice"] = centre_slice_nn_diff
        result_map["nn_normals"] = nn_normals
        result_map["nn_confidence"] = nn_confidence
            
        return result_map