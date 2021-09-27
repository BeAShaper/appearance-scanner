import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math
from linear_projection import LinearProjection
from PointNet import PointNetEncoder

class BRDF_PointNet(nn.Module):
    def __init__(self,args):
        super(BRDF_PointNet,self).__init__()

        channel = 12

        self.linear_projection = LinearProjection(args)
        self.point_net = PointNetEncoder(args,channel=channel)

    
    def forward(self,positions,n_t_xyz,lumitexels,infer=False):
        self.point_net.to(n_t_xyz.device)

        if infer:
            measurements = lumitexels
            pointnet_input = torch.cat([positions,n_t_xyz,measurements], dim=-1) # [batch,view                                                                                                                                                     _num_per_pattern,3+n+m_len]
            pointnet_input = pointnet_input.permute(0,2,1)
            local_feature, global_feature = self.point_net(pointnet_input)
            return global_feature
        
        lumitexels = torch.stack(lumitexels,dim=1) #[batch,view_num,lumi_len,3]
        
        batch_size, view_num, lumi_len, _ = lumitexels.shape
        lumitexels = lumitexels.reshape(-1,lumi_len,3)
        
        measurements = self.linear_projection(lumitexels) #[batch*view_num,1,1]
        measurements = measurements.unsqueeze(dim=-1).reshape([batch_size,view_num,3])

        positions = torch.stack(positions,dim=1)
        
        pointnet_input = torch.cat([positions,n_t_xyz,measurements], dim=-1) # [batch,view_num_per_pattern,3+n+m_len]
        pointnet_input = pointnet_input.permute(0,2,1)
        local_feature, global_feature = self.point_net(pointnet_input)

        return measurements,local_feature,global_feature