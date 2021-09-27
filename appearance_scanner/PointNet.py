import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

class PointNetEncoder(nn.Module):
    def __init__(self,args,channel=3,global_feature=False, points_transform=False, feature_transform=False):
        super(PointNetEncoder, self).__init__()

        self.local_len = args["pointnet_local_feat_length"]
        self.global_len = args["pointnet_global_feat_length"]

        self.local_part_model = self.local_part(channel)
        self.local_part_model = nn.Sequential(self.local_part_model)

        self.global_part_model,output_size = self.global_part(self.local_len)
        self.global_part_model = nn.Sequential(self.global_part_model)

        self.global_compress_part_model = self.global_compress_part(output_size)
        self.global_compress_part_model = nn.Sequential(self.global_compress_part_model)

        self.global_feature = global_feature
        self.points_transform = points_transform
        self.feature_transform = feature_transform
        self.channel = channel

        if points_transform:
            self.stn = STN3d(channel)

        if self.feature_transform:
            self.fstn = STNkd(k=64)
    
    def local_part(self,input_size,name_prefix = "pointnet_local_"):
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=64
        layer_stack[name_prefix+"Conv_{}".format(layer_count)] = torch.nn.Conv1d(input_size, output_size, 1)
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=64
        layer_stack[name_prefix+"Conv_{}".format(layer_count)] = torch.nn.Conv1d(input_size, output_size, 1)
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=64
        layer_stack[name_prefix+"Conv_{}".format(layer_count)] = torch.nn.Conv1d(input_size, output_size, 1)
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        
        return layer_stack

    def global_part(self,input_size,name_prefix = "pointnet_global_"):
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=128
        layer_stack[name_prefix+"Conv_{}".format(layer_count)] = torch.nn.Conv1d(input_size, output_size, 1)
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=512
        layer_stack[name_prefix+"Conv_{}".format(layer_count)] = torch.nn.Conv1d(input_size, output_size, 1)
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=1024
        layer_stack[name_prefix+"Conv_{}".format(layer_count)] = torch.nn.Conv1d(input_size, output_size, 1)
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_count+=1

        return layer_stack,output_size

    def global_compress_part(self,input_size,name_prefix = "pointnet_compress_"):
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=1024
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = torch.nn.Linear(input_size, output_size)
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=1024
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = torch.nn.Linear(input_size, output_size)
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=self.global_len
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = torch.nn.Linear(input_size, output_size)
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()

        return layer_stack
    
    def forward(self, x):
        B, D, N = x.size()  # [batch, channel+feature, npoints]

        if self.points_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1) # [batch, npoints, channel+feature]

            if D > 3:
                x, feature = x.split(3, dim=2)
            x = torch.bmm(x, trans)

            if D > 3:
                x = torch.cat([x, feature], dim=2)
            x = x.transpose(2, 1)  # [batch, channel+feature, npoints]
        else:
            trans = None

        x = self.local_part_model(x)
        
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        local_feature = x


        x = self.global_part_model(x)
        x = torch.max(x, 2, keepdim=True)[0] # [batch, 1024, 1]   # change max_pooling to LIP pooling 5.20
        x = x.view(-1, 1024)
        global_feature = self.global_compress_part_model(x)
        
        
        if self.global_feature:
            return global_feature #, trans, trans_feat
        else:
            # x = x.view(-1, self.global_len, 1).repeat(1, 1, N) # [batch, 1024, npoints]
            return local_feature, global_feature  # torch.cat([x, pointfeat], 1), trans, trans_feat
