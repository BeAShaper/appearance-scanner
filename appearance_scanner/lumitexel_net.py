import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math

class LumitexelNet(nn.Module):
    def __init__(self,args):
        super(LumitexelNet,self).__init__()

        self.keep_prob = args["keep_prob"]
        self.keep_prob_diff = 0.9
        # self.diff_slice_length = 8*8*6#args["slice_width"] * args["slice_height"]
        # self.spec_slice_length = 32*32*6#args["slice_width"] * args["slice_height"]

        self.slice_sample_num_spec = args["slice_sample_num_spec"]
        self.spec_slice_length = self.slice_sample_num_spec*self.slice_sample_num_spec*6
        self.slice_sample_num_diff = args["slice_sample_num_diff"]
        self.diff_slice_length = self.slice_sample_num_diff*self.slice_sample_num_diff*6

        self.input_length = args["pointnet_global_feat_length"]#*args["lighting_pattern_num"] #+ 3 + 3
        #############construct model
        self.shared_part_model,output_size = self.shared_part(self.input_length)
        self.shared_part_model = nn.Sequential(self.shared_part_model)
        
        self.diff_net_model,_ = self.diff_net(output_size)
        self.spec_net_model = self.spec_net(output_size)

        self.diff_net_model = nn.Sequential(self.diff_net_model)
        self.spec_net_model = nn.Sequential(self.spec_net_model)
        
    
    def shared_part(self,input_size,name_prefix = "Share_"):
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=512
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=512
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size
        
        return layer_stack,output_size
    
    def spec_net(self,input_size,name_prefix = "Spec_"):
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=512
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=1024
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=2048
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=4096
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=4096
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=4096
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=self.spec_slice_length
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        
        return layer_stack
    
    
    def diff_net(self,input_size,name_prefix = "Diff_"):
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=512
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob_diff)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=512
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob_diff)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=512
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob_diff)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=self.diff_slice_length
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)

        return layer_stack,output_size

    def forward(self,net_input):
        '''
        vi_code = [batch,vicode_len]
        vd_code = [batch,vdcode_len]
        return = [batch_size,param_len]
        '''
        # batch_size = vd_code.size()[0]
        # device = vd_code.device

        #shared part
        # x_n = torch.cat([vi_code,vd_code],dim=1)
        x_n = net_input
        x_n = self.shared_part_model(x_n)

        #specular part
        x_n_spec = self.spec_net_model(x_n)
        slice_nn_spec = torch.unsqueeze(x_n_spec,dim=2)
        #diffuse pos part
        x_n_diff = self.diff_net_model(x_n)
        slice_nn_diff = torch.unsqueeze(x_n_diff,dim=2)
        

        return slice_nn_spec,slice_nn_diff