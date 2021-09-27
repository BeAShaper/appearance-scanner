import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math

class NormalNet(nn.Module):
    def __init__(self,args):
        super(NormalNet,self).__init__()

        self.keep_prob = args["keep_prob"]
        self.keep_prob_diff = 0.9
        
        self.input_length = args["pointnet_global_feat_length"] #+ 3 + 3#args["sample_view_num"] * args["measurements_num"]
        self.output_length = 3
        #############construct model
        
        self.normal_net_model = self.normal_decoder(self.input_length)
        self.normal_net_model = nn.Sequential(self.normal_net_model)

    
    def normal_decoder(self,input_size,name_prefix = "normal_net_"):
        layer_stack = OrderedDict()

        layer_count = 0
        input_size = self.input_length

        output_size=512
        layer_stack["Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack["LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_stack["Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        output_size=512
        layer_stack["Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack["LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_stack["Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        output_size=256
        layer_stack["Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack["LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_stack["Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        output_size=256
        layer_stack["Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack["LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_stack["Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        output_size=128
        layer_stack["Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack["LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_stack["Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        output_size=64
        layer_stack["Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack["LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_stack["Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        output_size=3
        layer_stack["Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_count+=1
        input_size = output_size
        
        
        return layer_stack

    
    def forward(self,measurements):
        batch_size = measurements.size()[0]
        device = measurements.device
        
        x_n = measurements#[batch,sampleviewnum,mlen,1]
        x_n = x_n.reshape(batch_size,-1)#x_n.permute(0,1,3,2)
        x_n = torch.nn.functional.normalize(x_n, dim=1)
        x_n = torch.reshape(x_n,[batch_size,-1])#[batch,mlen*1]

        x_n = self.normal_net_model(x_n)#[batch,3]

        normal_nn = torch.nn.functional.normalize(x_n, dim=1)      
        
        return normal_nn