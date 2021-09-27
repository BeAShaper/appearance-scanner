import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math

class ConfidenceNet(nn.Module):
    def __init__(self,args):
        super(ConfidenceNet,self).__init__()

        self.keep_prob = args["keep_prob"]
        self.keep_prob_diff = 0.9
        
        self.input_length = args["pointnet_global_feat_length"] #+ 3 + 3#args["sample_view_num"] * args["measurements_num"]
        self.output_length = 1
        #############construct model
        
        self.confidence_net_model = self.confidence_decoder(self.input_length)
        self.confidence_net_model = nn.Sequential(self.confidence_net_model)

    def confidence_decoder(self,input_size,name_prefix = "confidence_net_"):
        layer_stack = OrderedDict()

        layer_count = 0
        input_size = self.input_length

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

        output_size=self.output_length
        layer_stack["Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_count+=1
        input_size = output_size
        
        return layer_stack

    
    def forward(self,measurements):
        batch_size = measurements.size()[0]
        device = measurements.device
        
        x_n = measurements#[batch,sampleviewnum,mlen,1]
        x_n = x_n.reshape(batch_size,-1)#x_n.permute(0,1,3,2)

        x_n = self.confidence_net_model(x_n)#[batch,3]

        return x_n