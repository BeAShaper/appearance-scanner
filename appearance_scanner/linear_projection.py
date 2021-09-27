import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearProjection(nn.Module):
    def __init__(self,args):
        super(LinearProjection,self).__init__()
        
        self.m_len = args["m_len"]
        self.lumitexel_length = args["lumitexel_length"]
        self.noise_stddev = args["noise_stddev"]
        self.device_cpu = torch.device("cpu")
        self.channel = self.m_len
        self.color_tensor = args["color_tensor"].reshape([9,3])
        
        self.kernel = torch.nn.Parameter(
            data = self.init_lighting_pattern(),#torch.nn.init.xavier_normal_(torch.empty(self.measurements_length,self.lumitexel_length)),
            requires_grad=args["train_lighting_pattern"]#True
        )
    
    def gaussian_random_matrix(self,n_components, n_features, random_state=None):
        components = np.random.normal(loc=0.0,
                                    scale=1.0 / np.sqrt(n_components),
                                    size=(n_components, n_features))
        
        return components

    def getW(self,lumi_len, K):
        random = self.gaussian_random_matrix(lumi_len, K)
        random_reshape = np.reshape(random, (lumi_len, K))
        return random_reshape

    def init_lighting_pattern(self):
        W = self.getW(self.lumitexel_length, self.channel)
        W = torch.from_numpy(W.astype(np.float32))
        return W
    
    def get_lighting_patterns(self,device,withclamp=True):
        tmp_kernel = self.kernel      
        return torch.sigmoid(tmp_kernel)

    def forward(self,lumitexels,add_noise=True):

        batch_size = lumitexels.shape[0]
        device = lumitexels.device

        measurement_list = []

        lumitexels = lumitexels.unsqueeze(dim=2) #[batch*view_num,lumi_len,1,3]
        kernel = self.get_lighting_patterns(device).unsqueeze(dim=-1)#[lumi_len,3,1]
        tmp_kernel = kernel.unsqueeze(dim=0)     #(1,lumi_len,3,1)
        tmp_measurement = tmp_kernel*lumitexels  #(batch*view_num,lumi_len,3,3)


        tmp_measurement = torch.sum(tmp_measurement,dim=1).reshape(batch_size,1,9)#[batch*view_num,lumi_len,3,3] => [batch*view_num,3,3] => [batch*view_num,1,9] 
        tmp_measurement = torch.matmul(tmp_measurement,self.color_tensor).unsqueeze(dim=1) #[batch*view_num,3]

        if add_noise:
            tmp_noise = torch.randn_like(tmp_measurement)*self.noise_stddev+1.
            tmp_measurements_noised = tmp_measurement*tmp_noise
            return tmp_measurements_noised
        else:
            return tmp_measurement

