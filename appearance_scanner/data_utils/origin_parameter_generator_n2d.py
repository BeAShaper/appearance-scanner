import numpy as np
import math
import os
import sys
from math_utils import *

param_dim = 11  # n_2d,t,ax,ay,pd3,ps3
epsilon = 1e-3
param_bounds={}
param_bounds["n"] = (epsilon,1.0-epsilon)
param_bounds["theta"] = (0.0,math.pi)
param_bounds["a"] = (0.006,0.503)
param_bounds["pd"] = (0.0,1.0)
param_bounds["ps"] = (0.0,10.0)
param_bounds["box"] = (-75.0,75.0)
param_bounds["angle"] = (0.0,2.0*math.pi)

def rejection_sampling_axay(batch_size):
    origin = np.exp(np.random.uniform(np.log(param_bounds["a"][0]),np.log(param_bounds["a"][1]),[batch_size,2]))
    origin = np.where(origin[:,[0]]>origin[:,[1]],origin,origin[:,::-1])
    while True:
        still_need = np.logical_and(origin[:,0]>0.35,origin[:,1] >0.35)
        where = np.nonzero(still_need)
        num = where[0].shape[0]
        if num == 0:
            break
        new_data= np.exp(np.random.uniform(np.log(param_bounds["a"][0]),np.log(param_bounds["a"][1]),[num,2]))
        new_data = np.where(new_data[:,[0]]>new_data[:,[1]],new_data,new_data[:,::-1])
        origin[where] = new_data
    return origin

if __name__ == "__main__":
    
    data_root = sys.argv[1]
    sample_num = int(sys.argv[2])
    train_data_ratio = float(sys.argv[3])

    os.makedirs(data_root,exist_ok=True)

    train_data_file = open(data_root+"train.bin","wb")
    val_data_file = open(data_root+"val.bin","wb")

    batch_size = 1000000
    tmp_sample_num = 0

    while tmp_sample_num < sample_num:
        raw_random_numbers = np.random.uniform(0,1,[batch_size,param_dim]) 

        ns = np.ones_like(raw_random_numbers[:,:2]) * 0.5
        ts = np.zeros_like(raw_random_numbers[:,[2]]) 
        
        n2d = ns
        n3d = back_full_octa_map(n2d)

        axay = rejection_sampling_axay(batch_size).astype(np.float32)

        pds = raw_random_numbers[:,5:8]*(param_bounds["pd"][1]-param_bounds["pd"][0])+param_bounds["pd"][0]
        pss = raw_random_numbers[:,8:]*(param_bounds["ps"][1]-param_bounds["ps"][0])+param_bounds["ps"][0]

        cam_pos = np.array([[-148.547,97.8399,-226.435],[151.268,98.5092,-229.427],[-149.069,-101.236,-228.274],[153.498,-104.214,-230.302]])
        
        cooked_params = np.concatenate([n2d,ts,axay,pds,pss],axis=-1)
        print(cooked_params.shape)
        np.random.shuffle(cooked_params)

        total_size = cooked_params.shape[0]
        train_data_size = int(total_size*train_data_ratio)
        val_data_size = total_size-train_data_size
        
        # with open(data_root+"gen_log.txt","w") as plogf:
        #     plogf.write("total size:{}\ntrain size:{}\nvalidate size:{}".format(total_size,train_data_size,val_data_size))

        cooked_params_train = cooked_params[:train_data_size]
        cooked_params_val = cooked_params[-val_data_size:]

        cooked_params_train.astype(np.float32).tofile(train_data_file)
        cooked_params_val.astype(np.float32).tofile(val_data_file)

        tmp_sample_num += batch_size
    # np.random.shuffle(cooked_params)
    # np.savetxt(data_root+"tmp_spare.csv",cooked_params[:200],delimiter=',')

    train_data_file.close()
    val_data_file.close()

        



    