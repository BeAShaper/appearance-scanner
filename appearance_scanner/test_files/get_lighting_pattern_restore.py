import torch
import numpy as np
import os
import argparse
import time
import cv2
import sys
TORCH_RENDER_PATH = "../../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render

lumitexel_size = 512


def lighting_pattern_process(kernel):
    kernel = torch.sigmoid(kernel)
    return kernel

def quantize_pattern(pattern_float):
    if pattern_float.dtype != np.float32:
        print("[QUATIZE ERROR]error input type:",pattern_float.dtype)
    pattern_quantized = pattern_float*255.0
    pattern_quantized = pattern_quantized.astype(np.uint8)
    return pattern_quantized

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root")
    parser.add_argument("model_file_name")
    parser.add_argument("node_name")
    parser.add_argument("sample_view_num",type=int)

    args = parser.parse_args()

    checkpoint = torch.load(args.model_root + args.model_file_name, map_location='cpu')

    key_collector = []
    for a_key in checkpoint:
        print(a_key)
        if args.node_name + "linear_projection" in a_key:    
            key_collector.append(a_key)

    # print("[WARNING] we load the unadulterated parameters from model, you may processed that in forward func.")
    # print("[WARNING] You need to process them here too.")
    # print("[WARNING] Understood?(y/n)")
    # if input() != "y":
    #     exit(-1)

    cur_save_root = args.model_root+"0/"
    os.makedirs(cur_save_root,exist_ok=True)
    lighting_patterns = np.zeros([0,lumitexel_size,3],np.float32)

    for which_view,a_key in enumerate(key_collector):
        lighting_pattern = checkpoint[a_key]
        lighting_pattern = lighting_pattern_process(lighting_pattern)
        lighting_pattern = lighting_pattern.numpy()

        print("origin lighting pattern param shape:",lighting_pattern.shape)

        if len(lighting_pattern.shape) == 2:
            lighting_pattern = np.expand_dims(lighting_pattern,axis=0)
        
        if lighting_pattern.shape[-1] == 1:
            lighting_pattern = np.repeat(lighting_pattern,3,axis=-1)
        
        lighting_patterns = np.concatenate([lighting_patterns,lighting_pattern],axis=0)


    print("W.bin shape:",lighting_patterns.shape)
    lighting_patterns.astype(np.float32).tofile(cur_save_root+"W.bin")

    pattern_num = lighting_patterns.shape[0]

    maxs = np.zeros([pattern_num],np.float32)
    quantized_pattern = np.zeros(lighting_patterns.shape,np.uint8)
    for idx,a_pattern in enumerate(lighting_patterns):
        print(a_pattern.max())
        maxs[idx] = a_pattern.max()
        quantized_pattern[idx] = quantize_pattern(a_pattern/maxs[idx])
    maxs.tofile(cur_save_root+"maxs.bin")
    np.savetxt(cur_save_root+"maxs.txt",maxs,delimiter=',',fmt='%.3f')
    quantized_pattern.tofile(cur_save_root+"W_quantized.bin")


    standard_rendering_parameters = {
            "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/handheld_device_render_config_16x32/"
        }
    setup = torch_render.Setup_Config(standard_rendering_parameters)
    
    with open(cur_save_root+"W_quantized.bin","rb") as f:
        data = np.fromfile(f,np.uint8).reshape([-1,lumitexel_size,3])
        img = torch_render.visualize_lumi(data,setup)

        for i in range(img.shape[0]):
            
            cv2.imwrite(cur_save_root+"W_{}.png".format(i),img[i][:,:,::-1])

        