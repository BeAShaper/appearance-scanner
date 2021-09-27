from mine_pro import Mine_Pro
import torch
from torch.utils.tensorboard import SummaryWriter 
import torchvision
import torch.optim as optim
import argparse
import time
import sys
import random
import numpy as np
import os
import cv2
from PIL import Image  

TORCH_RENDER_PATH="../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
from torch_render import Setup_Config
import queue
from scanner_net import ScannerNet
groups = {}
groups["0"] = "0_gt"
groups["1"] = "1_gt_pd" 
groups["2"] = "2_gt_ps"

img_num = 1
MAX_ITR = 5000000
VALIDATE_ITR = 5
CHECK_QUALITY_ITR=1000
SAVE_MODEL_ITR=100000
LOG_MODEL_ITR=100000

def log_loss(writer,loss_terms,global_step,is_training):
    train_val_postfix = "_train" if is_training else "_val"
    for a_key in loss_terms:
        writer.add_scalar('Loss/{}'.format(a_key)+train_val_postfix, loss_terms[a_key], global_step)

def log_quality(writer,quality_terms,global_step):
    trans = torchvision.transforms.Resize(100)

    term_key = "multiview_lumi_img"
    term_key_p = "lumi_img_center"
    
    batch_size = quality_terms[term_key].shape[0] // img_num
    
    # writer.add_images(term_key, quality_terms[term_key], global_step=global_step, dataformats='NHWC')
    for which_sample in range(batch_size):
        for which_group in range(img_num):
            tmp_img = quality_terms[term_key][which_sample*img_num+which_group].permute(1,2,0)
            writer.add_image("{}/{}_{}".format(term_key,which_sample,groups["{}".format(which_group)]), tmp_img,  global_step=global_step, dataformats='HWC')
        writer.add_image("{}/{}_center".format(term_key,which_sample), quality_terms[term_key_p][which_sample],  global_step=global_step, dataformats='CHW')
    
    term_key = "lighting_patterns"
    lighting_pattern_num = quality_terms[term_key].shape[0]
    
    for which_pattern in range(lighting_pattern_num):
        writer.add_image("{}/{}".format(term_key,which_pattern),quality_terms[term_key][which_pattern], global_step=global_step, dataformats='HWC')


if __name__ == "__main__":
    random_seed = 123
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    np.set_printoptions(precision=3,threshold=np.inf,suppress=True,linewidth=120)

    ##########################################
    ### parser training configuration
    ##########################################
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("slice_width",type=int)
    parser.add_argument("slice_height",type=int)
    parser.add_argument("--sample_method",type=str,default="random")
    parser.add_argument("--sample_ratio",type=int,default=64)
    parser.add_argument("--training_gpu",type=int,default=0)
    parser.add_argument("--sample_view_num",type=int,default=64)
    parser.add_argument("--lighting_pattern_num",type=int,default=1)
    parser.add_argument("--m_len",type=int,default=3)
    parser.add_argument("--dropout_rate",type=float,default=0.7)
    parser.add_argument("--m_noise_rate",type=float,default=0.05)
    parser.add_argument("--log_dir",type=str,default="../runs/")
    parser.add_argument("--pretrained_model_pan",type=str,default="")

    args = parser.parse_args()
    
    
    ## about rendering devices
    standard_rendering_parameters = {
        "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/handheld_device_render_config_16x32/"
    }
    setup_input = Setup_Config(standard_rendering_parameters)

    standard_rendering_parameters["config_dir"] = TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_cube_slice_32x32/"
    setup_output_spec = Setup_Config(standard_rendering_parameters)

    standard_rendering_parameters["config_dir"] = TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_cube_slice_8x8/"
    setup_output_diff = Setup_Config(standard_rendering_parameters)

    standard_rendering_parameters["config_dir"] = TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_cube_slice_64x64/"
    setup_output = Setup_Config(standard_rendering_parameters)

    ## build train_configs
    train_configs = {}
    train_configs["rendering_devices"] = [torch.device("cuda:{}".format(args.training_gpu))] # for multiple GPU
    train_configs["training_device"] = torch.device("cuda:{}".format(args.training_gpu))
    train_configs["sample_view_num"] = args.sample_view_num
    train_configs["lighting_pattern_num"] = args.lighting_pattern_num
    train_configs["m_len"] = args.m_len
    train_configs["train_lighting_pattern"] = True
    train_configs["spec_loss_form"] = "direct" #"ps_one" "direct"
    train_configs["lumitexel_length"] = 512
    train_configs["pointnet_local_feat_length"] = 64
    train_configs["pointnet_global_feat_length"] = 512
    train_configs["conf_thresh"] = 0.5

    train_configs["noise_stddev"] = args.m_noise_rate
    train_configs["keep_prob"] = args.dropout_rate
    train_configs["setup_input"] = setup_input
    train_configs["setup_slice_spec"] = setup_output_spec
    train_configs["setup_slice_diff"] = setup_output_diff
    train_configs["setup_output"] = setup_output
    train_configs["color_tensor"] = setup_input.get_color_tensor(train_configs["training_device"])
    train_configs["pretrained_model"] = args.pretrained_model_pan
    train_configs["slice_shrink_step_spec"] = 2
    train_configs["slice_sample_num_spec"] = 64 // train_configs["slice_shrink_step_spec"]
    train_configs["slice_shrink_step_diff"] = 8
    train_configs["slice_sample_num_diff"] = 64 // train_configs["slice_shrink_step_diff"]

    lambdas = {}  # loss weight
    lambdas["diff"] = 1.0
    lambdas["spec"] = 1e-2
    lambdas["normal"] = 1.0
    lambdas["confidence"] = 5e2


    train_configs["lambdas"] = lambdas

    train_configs["data_root"] = args.data_root
    train_configs["batch_size"] = 50
    train_configs["pre_load_buffer_size"] = 500000

    ##########################################
    ### data loader
    ########################################## 
    train_queue = queue.Queue(25)
    val_queue = queue.Queue(10)
    train_mine = Mine_Pro(train_configs,"train",train_queue,None)
    train_mine.start()
    val_mine = Mine_Pro(train_configs,"val",val_queue,None)
    val_mine.start()

    ##########################################
    ### net and optimizer
    ##########################################
    training_net = ScannerNet(train_configs)
    training_net.to(train_configs["training_device"])
    optimizer = optim.Adam(training_net.parameters(), lr=1e-4)

    ##########################################
    ### logs
    ##########################################
    writer = SummaryWriter(args.log_dir)
    log_dir = writer.get_logdir()
    log_dir_model = log_dir+"/models/"
    os.makedirs(log_dir_model,exist_ok=True)
    train_file_bak = log_dir+"/training_files/"
    os.makedirs(train_file_bak,exist_ok=True)

    os.system("scp *.py "+train_file_bak)
    os.system("scp *.sh "+train_file_bak)
    os.system("scp *.bat "+train_file_bak)

    with open(log_dir_model+"model_params.txt","w") as pf:
        pf.write("{}".format(training_net))
        pf.write("-----------------")
        for parameter in training_net.parameters():
            pf.write("{}\n".format(parameter.shape))

    start_step = 0
    ##########################################
    ### load models
    ##########################################
    if args.pretrained_model_pan != "":
        print("loading trained model...")
        state = torch.load(args.pretrained_model_pan_con, map_location=torch.device('cpu'))
        for key, v in enumerate(state):
            print(key, v)
        
        training_net.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])

        start_step = state['epoch']
        training_net.to(train_configs["training_device"])
        print("done.")

    ##########################################
    ### training part
    ##########################################
    for global_step in range(start_step, MAX_ITR):
        if global_step % 100 == 0:
            print("global step:{}".format(global_step))
        ## 1 validate
        if global_step % VALIDATE_ITR == 0:
            val_data = val_queue.get()
            training_net.eval()
            with torch.no_grad():
                _,loss_log_terms = training_net(val_data)
            log_loss(writer,loss_log_terms,global_step,False)

        ## 2 check quality
        if global_step % CHECK_QUALITY_ITR == 0:
            val_data = val_queue.get()
            training_net.eval()
            with torch.no_grad():
                quality_terms = training_net(val_data,call_type="check_quality")
            log_quality(writer,quality_terms,global_step)

        ## 3 save model
        if global_step % SAVE_MODEL_ITR == 0:
            training_state = {
                'epoch': global_step,
                'state_dict': training_net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(training_state, log_dir_model+"training_state.pkl",_use_new_zipfile_serialization=False)
            
        if global_step % LOG_MODEL_ITR == 0 :
            torch.save(training_net.state_dict(), log_dir_model+"model_state_{}.pkl".format(global_step),_use_new_zipfile_serialization=False)

        ## 4 training
        training_net.train()
        optimizer.zero_grad()
        train_data = train_queue.get()
        
        total_loss,loss_log_terms = training_net(train_data)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(training_net.parameters(),800.0)
        optimizer.step()
        
        loss_log_terms["lr"] = optimizer.param_groups[0]['lr']
        log_loss(writer,loss_log_terms,global_step,True)
        
    writer.close()
