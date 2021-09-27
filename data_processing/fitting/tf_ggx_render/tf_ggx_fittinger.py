from tf_ggx_render import tf_ggx_render
import sys
sys.path.append("../utils/")
from lumitexel_related import visualize_cube_slice_init,visualize_cube_slice,expand_img,visualize_plane
from dir_folder_and_files import make_dir
import tensorflow as tf
import numpy as np
import math
import scipy
import cv2
import argparse
import time
# from tf_ggx_render_utils import visualize_init,visualize_new,make_dir
import scipy.optimize as opt

MAX_TRY_ITR = 5
PARAM_BOUNDS=(
                (0.0,1.0),#n1
                (0.0,1.0),#n2
                (0.0,2*math.pi),#theta
                (0.006,0.503),#ax
                (0.006,0.503),#ay
                (0.0,1.0),#pd
                (0.0,10.0),#ps
                )
RENDER_SCALAR =  5*1e3/math.pi
FITTING_SCALAR = 1.0/RENDER_SCALAR

def __compute_init_pdps(for_what,be_fitted,standard_lumi):
    '''
    for_what = "pd" or "ps"
    be_fitted = [batch,light_num]. MEANING: the data to be fitted
    '''

    #[step2] rendering
    rendered_lumitexel = np.squeeze(standard_lumi,axis=-1)#[batch,light_num]
    be_fitted_idxs = np.argsort(be_fitted)
    if for_what == "pd":
        psd_idxs = be_fitted_idxs#be_fitted_idxs[:,:self.lumitexel_size//2]
    elif for_what == "ps":
        psd_idxs = be_fitted_idxs[:,self.lumitexel_size//2:]
    else:
        print("[ERROR]unsupported init")
        exit()

    x_axis_index=np.tile(np.arange(len(rendered_lumitexel)), (psd_idxs.shape[1],1)).transpose()
    part_lumitexels_befitted = be_fitted[x_axis_index,psd_idxs]
    part_lumitexels_rendered = rendered_lumitexel[x_axis_index,psd_idxs]
    
    downside = np.einsum('ij, ij->i', part_lumitexels_rendered, part_lumitexels_rendered)
    upperside = np.einsum('ij, ij->i', part_lumitexels_rendered, part_lumitexels_befitted)
    scalers = upperside/downside
    return scalers

def __fitting_pdps_from_grey(one_of_rgb_lumi,standard_lumi_d,standard_lumi_s,pdps):
    '''
    one_of_rgb_lumi = [batch,light_num]
    standard_lumi_d = [batch,light_num,1]
    standard_lumi_s = [batch,light_num,1]
    pdps = [batch,2,1]
    '''
    b = tf.expand_dims(one_of_rgb_lumi,axis=-1)#[batch,light_num,1]
    a = tf.concat([standard_lumi_d,standard_lumi_s],axis=-1)#[batch,light_num,2]
    at = tf.transpose(a,perm=[0,2,1])#[batch,2,light_num]
    atb = tf.matmul(at,b)#[batch,2,1]
    ata = tf.matmul(at,a)#[batch,2,2]
    ata_pdps = tf.matmul(ata,pdps)#[batch,2,1]

    pdps_loss = tf.nn.l2_loss(tf.squeeze(ata_pdps-atb,axis=2))#[1]

    return pdps_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("thread_id",type=int)
    parser.add_argument("data_for_server_path")
    parser.add_argument("pd_config_name")
    parser.add_argument("ps_config_name")
    parser.add_argument("--need_dump",action="store_true")
    parser.add_argument("--free_spec_normal",action="store_true")
    # parser.add_argument("--origin_colorful",action="store_true")

    args = parser.parse_args()

    thread_id = args.thread_id #int(sys.argv[1])
    data_path = args.data_for_server_path+"{}/".format(thread_id)#sys.argv[2]+"{}/".format(thread_id)
    log_path = data_path+"logs_thread{}/".format(thread_id)#sys.argv[3]
    make_dir(log_path)
    log_start_idx = 0#thread_id*int(sys.argv[4])
    # origin_colorful = args.origin_colorful#int(sys.argv[3]) == 1
    need_dump = args.need_dump#int(sys.argv[4]) == 1
    #data_file_name_base = sys.argv[5]
    if_use_guessed_param = False#int(sys.argv[6]) == 1
    pd_config_file_name = args.pd_config_name#sys.argv[7]
    ps_config_file_name = args.ps_config_name#sys.argv[7]
    #fitting_pd_ps_or_both = sys.argv[8] #pd_only ps_only both "CAUTION! you can only choose ps_only now!"
    pd_slice_sample_num = int(pd_config_file_name.split("_")[-1].split('x')[-1])
    ps_slice_sample_num = int(ps_config_file_name.split("_")[-1].split('x')[-1])
    # sub_sample_rate = int(sys.argv[10])
    # edge_sample_num = int(sys.argv[11])
    #fix_normal = sys.argv[10] == "1"
    ######################################################
    ###prepare fitting
    ######################################################
    #initialize visualization
    visualize_cube_slice_init("../utils/",pd_slice_sample_num)
    visualize_cube_slice_init("../utils/",ps_slice_sample_num)

    #initialize render params
    parameters_pd = {}
    parameters_pd["shrink_size"] = 1
    parameters_pd["batch_size"] = 1
    parameters_pd["is_grey_scale"] = True
    parameters_pd["parameter_len"] = 7
    parameters_pd["slice_sample_num"] = pd_slice_sample_num
    parameters_pd["sub_sample_rate"] = 64 // pd_slice_sample_num
    parameters_pd["lumitexel_size"] = pd_slice_sample_num*pd_slice_sample_num*6 #24576//parameters["shrink_size"]//parameters["shrink_size"]
    parameters_pd["is_grey_scale"] = True
    parameters_pd["config_dir"] = pd_config_file_name+"/"#'./tf_ggx_render_configs_{}x{}/'.format(parameters["shrink_size"],parameters["shrink_size"])#指向随包的config文件夹

    parameters_ps = parameters_pd.copy()
    parameters_ps["slice_sample_num"] = ps_slice_sample_num
    parameters_ps["sub_sample_rate"] = 64 // ps_slice_sample_num
    parameters_ps["lumitexel_size"] = ps_slice_sample_num*ps_slice_sample_num*6 #24576//parameters["shrink_size"]//parameters["shrink_size"]
    parameters_ps["is_grey_scale"] = True
    parameters_ps["config_dir"] = ps_config_file_name+"/"#'./tf_ggx_render_configs_{}x{}/'.format(parameters["shrink_size"],parameters["shrink_size"])#指向随包的config文件夹

    #initialize renderer
    renderer_pd = tf_ggx_render(parameters_pd)
    renderer_ps = tf_ggx_render(parameters_ps)
    
    #initialize file names
    pd_data_file_name = "pd_slice.bin"
    ps_data_file_name = "ps_slice_normalized.bin"
    position_file_name = "position.bin"
    normal_file_name = "n_2d.bin"

    #open files
    pf_fitting_data_pd = open(data_path+pd_data_file_name,"rb")
    pf_fitting_data_pd.seek(0,2)
    lumitexel_num_pd = pf_fitting_data_pd.tell()//4//parameters_pd["lumitexel_size"]
    pf_fitting_data_pd.seek(0,0)

    pf_fitting_data_ps = open(data_path+ps_data_file_name,"rb")
    pf_fitting_data_ps.seek(0,2)
    lumitexel_num_ps = pf_fitting_data_ps.tell()//4//parameters_ps["lumitexel_size"]
    pf_fitting_data_ps.seek(0,0)

    assert lumitexel_num_pd == lumitexel_num_ps,"num of lumitexels for diffuse and specular are not equal. diffuse:{} specular:{}".format(lumitexel_num_pd,lumitexel_num_ps)

    lumitexel_num_grey = lumitexel_num_pd
    RESULT_params = np.zeros([lumitexel_num_grey,7],np.float32)
    
    print("lumitexel num:",lumitexel_num_grey)
    
    #init positions as origin
    fitting_data_positions = np.zeros((lumitexel_num_ps,3),np.float32)
    
    ######################################################
    ### main logic 
    ######################################################
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        ##########################################
        ####[STEP 1] draw rendering graph
        ##########################################
        ####################
        ####define inputs
        ####################
        # pd part
        input_params_without_ps_axay_t = tf.get_variable(name = "input_params_pd",dtype=tf.float32,shape = [parameters_pd["batch_size"],3])
        n1_n2,input_pd = tf.split(input_params_without_ps_axay_t,[2,1],axis=1)
        input_ps_useless = tf.zeros_like(input_pd)
        input_axay_useless = tf.ones_like(n1_n2)
        input_t_useless = tf.zeros_like(input_pd)
        input_params_pd = tf.concat([n1_n2,input_t_useless,input_axay_useless,input_pd,input_ps_useless],axis=-1)

        #ps part
        # if fix_normal:
        if args.free_spec_normal:
            input_params_without_pd_normal = tf.get_variable(name = "input_params_ps",dtype=tf.float32,shape=[parameters_ps["batch_size"],6])
            n1_n2_spec,input_params_without_pd_ps_normal,input_ps = tf.split(input_params_without_pd_normal,[2,3,1],axis=1)#input_params_without_pd_ps_normal:n2,t,axay
            input_pd_useless = tf.zeros_like(input_ps)
            input_params_ps = tf.concat([n1_n2_spec,input_params_without_pd_ps_normal,input_pd_useless,input_ps],axis=-1)
            input_params_all = tf.concat([n1_n2_spec,input_params_without_pd_ps_normal,input_pd,input_ps],axis=-1)#(to get the result)
        else:
            input_params_without_pd_normal = tf.get_variable(name = "input_params_ps",dtype=tf.float32,shape=[parameters_ps["batch_size"],4])
            input_params_without_pd_ps_normal,input_ps = tf.split(input_params_without_pd_normal,[3,1],axis=1)#input_params_without_pd_ps_normal:t,axay
            input_pd_useless = tf.zeros_like(input_ps)
            input_params_ps = tf.concat([n1_n2,input_params_without_pd_ps_normal,input_pd_useless,input_ps],axis=-1)
        
            input_params_all = tf.concat([n1_n2,input_params_without_pd_ps_normal,input_pd,input_ps],axis=-1)#(to get the result)

        input_positions = tf.get_variable(name="position",dtype=tf.float32,shape=[parameters_pd["batch_size"],3],trainable=False)
        input_labels_pd = tf.get_variable(name = "input_labels_pd" ,dtype=tf.float32, shape = [parameters_pd["batch_size"],parameters_pd["lumitexel_size"]],trainable=False)
        input_labels_ps = tf.get_variable(name = "input_labels_ps" ,dtype=tf.float32, shape = [parameters_ps["batch_size"],parameters_ps["lumitexel_size"]],trainable=False)

        input_labels_pd_deform = tf.get_variable(name = "input_labels_pd_deform" ,dtype=tf.float32, shape = [parameters_pd["batch_size"],parameters_pd["lumitexel_size"]],trainable=False)
        input_labels_ps_deform = tf.get_variable(name = "input_labels_ps_deform" ,dtype=tf.float32, shape = [parameters_ps["batch_size"],parameters_ps["lumitexel_size"]],trainable=False)
        
        ########################################
        ####draw rendering net & init & loss
        ########################################
        rotate_thetas = tf.zeros([parameters_pd["batch_size"],1],tf.float32)
        rendered_res_pd = renderer_pd.draw_rendering_net(input_params_pd,input_positions,rotate_thetas,"my_little_render_pd",with_cos=False,pd_ps_wanted="pd_only")
        rendered_res_ps = renderer_ps.draw_rendering_net(input_params_ps,input_positions,rotate_thetas,"my_little_render_ps",with_cos=False,pd_ps_wanted="ps_only")
        
        #deform factor
        r_2_cos_node_pd = tf.squeeze(renderer_pd.get_r_2_cos_node("my_little_render_pd"),axis=2)#(batchsize,lightnum)
        r_2_cos_node_ps = tf.squeeze(renderer_ps.get_r_2_cos_node("my_little_render_ps"),axis=2)#(batchsize,lightnum)
        input_labels_pd_deform_tmp = input_labels_pd*r_2_cos_node_pd
        input_labels_ps_deform_tmp = input_labels_ps*r_2_cos_node_ps
        init_input_labels_pd = tf.assign(input_labels_pd_deform,input_labels_pd_deform_tmp)#(batchsize,lightnum)
        init_input_labels_ps = tf.assign(input_labels_ps_deform,input_labels_ps_deform_tmp)#(batchsize,lightnum)
        init_input_labels = [init_input_labels_pd,init_input_labels_ps]

        # n_dot_view_dir,n_dot_view_penalty = rendered_res_pd.calculate_n_dot_view("my_little_render")
        _,init_params_pd0 = renderer_pd.param_initializer(input_labels_pd,"my_little_render_pd")
        init_params_ps0,_ = renderer_ps.param_initializer(input_labels_ps,"my_little_render_ps")

        l2_loss_pd = tf.nn.l2_loss(tf.reshape(input_labels_pd_deform,[parameters_pd["batch_size"],parameters_pd["lumitexel_size"],1])-rendered_res_pd)
        l2_loss_ps = tf.nn.l2_loss(tf.reshape(input_labels_ps_deform,[parameters_ps["batch_size"],parameters_ps["lumitexel_size"],1])-rendered_res_ps)
    
        total_loss_pd = l2_loss_pd#+n_dot_view_penalty
        total_loss_ps = l2_loss_ps#+n_dot_view_penalty
    
        #######################
        ####define optimizers
        #######################
        from tf_ggx_render_utils import print_loss,print_step,init_step,report_step
        epsilon = 1e-6
        # elif fitting_pd_ps_or_both =="pd_only":
        #pd part   
        bounds_pd ={
            input_params_without_ps_axay_t: ([epsilon,epsilon,0.0], [1.0-epsilon,1.0-epsilon,1.0])
        }
        optimizer_pd = tf.contrib.opt.ScipyOptimizerInterface(
            total_loss_pd,
            # options={'maxfun':9999999999,'maxiter': 100,"maxls":20},
            # var_list = [input_ps],
            options={'maxiter': 100},
            var_to_bounds=bounds_pd,
            method='L-BFGS-B'
        )

        # elif fitting_pd_ps_or_both == "ps_only":
        #ps part
        if args.free_spec_normal:
            bounds_ps = {
                input_params_without_pd_normal:([0.0,0.0,0.0,0.006,0.006,0.0], [1.0,1.0,2*math.pi,0.503,0.503,10.0])
            }
        else:
            bounds_ps = {
                input_params_without_pd_normal:([0.0,0.006,0.006,0.0], [2*math.pi,0.503,0.503,10.0])
            }
        optimizer_ps = tf.contrib.opt.ScipyOptimizerInterface(
            total_loss_ps,
            # options={'maxfun':9999999999,'maxiter': 100,"maxls":20},
            # var_list = [input_ps],
            options={'maxiter': 100},
            var_to_bounds=bounds_ps,
            method='L-BFGS-B'
        )
        
        ##########################################
        ####[STEP 2] init constants
        ##########################################
        #global variable init should be called before rendering
        init = tf.global_variables_initializer()
        sess.run(init)
        
        ##########################################
        #####[STEP 2]fitting here
        ##########################################
        plogf = open(log_path+"logs.txt","w")
        fitting_time_cost_preparation = 0
        fitting_time_cost_nta = 0
        fitting_time_cost_pdps = 0
        tmp_from = 0
        total_time_cost= 0
        # pfittedlumitexelf = open(log_path+"fitted.bin","wb")
        for idx in range(tmp_from,lumitexel_num_grey):
            if idx % 100 == 0:
                print("[THREAD{}]{}/{} cost time:{}s".format(thread_id,idx,lumitexel_num_grey,total_time_cost))
            a_lumi_pd = np.fromfile(pf_fitting_data_pd,np.float32,count=parameters_pd["lumitexel_size"])*FITTING_SCALAR
            a_lumi_ps = np.fromfile(pf_fitting_data_ps,np.float32,count=parameters_ps["lumitexel_size"])*FITTING_SCALAR
            ##########################################
            ####[STEP 2.1] compute init
            ##########################################
            #load labels an position
            START_TIME = time.time()
            input_positions.load(fitting_data_positions[[idx]],sess)#TODO CAN BE SET
            input_labels_pd.load(np.expand_dims(a_lumi_pd,axis=0),sess)#a_lumi=[24576] ->[1,24576]
            input_labels_ps.load(np.expand_dims(a_lumi_ps,axis=0),sess)#a_lumi=[24576] ->[1,24576]
            
            #de formfactor here
            sess.run(init_input_labels)

            #[INITIALIZE]calculate initial params and load data
            #pd
            standard_params_pd = np.repeat(np.array([
                0.5,0.5,0.0,0.0,0.0,0.0,0.0
            ],np.float32).reshape((-1,7)),parameters_pd["batch_size"],axis=0)
            #ps
            standard_params_ps = sess.run(init_params_ps0)
            standard_params_ps[:,5] = 0.0#pds*0.5#np.expand_dims(pds,1)#[batch,1]
            standard_params_ps[:,6] = 0.0#pds*0.5#np.zeros_like(pss,np.float32)#np.expand_dims(pss,1)#[batch,1]

            input_params_without_ps_axay_t.load(standard_params_pd[:,[0,1,5]],sess)
            if args.free_spec_normal:
                input_params_without_pd_normal.load(standard_params_ps[:,[0,1,2,3,4,6]],sess)
            else:
                input_params_without_pd_normal.load(standard_params_ps[:,[2,3,4,6]],sess)

            fitting_time_cost_preparation +=time.time() - START_TIME

            ##########################################
            ####[STEP 2.2] fitting here
            ##########################################
            START_TIME = time.time()

            ####
            ##fitting diffuse first
            ####
            for try_itr in range(MAX_TRY_ITR):
                init_step()
                optimizer_pd.minimize(
                    sess,
                    step_callback=print_step,
                    # loss_callback=print_loss,
                    # fetches=[total_loss, input_params]
                )
                if report_step() == 1:
                    # print("[ERROR]recompute!#######################")
                    x0 = standard_params_pd*np.random.normal(1.0,0.01,size=standard_params_pd.shape)
                    input_params_without_ps_axay_t.load(x0[:,[0,1,5]],sess)
                else:
                    break
            ####
            ##fitting specular then
            ####
            for try_itr in range(MAX_TRY_ITR):
                init_step()
                optimizer_ps.minimize(
                    sess,
                    step_callback=print_step,
                    # loss_callback=print_loss,
                    # fetches=[total_loss, input_params]
                )
                if report_step() == 1:
                    # print("[ERROR]recompute!#######################")
                    x0 = standard_params_ps*np.random.normal(1.0,0.01,size=standard_params_ps.shape)
                    if args.free_spec_normal:
                        input_params_without_pd_normal.load(x0[:,[0,1,2,3,4,6]],sess)
                    else:
                        input_params_without_pd_normal.load(x0[:,[2,3,4,6]],sess)
                else:
                    break
            

            fitting_time_cost_nta +=time.time() - START_TIME

            # print("done.")
            fitted_params_grey = sess.run(input_params_all)#[batch,7]
            RESULT_params[tmp_from+idx] = fitted_params_grey.reshape([-1])
            
            if need_dump or idx % 500 == 0:
                input_lumi_pd_deform_np = np.reshape(sess.run(input_labels_pd_deform),[-1])
                input_lumi_ps_deform_np = np.reshape(sess.run(input_labels_ps_deform),[-1])
                fitted_lumi_pd_np = np.reshape(sess.run(rendered_res_pd),[-1])
                fitted_lumi_ps_np = np.reshape(sess.run(rendered_res_ps),[-1])

                input_img_pd = visualize_cube_slice(input_lumi_pd_deform_np,scalerf=RENDER_SCALAR*3e-3,sample_num=pd_slice_sample_num)
                input_img_pd = expand_img(input_img_pd,step=8,method="copy_only")
                fitted_img_pd = visualize_cube_slice(fitted_lumi_pd_np,scalerf=RENDER_SCALAR*3e-3,sample_num=pd_slice_sample_num)
                fitted_img_pd = expand_img(fitted_img_pd,step=8,method="copy_only")
                input_img_ps = visualize_cube_slice(input_lumi_ps_deform_np,scalerf=RENDER_SCALAR*3e-3,sample_num=ps_slice_sample_num)
                fitted_img_ps = visualize_cube_slice(fitted_lumi_ps_np,scalerf=RENDER_SCALAR*3e-3,sample_num=ps_slice_sample_num)
                
                cv2.imwrite(log_path+"img{}_o_pd.png".format(log_start_idx+idx),input_img_pd)
                cv2.imwrite(log_path+"img{}_f_pd.png".format(log_start_idx+idx),fitted_img_pd)
                cv2.imwrite(log_path+"img{}_o_ps.png".format(log_start_idx+idx),input_img_ps)
                cv2.imwrite(log_path+"img{}_f_ps.png".format(log_start_idx+idx),fitted_img_ps)
            
            if idx % 5000 == 0:
                RESULT_params.astype(np.float32).tofile(log_path+"fitted_{}_log_{}.bin".format(thread_id,idx))
            
            total_time_cost = fitting_time_cost_preparation+fitting_time_cost_nta+fitting_time_cost_pdps

        total_time_cost = fitting_time_cost_preparation+fitting_time_cost_nta+fitting_time_cost_pdps
        plogf.write("TIME COST(preparation):{}s\n".format(fitting_time_cost_preparation))
        plogf.write("TIME COST(nta):{}s\n".format(fitting_time_cost_nta))
        plogf.write("TIME COST(pdps):{}s\n".format(fitting_time_cost_pdps))
        plogf.write("TIME COST(total):{}s\n".format(total_time_cost))

        plogf.close()

    RESULT_params.astype(np.float32).tofile(data_path+"fitted.bin")