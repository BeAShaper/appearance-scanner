import numpy as np
import sys
sys.path.append("../utils/")
from dir_folder_and_files import make_dir
from lumitexel_related import visualize_cube_slice,visualize_cube_slice_init
from tf_ggx_render import tf_ggx_render
import argparse
import math
import tensorflow as tf

parser = argparse.ArgumentParser(usage="process the fitting result of normalized ps\n fitting pd")
parser.add_argument("data_root")
parser.add_argument("pd_sample_num",type=int)
parser.add_argument('pd_slice_config_file_name')
parser.add_argument('pd_slice_data_file_name')
parser.add_argument("position_file_name")

args = parser.parse_args()

RENDER_SCALAR =  5*1e3/math.pi

if __name__ == "__main__":
    fitted_ps_param = np.fromfile(args.data_root+"fitted_ps.bin",np.float32).reshape([-1,11])
    # guessed_rgb_ps = np.fromfile(args.data_root+"ps_length_predicted.bin",np.float32).reshape([-1,3])
    # fitted_ps_lengths = np.fromfile(args.data_root+"fitted_ps_lengths.bin",np.float32).reshape([-1,1])
    # assert fitted_ps_param.shape[0] == guessed_rgb_ps.shape[0]

    normals = fitted_ps_param[:,:2]
    normals.astype(np.float32).tofile(args.data_root+"fitted_n2_split.bin")
    ts = fitted_ps_param[:,[2]]
    ts.astype(np.float32).tofile(args.data_root+"fitted_t_split.bin")
    axay = fitted_ps_param[:,3:5]
    axay.astype(np.float32).tofile(args.data_root+"fitted_axay_split.bin")

    ps = fitted_ps_param[:,8:]
    # ps = guessed_rgb_ps*ps/fitted_ps_lengths

    ps.astype(np.float32).tofile(args.data_root+"fitted_ps_split.bin")

    normals=None
    guessed_rgb_ps = None
    ts=None
    axay = None
    ps = None
    ##############################
    ############process pd
    ##############################
    parameters = {}
    parameters["shrink_size"] = 1
    parameters["batch_size"] = 100*3
    parameters["lumitexel_size"] = 512 #args.pd_sample_num*args.pd_sample_num*6#24576//parameters["shrink_size"]//parameters["shrink_size"]
    parameters["is_grey_scale"] = True
    parameters["parameter_len"] = 7
    parameters["config_dir"] = args.pd_slice_config_file_name+"/"#'./tf_ggx_render_configs_{}x{}/'.format(parameters["shrink_size"],parameters["shrink_size"])#指向随包的config文件夹
    parameters["sub_sample_rate"] = 1
    parameters["slice_sample_num"] = args.pd_sample_num
    # visualize_cube_slice_init("../utils/")

    # renderer = tf_ggx_render(parameters)#实例化渲染器

    # pf_fitting_data = open(args.data_root+args.pd_slice_data_file_name+".bin","rb")
    # pf_fitting_data.seek(0,2)
    # lumitexel_num = pf_fitting_data.tell()//4//parameters["lumitexel_size"]
    # pf_fitting_data.seek(0,0)

    # fitting_data_positions = np.fromfile(args.data_root+args.position_file_name,np.float32).reshape([-1,3])

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # with tf.Session(config = config) as sess:
    #     input_fitting_data = tf.placeholder(tf.float32,[parameters["batch_size"],parameters["lumitexel_size"]])

    #     input_params = tf.placeholder(tf.float32,[parameters["batch_size"],parameters["parameter_len"]])

    #     input_positions = tf.placeholder(tf.float32,shape=[parameters["batch_size"],3])
    #     rotate_thetas = tf.zeros([parameters["batch_size"],1],tf.float32)
    #     rendered_res = renderer.draw_rendering_net(input_params,input_positions,rotate_thetas,"my_little_render",with_cos=True,pd_ps_wanted = "pd_only")#[batch,lumitexel_size,1]
    #     rendered_res = rendered_res * RENDER_SCALAR
    #     rendered_res = tf.squeeze(rendered_res,axis=2)#[batch,lumitexel_size]

    #     ri = tf.reduce_sum(input_fitting_data*rendered_res,axis=1,keepdims=True)#[batch,1] rendered*input
    #     rr = tf.reduce_sum(rendered_res*rendered_res,axis=1,keepdims=True)#[batch,1] rendered*rendered
        
    #     pd_node = ri/rr

    #     init = tf.global_variables_initializer()
    #     sess.run(init)

    #     pf_fitting_pd = open(args.data_root+"fitted_pd_split.bin","wb")
    #     ###################
    #     data_ptr = 0
    #     while True:
    #         if data_ptr == lumitexel_num:
    #             break
    #         if data_ptr + parameters["batch_size"] > lumitexel_num:
    #             tmp_pd_slices = np.fromfile(pf_fitting_data,np.float32).reshape([-1,parameters["lumitexel_size"]])
    #         else:
    #             tmp_pd_slices = np.fromfile(pf_fitting_data,np.float32,parameters["batch_size"]*parameters["lumitexel_size"]).reshape([-1,parameters["lumitexel_size"]])

    #         valid_num = tmp_pd_slices.shape[0]
    #         tmp_positions = np.repeat(np.expand_dims(fitting_data_positions[data_ptr//3:data_ptr//3+valid_num//3],axis=1),3,axis=1).reshape([-1,3])
    #         tmp_params = np.repeat(np.expand_dims(fitted_ps_param[data_ptr//3:data_ptr//3+valid_num//3],axis=1),3,axis=1).reshape([-1,11])[:,:7]
    #         tmp_params[:,5] = 1.0#pd=1
    #         tmp_params[:,6] = 0.0#ps=0.0

    #         if valid_num < parameters["batch_size"]:
    #             tmp_pd_slices = np.concatenate([tmp_pd_slices,np.zeros([parameters["batch_size"]-valid_num,parameters["lumitexel_size"]],np.float32)],axis=0)
    #             tmp_positions = np.concatenate([tmp_positions,np.zeros([parameters["batch_size"]-valid_num,3],np.float32)],axis=0)
    #             tmp_params = np.concatenate([tmp_params,0.5*np.ones([parameters["batch_size"]-valid_num,parameters["parameter_len"]],np.float32)],axis=0)


    #         pds = sess.run(
    #             pd_node,
    #             feed_dict={
    #                 input_fitting_data:tmp_pd_slices,
    #                 input_params:tmp_params,
    #                 input_positions:tmp_positions
    #             }
    #         )

    #         pds = pds[:valid_num,:]

    #         pds.astype(np.float32).tofile(pf_fitting_pd)
    #         data_ptr += valid_num

    #     pf_fitting_pd.close()
    

    normals = np.fromfile(args.data_root+"fitted_n2_split.bin",np.float32).reshape([-1,2])
    ts = np.fromfile(args.data_root+"fitted_t_split.bin",np.float32).reshape([-1,1])
    axay = np.fromfile(args.data_root+"fitted_axay_split.bin",np.float32).reshape([-1,2])
    pd = np.fromfile(args.data_root+"fitted_pd_split.bin",np.float32).reshape([-1,3])
    ps = np.fromfile(args.data_root+"fitted_ps_split.bin",np.float32).reshape([-1,3])

    res = np.concatenate([
        normals,ts,axay,pd,ps
    ],axis=-1)

    res.astype(np.float32).tofile(args.data_root+"fitted.bin")
