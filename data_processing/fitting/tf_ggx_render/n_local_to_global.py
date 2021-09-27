import numpy as np
import tensorflow as tf
from tf_ggx_render import tf_ggx_render
import sys
sys.path.append("../utils/")
TORCH_RENDER_PATH="../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render
from torch_render import Setup_Config
import torch
import argparse
from parser_related import get_bool_type

parser = argparse.ArgumentParser(usage="convert local params to global")
get_bool_type(parser)
parser.add_argument("data_root")
parser.add_argument("texture_folder_name")

parser.add_argument("config_file")
parser.add_argument("is_for_server",type="bool")
parser.add_argument("sub_folder_name")



args = parser.parse_args()

if __name__ == "__main__":
    parameters = {}
    parameters["shrink_size"] = 1
    parameters["batch_size"] = 10000
    parameters["slice_width"] = 256
    parameters["slice_height"] = 192
    parameters["lumitexel_size"] = 24576//parameters["shrink_size"]//parameters["shrink_size"]
    parameters["is_grey_scale"] = True
    parameters["parameter_len"] = 7
    parameters["config_dir"] = args.config_file+"/"#'./tf_ggx_render_configs_{}x{}/'.format(parameters["shrink_size"],parameters["shrink_size"])#指向随包的config文件夹
    parameters["sub_sample_rate"] = 1
    parameters["slice_sample_num"] = 32

    renderer = tf_ggx_render(parameters)#实例化渲染器

    # standard_rendering_parameters = {
    #     "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_plane_render_configs/"
    # }
    # setup_input = Setup_Config(standard_rendering_parameters)
    # device = "cuda:0"
    # random_rotate = np.fromfile(args.data_root+args.texture_folder_name+"/random_rotate.bin",np.float32).reshape([-1,60,3])
    # random_rotate = torch.from_numpy(random_rotate).to(device)

    # best_view_id = np.fromfile(args.data_root+args.texture_folder_name+args.sub_folder_name+"selected_views.bin",np.int32).reshape([-1,16])
    # best_cam_id = np.fromfile(args.data_root+args.texture_folder_name+args.sub_folder_name+"best_cam_id_{}.bin".format(args.kind),np.int32).reshape([-1,16])
    # best_cam_id = np.expand_dims(best_cam_id[:,0],axis=-1)
    
    # texel_num = best_cam_id.shape[0]
    # texel_indices = np.arange(texel_num)[:,None]
    # best_cam_id = best_view_id[texel_indices,best_cam_id]
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        input_params = tf.get_variable(name = "input_parameters",dtype=tf.float32,shape = [parameters["batch_size"],parameters["parameter_len"]])
        input_positions = tf.get_variable(name="position",dtype=tf.float32,shape=[parameters["batch_size"],3],trainable=False)
        rotate_thetas = tf.zeros([parameters["batch_size"],1],tf.float32)
        _ = renderer.draw_rendering_net(input_params,input_positions,rotate_thetas,"my_little_render",with_cos=False)

        n_global_node = renderer.get_global_normal("my_little_render")
        t_global_node = renderer.get_compute_node_by_name("my_little_render","tangent")
        b_global_node = renderer.get_compute_node_by_name("my_little_render","binormal")

        init = tf.global_variables_initializer()
        sess.run(init)

        root = "data_for_server" if args.is_for_server else "data_for_local"

        fitted_params = np.fromfile(args.data_root+"images_{}/{}/gathered_all/params_gathered.bin".format(args.sub_folder_name,root),np.float32).reshape([-1,11])
        positions = np.fromfile(args.data_root+"images_{}/{}/gathered_all/position_gathered.bin".format(args.sub_folder_name,root),np.float32).reshape([-1,3])
        assert fitted_params.shape[0] == positions.shape[0]

        n_global_gather = np.zeros([0,3],np.float32)
        tangent_gather = np.zeros([0,3],np.float32)

        sequence = np.arange(fitted_params.shape[0])
        ptr = 0

        while True:
            tmp_sequence = sequence[ptr:ptr+parameters["batch_size"]]
            if tmp_sequence.shape[0] == 0:
                break

            tmp_params = fitted_params[tmp_sequence]
            tmp_positions = positions[tmp_sequence]
            # tmp_best_cam_id = best_cam_id[tmp_sequence]
            # tmp_batch_indices = texel_indices[tmp_sequence]
            # tmp_rotate_angles = torch.squeeze(random_rotate[tmp_batch_indices,tmp_best_cam_id],dim=1)
            
            valid = tmp_params.shape[0]
            if tmp_params.shape[0] == 0:
                break
            elif tmp_params.shape[0] < parameters["batch_size"]:
                tmp_params = np.concatenate([tmp_params,np.zeros([parameters["batch_size"]-valid,tmp_params.shape[1]])],axis=0)
                tmp_positions = np.concatenate([tmp_positions,np.zeros([parameters["batch_size"]-valid,3])],axis=0)
            
            result = sess.run(
                [
                    n_global_node,
                    t_global_node,
                    b_global_node
                ],feed_dict={
                    input_params:tmp_params[:,:7],
                    input_positions:tmp_positions
                }
            )
            tmp_n_global = result[0][:valid]
            tmp_t_global = result[1][:valid]
            tmp_b_global = result[2][:valid]

            tmp_t_global = np.where(tmp_params[:valid,[3]]>tmp_params[:valid,[4]],tmp_t_global,tmp_b_global)

            dot_result = np.sum(tmp_t_global*np.array([0.0,0.0,1.0]),axis=1,keepdims=True)

            tmp_t_global = np.where(dot_result>0.0,tmp_t_global,-1.0*tmp_t_global)
            
            # tmp_n_global = torch.from_numpy(tmp_n_global).to(device)
            # tmp_t_global = torch.from_numpy(tmp_t_global).to(device)

            # AXIS = [2,1,0] 
            # for which_axis in AXIS:
            #     setup_input.set_rot_axis_torch(which_axis)
            #     tmp_n_global = torch_render.rotate_vector_along_axis(setup_input,-tmp_rotate_angles[:,[which_axis]],tmp_n_global,is_list_input=False)
            #     tmp_t_global = torch_render.rotate_vector_along_axis(setup_input,-tmp_rotate_angles[:,[which_axis]],tmp_t_global,is_list_input=False)
            
            # tmp_n_global = tmp_n_global.cpu().numpy()
            # tmp_t_global = tmp_t_global.cpu().numpy()

            n_global_gather = np.concatenate([n_global_gather,tmp_n_global],axis=0)
            tangent_gather = np.concatenate([tangent_gather,tmp_t_global],axis=0)
            ptr += valid
            print(ptr)
    
    axy_inverted = fitted_params[:,3:5]
    axy_inverted = axy_inverted[:,::-1]
    fitted_params[:,3:5] = np.where(fitted_params[:,[3]]>fitted_params[:,[4]],fitted_params[:,3:5],axy_inverted)

    res = np.concatenate(
        [
            n_global_gather,
            fitted_params[:,2:]
        ],
        axis=1
    )
        
    res.astype(np.float32).tofile(args.data_root+"images_{}/{}/gathered_all/params_gathered_gloabal_normal.bin".format(args.sub_folder_name,root))
    tangent_gather = tangent_gather
    tangent_gather.astype(np.float32).tofile(args.data_root+"images_{}/{}/gathered_all/gathered_global_tangent.bin".format(args.sub_folder_name,root))