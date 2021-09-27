import numpy as np
import tensorflow as tf
from tf_ggx_render import tf_ggx_render
import sys
sys.path.append("../utils/")
import argparse
from parser_related import get_bool_type

parser = argparse.ArgumentParser(usage="convert local params to global")
get_bool_type(parser)
parser.add_argument("data_root")
parser.add_argument("config_file")

args = parser.parse_args()

if __name__ == "__main__":
    parameters = {}
    parameters["shrink_size"] = 1
    parameters["batch_size"] = 100
    parameters["lumitexel_size"] = 24576#24576//parameters["shrink_size"]//parameters["shrink_size"]
    parameters["is_grey_scale"] = True
    parameters["parameter_len"] = 7
    parameters["config_dir"] = args.config_file+"/"#'./tf_ggx_render_configs_{}x{}/'.format(parameters["shrink_size"],parameters["shrink_size"])#指向随包的config文件夹
    parameters["sub_sample_rate"] = 1
    parameters["slice_sample_num"] = 32

    renderer = tf_ggx_render(parameters)#实例化渲染器

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        input_params = tf.get_variable(name = "input_parameters",dtype=tf.float32,shape = [parameters["batch_size"],parameters["parameter_len"]])
        input_positions = tf.get_variable(name="position",dtype=tf.float32,shape=[parameters["batch_size"],3],trainable=False)
        input_global_normals = tf.get_variable(name="global_normal",dtype=tf.float32,shape=[parameters["batch_size"],3],trainable=False)
        rotate_thetas = tf.zeros([parameters["batch_size"],1],tf.float32)
        _ = renderer.draw_rendering_net(input_params,input_positions,rotate_thetas,"my_little_render",with_cos=False)

        frame_n_node = renderer.get_compute_node_by_name("my_little_render","frame_n")#[batch,3]
        frame_t_node = renderer.get_compute_node_by_name("my_little_render","frame_t")
        frame_b_node = renderer.get_compute_node_by_name("my_little_render","frame_b")

        n_local_x = tf.reduce_sum(tf.multiply(input_global_normals,frame_t_node),axis=-1,keepdims=True)
        n_local_y = tf.reduce_sum(tf.multiply(input_global_normals,frame_b_node),axis=-1,keepdims=True)
        n_local_z = tf.reduce_sum(tf.multiply(input_global_normals,frame_n_node),axis=-1,keepdims=True)

        n_local = tf.concat([n_local_x,n_local_y,n_local_z],axis=-1)#[batch,3]

        n_2d_node = renderer.hemi_octa_map(n_local)#[batch,2]

        init = tf.global_variables_initializer()
        sess.run(init)

        prediected_normals = np.fromfile(args.data_root+"images/normals_geo_fulllocalview.bin",np.float32).reshape([-1,3])
        positions = np.fromfile(args.data_root+"images/position_fulllocalview.bin",np.float32).reshape([-1,3])
        assert prediected_normals.shape[0] == positions.shape[0]

        n_2d_gather = np.zeros([0,2],np.float32)

        ptr = 0

        while True:
            tmp_positions = positions[ptr:ptr+parameters["batch_size"]]
            tmp_global_normals = prediected_normals[ptr:ptr+parameters["batch_size"]]
            valid = tmp_positions.shape[0]
            if tmp_positions.shape[0] == 0:
                break
            elif tmp_positions.shape[0] < parameters["batch_size"]:
                tmp_positions = np.concatenate([tmp_positions,np.zeros([parameters["batch_size"]-valid,3])],axis=0)
                tmp_global_normals = np.concatenate([tmp_global_normals,np.zeros([parameters["batch_size"]-valid,3])],axis=0)
            
            result = sess.run(
                n_2d_node,
                feed_dict={
                    input_positions:tmp_positions,
                    input_global_normals:tmp_global_normals
                }
            )
            tmp_n_2d = result
            
            n_2d_gather = np.concatenate([n_2d_gather,tmp_n_2d[:valid]],axis=0)
            ptr += valid

    n_2d_gather.astype(np.float32).tofile(args.data_root+"images/predicted_n2d.bin")