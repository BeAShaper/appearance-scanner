import cv2
import tensorflow as tf
import numpy as np
import argparse
import sys
sys.path.append("auxiliary/tf_ggx_render_newparam/")
from tf_ggx_render_optimized import tf_ggx_render

batch_size=50

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("shot_root")
    parser.add_argument("feature_task")
    parser.add_argument("material_task")
    parser.add_argument("udt_folder_name")
    parser.add_argument("texture_folder_name")
    parser.add_argument("texture_resolution",type=int)
    parser.add_argument("slice_width",type=int)
    parser.add_argument("slice_height",type=int)
    args = parser.parse_args()

    tex_folder_root = args.shot_root+args.feature_task+"/"+args.udt_folder_name+"/"+args.texture_folder_name+"/"
    material_root = args.shot_root+args.material_task+"/images/"
    
    rot_axis = np.fromfile(tex_folder_root+"rot_axis.bin",np.float32)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        slice_rendering_parameters = {}
        slice_rendering_parameters["batch_size"] = batch_size
        slice_rendering_parameters["is_real_light"] = True
        slice_rendering_parameters["config_dir"] = "tf_ggx_render_configs_plane_new/"
        slice_rendering_parameters["slice_width"] = args.slice_width
        slice_rendering_parameters["slice_height"] = args.slice_height
        slice_rendering_parameters["lumitexel_size"] = 24576
        slice_rendering_parameters["is_real_light"] = False
        slice_rendering_parameters["shrink_step"] = 1
        slice_rendering_parameters["is_grey_scale"] = True

        renderer = tf_ggx_render(slice_rendering_parameters)

        input_rotate_thetas = tf.placeholder(tf.float32,[batch_size,1],name="input_rotate_thetas")#[batchsize]
        normal = tf.placeholder(tf.float32,[batch_size,3])
        tangent = tf.placeholder(tf.float32,[batch_size,3])

        rotate_axis = tf.constant(np.repeat(rot_axis.reshape([2,3])[[0],:],batch_size,axis=0))
        view_mat_model = renderer.rotation_axis(input_rotate_thetas*-1.0,rotate_axis)#[batch,4,4]
        view_mat_model_t = tf.matrix_transpose(view_mat_model)
        view_mat_for_normal =tf.matrix_transpose(tf.matrix_inverse(view_mat_model))
        view_mat_for_normal_t = tf.matrix_transpose(view_mat_for_normal)
        
        
        normal_4 = tf.expand_dims(tf.concat([normal,tf.ones([normal.shape[0],1],tf.float32)],axis=1),axis=1)
        normal_4 = tf.squeeze(tf.matmul(normal_4,view_mat_for_normal_t),axis=1)#position@view_mat_model_t
        normal_global,_ = tf.split(normal_4,[3,1],axis=1)#shape=[batch,3]

        tangent_4 = tf.expand_dims(tf.concat([tangent,tf.ones([tangent.shape[0],1],tf.float32)],axis=1),axis=1)
        tangent_4 = tf.squeeze(tf.matmul(tangent_4,view_mat_for_normal_t),axis=1)#position@view_mat_model_t
        tangent_global,_ = tf.split(tangent_4,[3,1],axis=1)#shape=[batch,3]

        rots = np.fromfile(material_root+"data_for_server/gathered_all/best_rotate_angle.bin",np.float32).reshape([-1,1])

        fitted_spec_params = np.fromfile(material_root+"data_for_server/gathered_all/params_gathered_gloabal_normal.bin",np.float32).reshape([-1,12])
        fitted_normals = fitted_spec_params[:,:3]
        fitted_tangents = np.fromfile(material_root+"data_for_server/gathered_all/gathered_global_tangent.bin",np.float32).reshape([-1,3])
    
        
        pf_normal_global = open(tex_folder_root+"normal_global_fitted.bin","wb")
        pf_tangent_global = open(tex_folder_root+"tangent_global_fitted.bin","wb")
        # pf_binormal_global = open(tex_folder_root+"binormal_global_net.bin","wb")

        ptr = 0
        while True:
            tmp_normal = fitted_normals[ptr:ptr+batch_size]
            tmp_tangent = fitted_tangents[ptr:ptr+batch_size]
            tmp_best_rot = rots[ptr:ptr+batch_size]
            valid_num = tmp_normal.shape[0]
            if valid_num == 0:
                break
            elif valid_num != batch_size:
                tmp_normal = np.concatenate([tmp_normal,np.zeros([batch_size-valid_num,3])],axis=0)
                tmp_tangent = np.concatenate([tmp_tangent,np.zeros([batch_size-valid_num,3])],axis=0)
                tmp_best_rot = np.concatenate([tmp_best_rot,np.zeros([batch_size-tmp_best_rot.shape[0],1])],axis=0)

            result = sess.run([
                normal_global,
                tangent_global
            ],feed_dict={
                normal:tmp_normal,
                tangent:tmp_tangent,
                input_rotate_thetas:tmp_best_rot
            })

            tmp_n_global = result[0]
            tmp_t_global = result[1]

            tmp_n_global[:valid_num].astype(np.float32).tofile(pf_normal_global)
            tmp_t_global[:valid_num].astype(np.float32).tofile(pf_tangent_global)
            
        
            ptr+=valid_num
        
        pf_normal_global.close()
        pf_tangent_global.close()
        # pf_binormal_global.close()