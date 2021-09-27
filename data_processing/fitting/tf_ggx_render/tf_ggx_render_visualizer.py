import numpy as np
import math
import cv2
import sys
sys.path.append("../utils/")
from lumitexel_related import shrink,expand_img
import tensorflow as tf
from tf_ggx_fittinger import tf_ggx_render
from tf_ggx_render_utils import visualize_init,visualize_new,make_dir

RENDER_SCALAR =  5*1e3/math.pi

if __name__ == "__main__":
    thread_id = int(sys.argv[1])
    data_root = sys.argv[2]
    img_root = data_root+"rendered_thread{}/".format(thread_id)
    make_dir(img_root)
    origin_colorful = int(sys.argv[3]) == 1
    shrink_size = 1
    parameters = {}
    parameters["batch_size"] = 10
    if origin_colorful:
        parameters["batch_size"]*=3
    parameters["lumitexel_size"] = 24576//shrink_size//shrink_size
    parameters["is_grey_scale"] = True
    parameters["config_dir"] = './tf_ggx_render_configs_{}x{}/'.format(shrink_size,shrink_size)#指向随包的config文件夹

    visualize_init(parameters["config_dir"]+"visualize_idxs.bin",parameters["lumitexel_size"])

    if parameters["is_grey_scale"]:  
        parameters["parameter_len"] = 7
    else:  
        parameters["parameter_len"] = 11
    
    renderer = tf_ggx_render(parameters)#实例化渲染器


    if origin_colorful:
        test_params = np.fromfile(data_root+"fitted_{}.bin".format(thread_id),np.float32).reshape([-1,11])
        rgbs_params_nta = np.tile(test_params[:,:5].reshape([-1,1,5]),[1,3,1])
        rgbs_params_pd = test_params[:,5:8].reshape([-1,3,1])
        rgbs_params_ps = test_params[:,8:].reshape([-1,3,1])
        test_params = np.concatenate([rgbs_params_nta,rgbs_params_pd,rgbs_params_ps],axis=-1)#[-1,3,7]
    else:
        test_params = np.fromfile(data_root+"fitted_{}.bin".format(thread_id),np.float32).reshape([-1,7])

    print("point num:",test_params.shape[0])

    with tf.Session() as sess:
        #define inputs
        input_params = tf.placeholder(tf.float32,shape=[parameters["batch_size"],parameters["parameter_len"]],name="render_params")
        input_positions = tf.placeholder(tf.float32,shape=[parameters["batch_size"],3],name="render_positions")

        #draw rendering net
        rendered_res = renderer.draw_rendering_net(input_params,input_positions,"my_little_render")
        
        #global variable init should be called before rendering
        init = tf.global_variables_initializer()
        sess.run(init)

        data_ptr = 0
        real_batch_size = parameters["batch_size"]//3 if origin_colorful else parameters["batch_size"]
        counter = 0
        while True:
            tmp_params = test_params[data_ptr:data_ptr+real_batch_size]
            
            if tmp_params.shape[0] == 0:
                break
            valid_num = tmp_params.shape[0]
            tmp_params = tmp_params.reshape([-1,7])
            result = sess.run(rendered_res,feed_dict={
                input_params: tmp_params,
                input_positions :np.zeros([tmp_params.shape[0],3],np.float32)
            })
            result = result.reshape([-1,3,parameters["lumitexel_size"]])
            
            for idx,a_colorful_lumi in enumerate(result):
                img = []
                img.append(expand_img(shrink(visualize_new(a_colorful_lumi[0],len = 64//shrink_size,scalerf=RENDER_SCALAR*255.0),8,"sum"),8))
                img.append(expand_img(shrink(visualize_new(a_colorful_lumi[1],len = 64//shrink_size,scalerf=RENDER_SCALAR*255.0),8,"sum"),8))
                img.append(expand_img(shrink(visualize_new(a_colorful_lumi[2],len = 64//shrink_size,scalerf=RENDER_SCALAR*255.0),8,"sum"),8))
                # img.append(visualize_new(a_colorful_lumi[1],len = 64//shrink_size,scalerf=RENDER_SCALAR*255.0))
                # img.append(visualize_new(a_colorful_lumi[2],len = 64//shrink_size,scalerf=RENDER_SCALAR*255.0))
                img = np.asarray(img)
                img = np.transpose(img,axes=[1,2,0])[:,:,::-1]
                cv2.imwrite(img_root+"{}.png".format(counter),img*0.5)
                counter+=1
            data_ptr+=valid_num
        # presf.close()