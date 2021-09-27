import tensorflow as tf
import numpy as np
import math
import os
import sys


class tf_ggx_render:
    '''
    net_load_constants:
        This function should be called 

    parameter lumitexel_size should be sub sampled length
    sub_sample can be used iif config is brdf_not_slice
    '''
    def __init__(self,parameters):
        self.endPoints = {}
        self.parameter_len = parameters["parameter_len"]
        self.fitting_batch_size = parameters["batch_size"]
        # self.slice_width = parameters["slice_width"]
        # self.slice_height = parameters["slice_height"]
        self.lumitexel_size = parameters["lumitexel_size"]
        self.if_grey_scale = parameters["is_grey_scale"]
        self.config_dir = parameters["config_dir"]
        if "slice" in self.config_dir:
            self.sub_sample_rate = 1#parameters["sub_sample_rate"]
            self.slice_sample_num = parameters["slice_sample_num"]

        self.load_constants_from_bin(self.config_dir)
        
    def load_constants_from_bin(self,config_file_dir):
        #load configs
        # self.cam_pos = np.fromfile(config_file_dir+"cam_pos.bin",np.float32)
        self.cam_pos = np.array([0,0,356.20557],np.float32)
        assert self.cam_pos.shape[0] == 3
        print("[RENDERER]cam_pos:",self.cam_pos)
        self.mat_model = np.fromfile(config_file_dir+"mat_model.bin",np.float32).reshape([4,4])
        self.mat_for_normal = np.fromfile(config_file_dir+"mat_for_normal.bin",np.float32).reshape([4,4])

        tmp_data = np.fromfile(config_file_dir+"lights.bin",np.float32).reshape([2,-1,3])
        if "slice" in self.config_dir:
            tmp_data = np.reshape(tmp_data,[2,6,self.slice_sample_num,self.slice_sample_num,3])
            tmp_data = tmp_data[:,:,::self.sub_sample_rate,::self.sub_sample_rate,:]
            tmp_data = np.reshape(tmp_data,[2,-1,3])
        
        assert tmp_data.shape[1] == self.lumitexel_size
        self.light_poses = tmp_data[0]
        self.light_normals = tmp_data[1]
        
        # with open(self.config_dir+"visualize_config_torch.bin","rb") as pf:
            # self.img_size = np.array([16,32]) # np.array()np.fromfile(pf,np.int32,2)
            # self.visualize_map = np.fromfile(pf,np.int32).reshape([-1,2])
            # self.visualize_map = self.visualize_map[:,[1,0]]
            # self.full_face_size = 64

            # self.img_size = np.fromfile(pf,np.int32,2)
            # self.visualize_map = np.fromfile(pf,np.int32).reshape([-1,2])
            # self.full_face_size = 64

        # mask = -np.ones([192,256],np.int32)
        # sub_plane = np.arange(0,self.slice_height*self.slice_width).reshape(self.slice_height,self.slice_width)
        # sub_plane = sub_plane.T
        # mask[64:64+self.slice_width,64*2:64*2+self.slice_height] = sub_plane

        # self.light_poses,self.light_normals = self.get_sub_light_pos(mask)
        
        # with open(self.config_dir+"visualize_config_torch_plane_{}x{}.bin".format(self.slice_height,self.slice_width),"rb") as pf:
        #      # np.array()np.fromfile(pf,np.int32,2)
        #     self.visualize_map = np.fromfile(pf,np.int32).reshape([-1,2])
        #     self.img_size = np.array([self.slice_height,self.slice_width])
        #     self.visualize_map = self.visualize_map[:,[1,0]]
        #     self.full_face_size = 64
        
    def get_sub_light_pos(self,mask):
        '''
        mask (192,256) ndarray
        return:
            sub_light_pos (wanted_lightnum,3) ndarray
            sub_light_normals (wanted_lightnum,3) ndarray
        '''
        idx = np.where(mask >= 0)
        sub_light_num = idx[0].shape[0]
        sub_light_pos = np.zeros((sub_light_num,3),np.float32)
        sub_light_normals = np.zeros((sub_light_num,3),np.float32)
        for which_light in range(sub_light_num):
            idx_lightstage = np.where((self.visualize_map == [idx[1][which_light],idx[0][which_light]]).all(axis=1))[0][0]
            sub_light_pos[mask[idx[0][which_light],idx[1][which_light]]] = self.light_poses[idx_lightstage]
            sub_light_normals[mask[idx[0][which_light],idx[1][which_light]]] = self.light_normals[idx_lightstage]
        return sub_light_pos,sub_light_normals

    def __net_load_constants(self,variable_scope_name):
        ###define some constants
        view_mat_for_normal_init = tf.constant_initializer(self.mat_for_normal)
        view_mat_for_normal = tf.get_variable(name="view_mat_for_normal",dtype=tf.float32,shape = self.mat_for_normal.shape,trainable=False,initializer=view_mat_for_normal_init)
        view_mat_for_normal_t = tf.matrix_transpose(view_mat_for_normal)
        view_mat_model_init = tf.constant_initializer(self.mat_model)
        view_mat_model = tf.get_variable(name="view_mat_model",dtype=tf.float32,shape = self.mat_model.shape,trainable=False,initializer=view_mat_model_init)
        view_mat_model_t = tf.matrix_transpose(view_mat_model)
        cam_pos_init = tf.constant_initializer(self.cam_pos)
        cam_pos = tf.get_variable(name="cam_pos",dtype=tf.float32,shape = self.cam_pos.shape,trainable=False,initializer=cam_pos_init)#shape=[3]
        self.endPoints[variable_scope_name+"cam_pos"] = cam_pos

        light_normals_init = tf.constant_initializer(self.light_normals)
        light_normals = tf.get_variable(name="light_normals",dtype=tf.float32,shape = self.light_normals.shape,trainable=False,initializer=light_normals_init)
        light_poses_init = tf.constant_initializer(self.light_poses)
        light_poses = tf.get_variable(name="light_poses",dtype=tf.float32,shape = self.light_poses.shape,trainable=False,initializer=light_poses_init)
        self.endPoints[variable_scope_name+"light_poses"] = light_poses

        return view_mat_for_normal_t,view_mat_model_t,light_normals,light_poses,cam_pos

    def hemi_octa_map(self,dir):
        '''
        a = [batch,3]
        return = [batch,2]
        '''
        p = dir/tf.reduce_sum(tf.abs(dir),axis=1,keepdims=True)#[batch,3]
        px,py,_ = tf.split(p,[1,1,1],axis=1)
        resultx = px-py
        resulty = px+py
        result = tf.concat([resultx,resulty],axis=-1)
        result = result * 0.5 + 0.5
        return result

    def back_hemi_octa_map(self,a):
        '''
        a = [batch,2]
        return = [batch,3]
        '''
        p = (a - 0.5)*2.0
        px,py = tf.split(p,[1,1],axis=1)#px=[batch,1] py=[batch,1]
        resultx = (px+py)*0.5#[batch,1]
        resulty = (py-px)*0.5#[batch,1]
        resultz = 1.0-tf.abs(resultx)-tf.abs(resulty)
        result = tf.concat([resultx,resulty,resultz],axis=-1)#[batch,3]

        return self.normalized(result)

    def full_octa_map(self,dir):
        '''
        dir = [batch,3]
        return=[batch,2]
        '''
        dirz = tf.expand_dims(tf.gather(dir,2,axis=1),1)#[batch,1]
        p = dir/tf.reduce_sum(tf.abs(dir),axis=1)
        px,py,pz = tf.split(p,[1,1,1],axis=1)
        x = px
        y = py

        judgements1 = tf.greater_equal(px,0.0)
        x_12 = tf.where(judgements1,1.0-py,-1.0+py)#[batch,1]
        y_12 = tf.where(judgements1,1.0-px,1.0+px)

        judgements2 = tf.less_equal(px,0.0)
        x_34 = tf.where(judgements2,-1.0-py,1.0+py)
        y_34 = tf.where(judgements2,-1.0-px,-1.0+px)

        judgements3 = tf.greater_equal(py,0.0)
        x_1234 = tf.where(judgements3,x_12,x_34)#[batch,1]
        y_1234 = tf.where(judgements3,y_12,y_34)#[batch,1]

        judgements4 = tf.less(dirz,0.0)
        x = tf.where(judgements4,x_1234,x)
        y = tf.where(judgements4,y_1234,y)

        return (tf.concat([x,y],axis=1)+1.0)*0.5

    def back_full_octa_map(self,a):
        '''
        a = [batch,2]
        return = [batch,3]
        '''
        p = a*2.0-1.0
        px,py = tf.split(p,[1,1],axis=1)#px=[batch,1] py=[batch,1]
        x = px#px=[batch,1]
        y = py#px=[batch,1]
        abs_px_abs_py = tf.abs(px)+tf.abs(py)
        
        judgements2 = tf.greater_equal(py,0.0)
        judgements3 = tf.greater_equal(px,0.0)
        judgements4 = tf.less_equal(px,0.0)

        x_12 = tf.where(judgements3,1.0-py,-1.0+py)
        y_12 = tf.where(judgements3,1.0-px,1.0+px)

        x_34 = tf.where(judgements4,-1.0-py,1.0+py)
        y_34 = tf.where(judgements4,-1.0-px,-1.0+px)

        x_1234 = tf.where(judgements2,x_12,x_34)
        y_1234 = tf.where(judgements2,y_12,y_34)

        
        judgements1 = tf.greater(abs_px_abs_py,1)

        resultx = tf.where(judgements1,x_1234,px)#[batch,1]
        resulty = tf.where(judgements1,y_1234,py)#[batch,1]
        resultz = 1.0-tf.abs(resultx)-tf.abs(resulty)
        resultz = tf.where(judgements1,-1.0*resultz,resultz)

        result = tf.concat([resultx,resulty,resultz],axis=-1)#[batch,3]

        return self.normalized(result)

    def build_frame_f_z(self,n,theta,with_theta=True):
        '''
        n = [batch,3]
        return =t[batch,3] b[batch,3]
        '''
        nz = tf.expand_dims(tf.gather(n,2,axis=1),axis=-1)#[batch,1]
        constant_001 = tf.constant(np.expand_dims(np.array([0,0,1],np.float32),0).repeat(self.fitting_batch_size,axis=0),dtype=tf.float32,shape=[self.fitting_batch_size,3])#[batch,3]
        constant_100 = tf.constant(np.expand_dims(np.array([1,0,0],np.float32),0).repeat(self.fitting_batch_size,axis=0),dtype=tf.float32,shape=[self.fitting_batch_size,3])#[batch,3]

        nz_notequal_1 = tf.greater(tf.abs(nz-1.0),1e-6)
        nz_notequal_m1 = tf.greater(tf.abs(nz+1.0),1e-6)

        judgements = tf.tile(tf.logical_and(nz_notequal_1,nz_notequal_m1),[1,3])#[batch,3]

        t = tf.where(judgements,constant_001,constant_100)#[batch,3]

        t = self.normalized(tf.cross(t,n))#[batch,3]
        b = tf.cross(n,t)#[batch,3]

        if not with_theta:
            return t,b

        t = self.normalized(t*tf.cos(theta)+b*tf.sin(theta))

        b = self.normalized(tf.cross(n,t))#[batch,3]
        return t,b

    def normalized(self,a):
        assert len(a.shape) == 2
        norm = tf.sqrt(tf.reduce_sum(tf.square(a), axis=1, keepdims=True))
        return a/(norm+1e-6)
    
    def normalized_nd(self,a):
        norm = tf.sqrt(tf.reduce_sum(tf.square(a),axis=-1,keepdims=True))
        return a/(norm+1e-6)

    def norm(self,a):
        return tf.sqrt(tf.reduce_sum(tf.square(a),axis=-1,keepdims=True))

    def dot_ndm_vector(self,nd_tensor,vec):#assume nd_tensor's shape = [bach,lightnum,3] vector is [batch,1,3] or [batch,lightnum,3]
        return tf.reduce_sum(tf.multiply(nd_tensor,vec),axis=-1,keepdims=True)#return is [bach,lightnum,1]

    def compute_form_factors(self,position,n,light_poses_broaded,light_normals_broaded,variable_scope_name,with_cos=True):
        '''
        position = [batch,3]
        n = [batch,3]
        light_poses_broaded = [batch,lightnum,3]
        light_normals_broaded = [batch,lightnum,3]

        with_cos: if this is true, form factor adds cos(ldir.light_normals)  

        return shape=[batch,lightnum,1]
        '''
        ldir = light_poses_broaded-tf.expand_dims(position,axis=1)#[batch,lightnum,3]
        dist = tf.sqrt(tf.reduce_sum(tf.square(ldir),axis=-1,keepdims=True))#[batch,lightnum,1]
        ldir = self.normalized_nd(ldir)

        a = tf.maximum(self.dot_ndm_vector(ldir,tf.expand_dims(n,axis=1)),0.0)#[batch,lightnum,1]
        b = dist*dist#[batch,lightnum,1]
        c = tf.maximum(self.dot_ndm_vector(ldir,light_normals_broaded),0.0)#[batch,lightnum,1]
        r_2_cos = b/(c+1e-6)
        cos_r_2 = c/b
        self.endPoints[variable_scope_name+"r_2_cos"] = r_2_cos
        self.endPoints[variable_scope_name+"cos_r_2"] = cos_r_2
        self.endPoints[variable_scope_name+"cos2"] = a
        if with_cos:
            return a/(b+1e-6)*c
        else:
            return a

    def ggx_G1_aniso_honntai(self,v,vz,ax,ay):
        axayaz = tf.concat([ax,ay,tf.ones([self.fitting_batch_size,1])],axis=-1)#[batch,3]
        vv = v*tf.expand_dims(axayaz,axis=1)#[batch,lightnum,3]
        # return 2.0/(1.0+(self.norm(vv)/(vz+1e-6)))
        return 2*vz/(vz+self.norm(vv)+1e-6)#[batch,lightnum,1]

    def ggx_G1_aniso(self,v,ax,ay,vz):
        '''
        v = [batch,lightnum,3]
        ax = [batch,1]
        ay = [batch,1]
        vz = [batch,lightnum,1] 
        return shape = [batch,lightnum,1]
        '''
        
        # comparison = tf.less_equal(vz,0)
        # hontai = self.ggx_G1_aniso_honntai(v,vz,ax,ay)
        # return tf.where(comparison,tf.zeros([self.fitting_batch_size,self.lumitexel_size,1]),hontai)#[batch,lightnum,1]
        comparison = (tf.sign(vz)+1.0)*0.5
        hontai = self.ggx_G1_aniso_honntai(v,vz,ax,ay)
        return hontai*comparison
        
    def ggx_brdf_aniso(self,wi,wo,ax,ay):
        '''
        wi = [batch,lightnum,3]
        wo = [batch,lightnum,3]
        ax = [batch,1]
        ay = [batch,1]
        return shape = [batch,lightnum,1]
        '''
        wi_z = tf.expand_dims(tf.gather(wi,indices=2,axis=2,name="wi_z"),axis=-1)#shape=[batch,lightnum,1]
        wo_z = tf.expand_dims(tf.gather(wo,indices=2,axis=2,name="wo_z"),axis=-1)#shape=[batch,lightnum,1]
        denom = 4*wi_z*wo_z#shape=[batch,lightnum,1]
        vhalf = self.normalized_nd(wi+wo)#[batch,lightnum,3]
        tmp = tf.minimum(tf.maximum(0.0,1-self.dot_ndm_vector(wi,vhalf)),1.0)#[batch,lightnum,1]
        F0 = 0.04
        F = F0+(1-F0)* tmp * tmp * tmp * tmp * tmp#[batch,lightnum,1]
        axayaz = tf.expand_dims(tf.concat([ax,ay,tf.ones([self.fitting_batch_size,1],tf.float32)],axis=-1),axis=1)#[batch,1,3]
        vhalf = vhalf/(axayaz+1e-6)
        vhalf_norm = self.norm(vhalf)#[batch,lightnum,1]
        length = vhalf_norm*vhalf_norm##[batch,lightnum,1]
        D = 1.0/(math.pi*tf.expand_dims(ax,axis=1)*tf.expand_dims(ay,axis=1)*length*length)#[batch,lightnum,1]

        judgement_wiz_less_equal_0 = tf.less_equal(wi_z,0.0)
        judgement_woz_less_equal_0 = tf.less_equal(wo_z,0.0)

        tmp_ones = tf.ones([self.fitting_batch_size,self.lumitexel_size,1])
        safe_denom = tf.where(judgement_wiz_less_equal_0,tmp_ones,denom)
        safe_denom = tf.where(judgement_woz_less_equal_0,tmp_ones,safe_denom)

        tmp = D*F*self.ggx_G1_aniso(wi,ax,ay,wi_z)*self.ggx_G1_aniso(wo,ax,ay,wo_z)/(safe_denom+1e-6)#[batch,lightnum,1]

        
        tmp_zeros = tf.zeros([self.fitting_batch_size,self.lumitexel_size,1])


        res = tf.where(judgement_wiz_less_equal_0,tmp_zeros,tmp)
        res = tf.where(judgement_woz_less_equal_0,tmp_zeros,res)

        wi_z_sign = (tf.sign(wi_z)+1.0)*0.5#shape=[batch,lightnum,1]
        wo_z_sign = (tf.sign(wo_z)+1.0)*0.5#shape=[batch,lightnum,1]
        
        # res = tmp*wi_z_sign*wo_z_sign
        self.endPoints["4"] = wi_z_sign
        self.endPoints["5"] = wo_z_sign
        self.endPoints["6"] = denom+1e-6
        return res

        
    def regularizer_relu(self,a,_bound1,_bound2,k=1e5):
        # return tf.reduce_mean(tf.tanh(800*(a-_bound2))+tf.tanh(800*(_bound1-a))+2)*k
        return tf.reduce_mean(tf.reshape(tf.nn.relu((a*-1.0+_bound1)*k)+tf.nn.relu((a-_bound2)*k),[-1]))


    def calc_light_brdf(self,wi_local,wo_local,ax,ay,pd,ps,pd_ps_wanted):
        '''
        wi_local = [batch,lightnum,3]
        wo_local = [batch,lightnum,3]
        ax = [batch,1]
        ay = [batch,1]
        pd = [batch,3]
        ps = [batch,3]
        return shape=[batch,lightnum,3]
        '''
        tmp = self.ggx_brdf_aniso(wi_local,wo_local,ax,ay)
        self.endPoints["1"]=tmp
        b = tf.tile(tmp,[1,1,3])#[batch,lightnum,1]
        ps = tf.tile(tf.expand_dims(ps,axis=1),[1,self.lumitexel_size,1])#[batch,lightnum,3]
        self.endPoints["2"] = ps
        a = tf.tile(tf.expand_dims(pd/math.pi,axis=1),[1,self.lumitexel_size,1])#[batch,lightnum,3]
        self.endPoints["3"] = a
        if pd_ps_wanted == "both":
            return a+b*ps
        elif pd_ps_wanted =="pd_only":
            return a
        elif pd_ps_wanted == "ps_only":
            return b*ps
        # return b*ps# return a+b*ps

    def calculate_n_dot_view(self,variable_scope_name):
        return self.endPoints[variable_scope_name+"n_dot_view_dir"],self.endPoints[variable_scope_name+"n_dot_view_penalty"]

    def param_initializer(self,input_lumitexels,variable_scope_name):
        '''
        input_lumitexels = [batch,lightnum]
        used
        '''
        max_poses = tf.argmax(input_lumitexels,axis=1)
        light_poses = tf.gather(self.endPoints[variable_scope_name+"light_poses"],max_poses,axis=0)#[batch,3]
        max_wi = self.normalized(light_poses-self.endPoints[variable_scope_name+"positions"])#[batch,3]
        view_dir = self.endPoints[variable_scope_name+"view_dir"]#shape=[batch,3]
        n_global = (max_wi+view_dir)*0.5
        n_local = tf.concat([self.dot_ndm_vector(n_global,self.endPoints[variable_scope_name+"frame_t"]),
                            self.dot_ndm_vector(n_global,self.endPoints[variable_scope_name+"frame_b"]),
                            self.dot_ndm_vector(n_global,self.endPoints[variable_scope_name+"frame_n"])],axis=1)#[batch,3]
        n = self.hemi_octa_map(n_local)#[batch,2]
        theta = tf.random_uniform([self.fitting_batch_size,1],minval=0,maxval=math.pi, seed=1)#[batch,1]
        axay_min = 0.006
        axay_max = 0.503
        axay_init = np.sqrt(axay_min*axay_max)
        axay = tf.constant(np.expand_dims(np.array([axay_init,axay_init],np.float32),0).repeat(self.fitting_batch_size,axis=0))#[batch,2]
        #axay = tf.exp(tf.random_uniform([self.fitting_batch_size,2],minval=tf.log(0.006),maxval=tf.log(0.503), seed=1))#[batch,1]
        pd = tf.random_uniform([self.fitting_batch_size,1],minval=1e-9,maxval=1.0-1e-9, seed=1)#[batch,1]
        ps = tf.random_uniform([self.fitting_batch_size,1],minval=1e-9,maxval=10.0-1e-9, seed=1)#[batch,1]
        pd_max = 1
        ps_max = 10
        all_params_pd0 = tf.concat([n,theta,axay,tf.zeros(shape=[self.fitting_batch_size,1]),tf.ones(shape=[self.fitting_batch_size,1])*ps_max],axis=-1)
        all_params_ps0 = tf.concat([n,theta,axay,tf.ones(shape=[self.fitting_batch_size,1])*pd_max,tf.zeros(shape=[self.fitting_batch_size,1])],axis=-1)
        self.endPoints[variable_scope_name+"all_params_ps0"] = all_params_ps0
        self.endPoints[variable_scope_name+"all_params_pd0"] = all_params_pd0
        return all_params_ps0,all_params_pd0

    def rebuild_local_frame(self,sess):
        for i in range(1000):
            print("[CAUTION] This function is deprecated.\n It will be removed after 3.30.2019.\nyou should delete it!")
        sess.run([self.endPoints["frame_t"],self.endPoints["frame_b"],self.endPoints["frame_n"]])
    
    def get_r_2_cos(self,sess,variable_scope_name):
        '''
        return shape=[batch,lightnum,1]
        '''
        return sess.run(self.endPoints[variable_scope_name+"r_2_cos"])

    def get_r_2_cos_node(self,variable_scope_name):
        '''
        return shape=[batch,lightnum,1]
        '''
        return self.endPoints[variable_scope_name+"r_2_cos"]

    def get_cos_r_2_node(self,variable_scope_name):
        '''
        return shape=[batch,lightnum,1]
        '''
        return self.endPoints[variable_scope_name+"cos_r_2"]

    def get_cos_2_node(self,variable_scope_name):
        '''
        return shape=[batch,lightnum,1]
        '''
        return self.endPoints[variable_scope_name+"cos2"]

    def rotation_axis(self,t,v,isRightHand=True):
        '''
        t = [batch,1]#rotate rad??
        v = [batch,3]#rotate axis(global) 
        return = [batch,4,4]#rotate matrix
        '''
        if isRightHand:
            theta = t
        else:
            print("[RENDERER]Error rotate system doesn't support left hand logic!")
            exit()
        
        c = tf.cos(theta)
        s = tf.sin(theta)

        v_x,v_y,v_z = tf.split(v,[1,1,1],axis=-1)

        m_11 = c + (1-c)*v_x*v_x
        m_12 = (1 - c)*v_x*v_y - s*v_z
        m_13 = (1 - c)*v_x*v_z + s*v_y

        m_21 = (1 - c)*v_x*v_y + s*v_z
        m_22 = c + (1-c)*v_y*v_y
        m_23 = (1 - c)*v_y*v_z - s*v_x

        m_31 = (1 - c)*v_z*v_x - s*v_y
        m_32 = (1 - c)*v_z*v_y + s*v_x
        m_33 = c + (1-c)*v_z*v_z

        tmp_zeros = tf.zeros([self.fitting_batch_size,1])
        tmp_ones = tf.ones([self.fitting_batch_size,1])

        res = tf.concat([
            m_11,m_12,m_13,tmp_zeros,
            m_21,m_22,m_23,tmp_zeros,
            m_31,m_32,m_33,tmp_zeros,
            tmp_zeros,tmp_zeros,tmp_zeros,tmp_ones
        ],axis=-1)

        res = tf.reshape(res,[self.fitting_batch_size,4,4])
        return res



    def debug_get_view_mats(self,variable_scope_name):
        return self.endPoints[variable_scope_name+"view_mat_model"],\
            self.endPoints[variable_scope_name+"view_mat_for_normal"],\
                self.endPoints[variable_scope_name+"position_origin"],\
                    self.endPoints[variable_scope_name+"position_rotated"],\
                        self.endPoints[variable_scope_name+"n"],\
                            self.endPoints[variable_scope_name+"t"],\
                                self.endPoints[variable_scope_name+"b"],\
                                    self.endPoints[variable_scope_name+"normal_local"]

    def draw_rendering_net(self,input_params,position,rotate_theta,variable_scope_name,with_cos = True,pd_ps_wanted="both"):
        '''
        input_params = (rendering parameters) shape = [self.fitting_batch_size,self.parameter_len] i.e.[24576,10]
        position = (rendering positions) shape=[self.fitting_batch_size,3]
        variable_scope_name = (for variable check a string like"rendering1") 
        rotate_theta = [self.fitting_batch_size,1]
        return shape = (rendered results)[batch,lightnum,1] or [batch,lightnum,3]


        with_cos: if True,lumitexl is computed with cos and dir
        '''
        with tf.variable_scope(variable_scope_name):
            ###[STEP 0]
            #load constants
            view_mat_for_normal_t,view_mat_model_t,light_normals,light_poses,cam_pos = self.__net_load_constants(variable_scope_name)
            #rotate object           
            rotate_axis = tf.constant(np.repeat(np.array([0,0,1],np.float32).reshape([1,3]),self.fitting_batch_size,axis=0))
            view_mat_model = self.rotation_axis(rotate_theta,rotate_axis)#[batch,4,4]
            self.endPoints[variable_scope_name+"view_mat_model"] = view_mat_model
            view_mat_model_t = tf.matrix_transpose(view_mat_model)

            view_mat_for_normal =tf.matrix_transpose(tf.matrix_inverse(view_mat_model))
            self.endPoints[variable_scope_name+"view_mat_for_normal"] = view_mat_for_normal
            view_mat_for_normal_t = tf.matrix_transpose(view_mat_for_normal)
            # self.endPoints[variable_scope_name+"view_mat_for_normal_t"] = view_mat_for_normal
            ###[STEP 1]
            ##define input
            with tf.variable_scope("fittinger"):
                self.endPoints[variable_scope_name+"input_parameters"] = input_params
                self.endPoints[variable_scope_name+"positions"] = position
                view_dir = cam_pos - position #shape=[batch,3]
                view_dir = self.normalized(view_dir)#shape=[batch,3]
                self.endPoints[variable_scope_name+"view_dir"] = view_dir

                #build local frame
                frame_t,frame_b = self.build_frame_f_z(view_dir,None,with_theta=False)#[batch,3]
                frame_n = view_dir
                self.endPoints[variable_scope_name+"frame_t"] = frame_t
                self.endPoints[variable_scope_name+"frame_b"] = frame_b
                self.endPoints[variable_scope_name+"frame_n"] = frame_n

            ###[STEP 1.1]
            ###split input parameters into position and others
            if self.if_grey_scale:
                n_2d,theta,ax,ay,pd,ps = tf.split(input_params,[2,1,1,1,1,1],axis=1)
                self.endPoints[variable_scope_name+"pd"] = pd
                pd = tf.tile(pd,[1,3])
                ps = tf.tile(ps,[1,3])
            else:
                n_local,t_local,ax,ay,pd,ps = tf.split(input_params,[2,1,1,1,3,3],axis=1)

            #position shape=[bach,3]
            # n_2d = tf.clip_by_value(n_2d,0.0,1.0)
            n_local = self.back_hemi_octa_map(n_2d)#[batch,3]
            self.endPoints[variable_scope_name+"normal_local"] = n_local
            t_local,_ = self.build_frame_f_z(n_local,theta,with_theta=True)
            n_local_x,n_local_y,n_local_z = tf.split(n_local,[1,1,1],axis=1)#[batch,1],[batch,1],[batch,1]
            n = n_local_x*frame_t+n_local_y*frame_b+n_local_z*frame_n#[batch,3]
            self.endPoints[variable_scope_name+"normal"]  = n
            t_local_x,t_local_y,t_local_z = tf.split(t_local,[1,1,1],axis=1)#[batch,1],[batch,1],[batch,1]
            t = t_local_x*frame_t+t_local_y*frame_b+t_local_z*frame_n#[batch,3]
            b = tf.cross(n,t)#[batch,3]
            self.endPoints[variable_scope_name+"tangent"] = t
            self.endPoints[variable_scope_name+"binormal"] = b
            #rotate frame
            pn = tf.expand_dims(tf.concat([n,tf.ones([n.shape[0],1],tf.float32)],axis=1),axis=1)
            pt = tf.expand_dims(tf.concat([t,tf.ones([t.shape[0],1],tf.float32)],axis=1),axis=1)
            pb = tf.expand_dims(tf.concat([b,tf.ones([b.shape[0],1],tf.float32)],axis=1),axis=1)

            n = tf.squeeze(tf.matmul(pn,view_mat_for_normal_t),axis=1)
            t = tf.squeeze(tf.matmul(pt,view_mat_for_normal_t),axis=1)
            b = tf.squeeze(tf.matmul(pb,view_mat_for_normal_t),axis=1)
            n,_ = tf.split(n,[3,1],axis=1)#shape=[batch,3]          
            t,_ = tf.split(t,[3,1],axis=1)#shape=[batch,3]
            b,_ = tf.split(b,[3,1],axis=1)#shape=[batch,3]
            self.endPoints[variable_scope_name+"n"] = n
            self.endPoints[variable_scope_name+"t"] = t
            self.endPoints[variable_scope_name+"b"] = b
            # n = tf.tile(tf.constant([0,0,1],dtype=tf.float32,shape=[1,3]),[self.fitting_batch_size,1])#self.normalized(n)
            # t = tf.tile(tf.constant([0,1,0],dtype=tf.float32,shape=[1,3]),[self.fitting_batch_size,1])#self.normalized(t)
            # ax = tf.clip_by_value(ax,0.006,0.503)
            # ay = tf.clip_by_value(ay,0.006,0.503)
            # pd = tf.clip_by_value(pd,0,1)
            # ps = tf.clip_by_value(ps,0,10)

            #regularizer of params
            # regular_loss_n = self.regularizer_relu(n_2d,1e-3,1.0)+self.regularizer_relu(theta,0.0,math.pi)+self.regularizer_relu(ax,0.006,0.503)+self.regularizer_relu(ay,0.006,0.503)+self.regularizer_relu(pd,1e-3,1)+self.regularizer_relu(ps,1e-3,10)
            # self.endPoints["regular_loss"] = regular_loss_n


            self.endPoints[variable_scope_name+"render_params"] = tf.concat([n,t,b,ax,ay,pd,ps],axis=-1)
            ###[STEP 2]
            ##define rendering
            with tf.variable_scope("rendering"):
                self.endPoints[variable_scope_name+"position_origin"] = position
                position = tf.expand_dims(tf.concat([position,tf.ones([position.shape[0],1],tf.float32)],axis=1),axis=1)
                position = tf.squeeze(tf.matmul(position,view_mat_model_t),axis=1)#position@view_mat_model_t
                position,_ = tf.split(position,[3,1],axis=1)#shape=[batch,3]
                self.endPoints[variable_scope_name+"position_rotated"] = position

                #get real view dir
                view_dir = cam_pos - position #shape=[batch,3]
                view_dir = self.normalized(view_dir)#shape=[batch,3]
                self.endPoints[variable_scope_name+"view_dir_rotated"] = view_dir



                light_poses_broaded = tf.tile(tf.expand_dims(light_poses,axis=0),[self.fitting_batch_size,1,1],name="expand_light_poses")#shape is [batch,lightnum,3]
                light_normals_broaded = tf.tile(tf.expand_dims(light_normals,axis=0),[self.fitting_batch_size,1,1],name="expand_light_normals")#shape is [batch,lightnum,3]
                position_broded = tf.tile(tf.expand_dims(position,axis=1),[1,self.lumitexel_size,1],name="expand_position")
                wi = light_poses_broaded-position_broded
                wi = self.normalized_nd(wi)#shape is [batch,lightnum,3]
                self.endPoints[variable_scope_name+"wi"] = wi


                wi_local = tf.concat([self.dot_ndm_vector(wi,tf.expand_dims(t,axis=1)),
                                      self.dot_ndm_vector(wi,tf.expand_dims(b,axis=1)),
                                      self.dot_ndm_vector(wi,tf.expand_dims(n,axis=1))],axis=-1)#shape is [batch,lightnum,3]
                
                view_dir_broaded = tf.tile(tf.expand_dims(view_dir,axis=1),[1,self.lumitexel_size,1])#shape is [batch,lightnum,3]
                wo_local = tf.concat([self.dot_ndm_vector(view_dir_broaded,tf.expand_dims(t,axis=1)),
                                      self.dot_ndm_vector(view_dir_broaded,tf.expand_dims(b,axis=1)),
                                      self.dot_ndm_vector(view_dir_broaded,tf.expand_dims(n,axis=1))],axis=-1)#shape is [batch,lightnum,3]

                self.endPoints[variable_scope_name+"wi_local"] = wo_local
                form_factors = self.compute_form_factors(position,n,light_poses_broaded,light_normals_broaded,variable_scope_name,with_cos)#[batch,lightnum,1]
                self.endPoints[variable_scope_name+"form_factors"] = form_factors
                lumi = self.calc_light_brdf(wi_local,wo_local,ax,ay,pd,ps,pd_ps_wanted)#[batch,lightnum,3]
                self.endPoints[variable_scope_name+"lumi_without_formfactor"] = lumi
                
                lumi = lumi*form_factors*1e4*math.pi*1e-2#[batch,lightnum,3]
                if self.if_grey_scale:
                    lumi = tf.reduce_mean(lumi,axis=2,keepdims=True)#[batch,lightnum,1] for greyscale#TODO should be depended by if is grey

                wi_dot_n = self.dot_ndm_vector(wi,tf.expand_dims(n,axis=1))#[batch,lightnum,1]
                lumi = lumi*((tf.sign(wi_dot_n)+1.0)*0.5)
                # judgements = tf.less(wi_dot_n,1e-5)
                # lumi = tf.where(judgements,tf.zeros([self.fitting_batch_size,self.lumitexel_size,1]),lumi)

                n_dot_view_dir = self.dot_ndm_vector(view_dir_broaded,tf.tile(tf.expand_dims(n,axis=1),[1,self.lumitexel_size,1]))#[batch,lightnum,1]
                
                n_dot_views = tf.gather(n_dot_view_dir,0,axis=1)
                self.endPoints[variable_scope_name+"n_dot_view_dir"] = n_dot_views
                n_dot_view_penalty = self.regularizer_relu(n_dot_views,1e-6,1.0)
                self.endPoints[variable_scope_name+"n_dot_view_penalty"] = n_dot_view_penalty

                judgements = tf.less(n_dot_view_dir,0.0)
                if self.if_grey_scale:
                    rendered_results = tf.where(judgements,tf.zeros([self.fitting_batch_size,self.lumitexel_size,1]),lumi)#[batch,lightnum]
                else:
                    rendered_results = tf.where(tf.tile(judgements,[1,1,3]),tf.zeros([self.fitting_batch_size,self.lumitexel_size,3]),lumi)#[batch,lightnum]
                self.endPoints[variable_scope_name+"rendered_results"] = rendered_results

        return rendered_results
    
    def get_global_normal(self,variable_scope_name):
        return self.endPoints[variable_scope_name+"normal"]

    def get_n_dot_view_penalty(self,variable_scope_name):
        return self.endPoints[variable_scope_name+"n_dot_view_penalty"],self.endPoints[variable_scope_name+"n_dot_view_dir"]
    
    def get_n(self,variable_scope_name):
        return self.endPoints[variable_scope_name+"normal_local"],self.endPoints[variable_scope_name+"normal"]
    
    def get_pd(self,variable_scope_name):
        return self.endPoints[variable_scope_name+"pd"]
    
    def get_wi(self,variable_scope_name):
        return self.endPoints[variable_scope_name+"wi"]
    
    def get_compute_node_by_name(self,variable_scope_name,node_name):
        return self.endPoints[variable_scope_name+node_name]

if __name__ == "__main__":
    thread_id = int(sys.argv[1])
    data_root = sys.argv[2]
    shrink_size = 2
    parameters = {}
    parameters["batch_size"] = 10
    parameters["lumitexel_size"] = 24576//shrink_size//shrink_size
    parameters["is_grey_scale"] = True
    parameters["config_dir"] = './tf_ggx_render_configs_{}x{}/'.format(shrink_size,shrink_size)#指向随包的config文件夹

    if parameters["is_grey_scale"]:  
        parameters["parameter_len"] = 7
    else:  
        parameters["parameter_len"] = 11
    
    renderer = tf_ggx_render(parameters)#实例化渲染器

    test_params = np.fromfile(data_root+"test_param{}.bin".format(thread_id),np.float32).reshape([-1,parameters["parameter_len"]])[:100]

    with tf.Session() as sess:
        #define inputs
        input_params = tf.placeholder(tf.float32,shape=[parameters["batch_size"],parameters["parameter_len"]],name="render_params")
        input_positions = tf.placeholder(tf.float32,shape=[parameters["batch_size"],3],name="render_positions")

        #draw rendering net
        rendered_res = renderer.draw_rendering_net(input_params,input_positions,"my_little_render")
        
        #global variable init should be called before rendering
        init = tf.global_variables_initializer()
        sess.run(init)

        presf = open(data_root+"test_rendered{}_shrink{}.bin".format(thread_id,shrink_size),"wb")
        data_ptr = 0
        while True:
            tmp_params = test_params[data_ptr:data_ptr+parameters["batch_size"]]
            if tmp_params.shape[0] == 0:
                break
            result = sess.run(rendered_res,feed_dict={
                input_params: tmp_params,
                input_positions :np.zeros([parameters["batch_size"],3],np.float32)
            })
            result.tofile(presf)
            data_ptr+=tmp_params.shape[0]
        presf.close()
