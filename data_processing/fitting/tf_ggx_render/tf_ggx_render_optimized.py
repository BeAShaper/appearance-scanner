import tensorflow as tf
import numpy as np
import math
import time
import os
import sys

default_axis_size = 9//2
default_axis_bold = 3//2

class tf_ggx_render:
    '''
    net_load_constants:
        This function should be called 
    '''
    def __init__(self,parameters):
        print("[RENDERER]----------------")
        self.endPoints = {}
        self.fitting_batch_size = parameters["batch_size"]
        self.lumitexel_size = parameters["lumitexel_size"]#the length of ultimate lumitexel
        self.shrink_step = parameters["shrink_step"]
        self.is_real_light = parameters["is_real_light"]
        self.if_grey_scale = parameters["is_grey_scale"]
        self.config_dir = parameters["config_dir"]
        self.load_constants_from_bin(self.config_dir)
        
    def load_constants_from_bin(self,config_file_dir):
        #load configs
        self.cam_pos = np.fromfile(config_file_dir+"cam_pos.bin",np.float32)
        assert self.cam_pos.shape[0] == 3
        print("[RENDERER]cam_pos:",self.cam_pos)
        self.mat_model = np.fromfile(config_file_dir+"mat_model.bin",np.float32).reshape([4,4])
        self.mat_for_normal = np.fromfile(config_file_dir+"mat_for_normal.bin",np.float32).reshape([4,4])

        tmp_data = np.fromfile(config_file_dir+"lights.bin",np.float32).reshape([2,-1,3])
        tmp_data = self.shrink_lights(tmp_data)
        assert tmp_data.shape[1] == self.lumitexel_size
        self.light_poses = tmp_data[0]
        self.light_normals = tmp_data[1]
    
    def get_tf_cam_pos(self):
        cam_pos_init = tf.constant_initializer(self.cam_pos)
        cam_pos = tf.get_variable(name="cam_pos",dtype=tf.float32,shape = self.cam_pos.shape,trainable=False,initializer=cam_pos_init)#shape=[3]
        return cam_pos
    
    def get_tf_light_poses(self):
        light_poses_init = tf.constant_initializer(self.light_poses)
        light_poses = tf.get_variable(name="light_poses",dtype=tf.float32,shape = self.light_poses.shape,trainable=False,initializer=light_poses_init)
        return light_poses

    def shrink_lights(self,tmp_data):
        '''
        tmp_data.shape=[2,24576,3] if real lights
        '''
        if not self.is_real_light:
            print("[RENDERER] You are using a fake light config. You should choose correct light config!")
            time.sleep(0.3)
            
            self.full_face_edge_light_num = 64
            self.shrinked_face_edge_light_num = self.full_face_edge_light_num // self.shrink_step
            self.full_light_num = self.full_face_edge_light_num*self.full_face_edge_light_num*6
            self.full_img_height = self.full_face_edge_light_num * 3
            self.full_img_width = self.full_face_edge_light_num * 4
            self.shrinked_img_height = self.shrinked_face_edge_light_num * 3
            self.shrinked_img_width = self.shrinked_face_edge_light_num * 4

            self.visualize_idxs = np.fromfile(self.config_dir+"visualize_idxs.bin",dtype = np.int32).reshape([-1,2])
            return tmp_data 
        else:
            print("[RENDERER] You are using a real light config and SHRINK STEP is{}.".format(self.shrink_step))
            time.sleep(0.3)

            self.full_face_edge_light_num = 64
            self.shrinked_face_edge_light_num = self.full_face_edge_light_num // self.shrink_step
            self.full_light_num = self.full_face_edge_light_num*self.full_face_edge_light_num*6
            self.full_img_height = self.full_face_edge_light_num * 3
            self.full_img_width = self.full_face_edge_light_num * 4
            self.shrinked_img_height = self.shrinked_face_edge_light_num * 3
            self.shrinked_img_width = self.shrinked_face_edge_light_num * 4
            assert tmp_data.shape[1] == self.full_light_num

            self.visualize_idxs_full = np.fromfile(self.config_dir+"visualize_idxs.bin",dtype = np.int32).reshape([-1,2])
            if self.visualize_idxs_full.shape[0] != self.full_light_num :
                print("[RENDERER]:error visualize dimension")
                exit()
            
            #visualize full
            img = np.zeros([self.full_img_height,self.full_img_width],np.bool)
            img[self.visualize_idxs_full[:,1],self.visualize_idxs_full[:,0]] = True

            #shrink 
            mask = np.zeros([self.full_img_height,self.full_img_width],np.bool)
            mask[0:self.full_face_edge_light_num,self.full_face_edge_light_num*1:self.full_face_edge_light_num*2] = True 
            mask[self.full_face_edge_light_num*1:self.full_face_edge_light_num*2] = True 
            mask[self.full_face_edge_light_num*2:self.full_face_edge_light_num*3,self.full_face_edge_light_num*1:self.full_face_edge_light_num*2] = True 

            img = np.zeros([self.full_img_height,self.full_img_width],np.float32)

            img_valid_point = img.copy()
            img_valid_point = img_valid_point.reshape([self.full_img_height//self.shrink_step,self.shrink_step,self.full_img_width//self.shrink_step,self.shrink_step])
            img_valid_point[:,0,:,0] = 1.0
            img_valid_point = img_valid_point.reshape([self.full_img_height,self.full_img_width])

            np.copyto(img,img_valid_point,where=mask)

            valid_pixel_idx = np.nonzero(img)

            #compute maped img
            light_id_array = np.array(range(self.full_light_num))
            img_map = np.zeros([self.full_img_height,self.full_img_width],np.int32)
            img_map[self.visualize_idxs_full[:,1],self.visualize_idxs_full[:,0]] = light_id_array

            #get valid light number
            valid_light_number = img_map[valid_pixel_idx[0],valid_pixel_idx[1]]
            # print(self.visualize_idxs_full.shape)
            self.visualize_idxs = self.visualize_idxs_full[valid_light_number]
            # print(self.visualize_idxs.shape)
            self.visualize_idxs = self.visualize_idxs//self.shrink_step
            # print(self.visualize_idxs.shape)
            # exit()
            tmp_data_selected = tmp_data[:,valid_light_number,:]

        return tmp_data_selected

    def visualize_lumi_vector(self,lumi_vector,with_expansion = True,scalerf=1.0,background=0.0,edge = False):
        assert lumi_vector.shape[0] == self.lumitexel_size
        if len(lumi_vector.shape) == 1:
            lumi_vector = np.expand_dims(lumi_vector,axis=-1)
        channel_num = lumi_vector.shape[1]
        assert (channel_num == 1 or channel_num == 3),"channel of lumitexel isn't correct get:{} wanted: 1 or 3".format(channel_num)

        img_shrinked = np.ones([self.shrinked_img_height,self.shrinked_img_width,channel_num],np.float)*background
        if edge:
            img_shrinked[0,:,:] = 255.0
            img_shrinked[:,0,:] = 255.0
        img_shrinked[self.visualize_idxs[:,1],self.visualize_idxs[:,0]] = lumi_vector*scalerf
        if not with_expansion:
            return img_shrinked
        else:
            img_shrinked = img_shrinked.reshape([self.shrinked_img_height,1,self.shrinked_img_width,1,channel_num])
            img_full = np.repeat(img_shrinked,self.shrink_step,axis=1)
            img_full = np.repeat(img_full,self.shrink_step,axis=3)
            img_full = np.reshape(img_full,[self.full_img_height,self.full_img_width,channel_num])
            return img_full
        
    def draw_vector_in_lumi_img(self,lumi_img,the_vector,obj_pos,color,axis_size = default_axis_size,axis_bold = default_axis_bold):
        '''
        the_vector = (3,)
        obj_pos = (3,)
        color = (3,)
        img = [full_img_height,full_img_width,3]
        '''
        # if self.is_real_light == False:
        #     print("[RENDER] currently the fakelights doesn't support draw vector fun")
        #     exit()
        #step 1 get max idx
        light_dir = self.light_poses - obj_pos.reshape([1,3])
        light_dir = light_dir/np.linalg.norm(light_dir,axis=1,keepdims=True)#[light_num,3]

        dot_res = np.sum(light_dir*the_vector,axis=1)
        max_idx = np.argmax(dot_res)
        
        #step 2 get max uv
        max_uv = self.visualize_idxs[max_idx] * self.shrink_step
        
        #step 3 draw point
        lumi_img[max(0,max_uv[1]-axis_size):max_uv[1]+axis_size+1,max(0,max_uv[0]-axis_bold):max_uv[0]+axis_bold+1,:] = color
        lumi_img[max(0,max_uv[1]-axis_bold):max_uv[1]+axis_bold+1,max(0,max_uv[0]-axis_size):max_uv[0]+axis_size+1,:] = color



    def reset_cam_pos(self,new_cam_pos):
        '''
        new_cam_pos = (3,)
        '''
        assert new_cam_pos.shape == (3,)
        self.cam_pos = new_cam_pos
        print("[RENDERER]cam_pos reset to:",self.cam_pos)

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
        
    def ggx_brdf_aniso(self,wi,wo,ax,ay,specular_component):
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

        tmp = 1.0
        if "D" in specular_component:
            tmp = tmp * D
        if "F" in specular_component:
            tmp = tmp * F
        if "G" in specular_component:
            tmp = tmp * self.ggx_G1_aniso(wi,ax,ay,wi_z)*self.ggx_G1_aniso(wo,ax,ay,wo_z)
        if "B" in specular_component:
            tmp = tmp / (safe_denom+1e-6) 
        
        #[batch,lightnum,1]

        
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


    def calc_light_brdf(self,wi_local,wo_local,ax,ay,pd,ps,pd_ps_wanted,specular_component):
        '''
        wi_local = [batch,lightnum,3]
        wo_local = [batch,lightnum,3]
        ax = [batch,1]
        ay = [batch,1]
        pd = [batch,channel]
        ps = [batch,channel]
        return shape=[batch,lightnum,channel]
        '''
        if pd_ps_wanted == "both":
            tmp = self.ggx_brdf_aniso(wi_local,wo_local,ax,ay,specular_component)#[batch,lightnum,1]
            b = tmp#[batch,lightnum,1]
            ps = tf.expand_dims(ps,axis=1)#[batch,1,channel]
            a = tf.expand_dims(pd/math.pi,axis=1)#[batch,1,1]
            return a+b*ps
        elif pd_ps_wanted =="pd_only":
            a = tf.expand_dims(pd/math.pi,axis=1)#[batch,1,channel]
            return tf.tile(a,[1,self.lumitexel_size,1])
        elif pd_ps_wanted == "ps_only":
            tmp = self.ggx_brdf_aniso(wi_local,wo_local,ax,ay,specular_component)#[batch,lightnum,1]
            b = tmp#[batch,lightnum,1]
            ps = tf.expand_dims(ps,axis=1)#[batch,1,channel]
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

    def compute_local_frame_wi_wo(self,input_params,position,rotate_theta,variable_scope_name,rotate_point = True,rotate_frame=True,use_custom_frame="",global_custom_frame=[],new_cam_pos=None,use_new_cam_pos=False):
        with tf.variable_scope(variable_scope_name):
            ###[STEP 0]
            #load constants
            view_mat_for_normal_t,view_mat_model_t,light_normals,light_poses,cam_pos = self.__net_load_constants(variable_scope_name)
            if use_new_cam_pos:
                cam_pos = new_cam_pos
            #rotate object           
            rotate_axis = tf.constant(np.repeat(np.array([0,0,1],np.float32).reshape([1,3]),self.fitting_batch_size,axis=0))
            view_mat_model = self.rotation_axis(rotate_theta,rotate_axis)#[batch,4,4]
            view_mat_model_t = tf.matrix_transpose(view_mat_model)

            view_mat_for_normal =tf.matrix_transpose(tf.matrix_inverse(view_mat_model))
            view_mat_for_normal_t = tf.matrix_transpose(view_mat_for_normal)
            # self.endPoints[variable_scope_name+"view_mat_for_normal_t"] = view_mat_for_normal
            ###[STEP 1]
            ##define input
            with tf.variable_scope("compute_global_frame"):
                view_dir = cam_pos - position #shape=[batch,3]
                view_dir = self.normalized(view_dir)#shape=[batch,3]
                #build local frame
                frame_t,frame_b = self.build_frame_f_z(view_dir,None,with_theta=False)#[batch,3]
                frame_n = view_dir

            ###[STEP 1.1]
            ###split input parameters into position and others
            if self.if_grey_scale:
                n_2d,theta,ax,ay,pd,ps = tf.split(input_params,[2,1,1,1,1,1],axis=1)
                pd = tf.tile(pd,[1,3])
                ps = tf.tile(ps,[1,3])
            else:
                n_local,t_local,ax,ay,pd,ps = tf.split(input_params,[2,1,1,1,3,3],axis=1)

            #position shape=[bach,3]
            
            if "n" in use_custom_frame:
                n = global_custom_frame[0]
                if "t" in use_custom_frame:
                    t = global_custom_frame[1]
                    b = global_custom_frame[2]
                else:
                    t,b = self.build_frame_f_z(n,None,with_theta=False)
            else:
                n_local = self.back_hemi_octa_map(n_2d)#[batch,3]
                self.endPoints[variable_scope_name+"normal_local"] = n_local
                t_local,_ = self.build_frame_f_z(n_local,theta,with_theta=True)
                n_local_x,n_local_y,n_local_z = tf.split(n_local,[1,1,1],axis=1)#[batch,1],[batch,1],[batch,1]
                n = n_local_x*frame_t+n_local_y*frame_b+n_local_z*frame_n#[batch,3]
                self.endPoints[variable_scope_name+"normal"]  = n
                t_local_x,t_local_y,t_local_z = tf.split(t_local,[1,1,1],axis=1)#[batch,1],[batch,1],[batch,1]
                t = t_local_x*frame_t+t_local_y*frame_b+t_local_z*frame_n#[batch,3]
                b = tf.cross(n,t)#[batch,3]
            
            if rotate_frame:
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

            if rotate_point:
                position = tf.expand_dims(tf.concat([position,tf.ones([position.shape[0],1],tf.float32)],axis=1),axis=1)
                position = tf.squeeze(tf.matmul(position,view_mat_model_t),axis=1)#position@view_mat_model_t
                position,_ = tf.split(position,[3,1],axis=1)#shape=[batch,3]

            #get real view dir
            view_dir = cam_pos - position #shape=[batch,3]
            view_dir = self.normalized(view_dir)#shape=[batch,3]

            light_poses_broaded = tf.tile(tf.expand_dims(light_poses,axis=0),[self.fitting_batch_size,1,1],name="expand_light_poses")#shape is [batch,lightnum,3]
            position_broded = tf.tile(tf.expand_dims(position,axis=1),[1,self.lumitexel_size,1],name="expand_position")
            wi = light_poses_broaded-position_broded
            wi = self.normalized_nd(wi)#shape is [batch,lightnum,3]

            wi_local = tf.concat([self.dot_ndm_vector(wi,tf.expand_dims(t,axis=1)),
                                    self.dot_ndm_vector(wi,tf.expand_dims(b,axis=1)),
                                    self.dot_ndm_vector(wi,tf.expand_dims(n,axis=1))],axis=-1)#shape is [batch,lightnum,3]
            
            view_dir_broaded = tf.tile(tf.expand_dims(view_dir,axis=1),[1,self.lumitexel_size,1])#shape is [batch,lightnum,3]
            wo_local = tf.concat([self.dot_ndm_vector(view_dir_broaded,tf.expand_dims(t,axis=1)),
                                    self.dot_ndm_vector(view_dir_broaded,tf.expand_dims(b,axis=1)),
                                    self.dot_ndm_vector(view_dir_broaded,tf.expand_dims(n,axis=1))],axis=-1)#shape is [batch,lightnum,3]

            n_dot_view_dir = self.dot_ndm_vector(view_dir_broaded,tf.tile(tf.expand_dims(n,axis=1),[1,self.lumitexel_size,1]))#[batch,lightnum,1]
                
            n_dot_views = tf.gather(n_dot_view_dir,0,axis=1)

            return n,t,b,wi_local,wo_local,n_dot_views

    def draw_custom_spec_rendering(self,input_params,rotate_theta,variable_scope_name,component="s_f_i_o_u_b_F_D",global_shared_frame=None,use_global_shared_frame=False):
        '''
        rendr_params is a list of render param
        return =[batch,slice_len]
        '''
        with tf.variable_scope(variable_scope_name):
            param_alpha = input_params[3]
            param_ax,param_ay = tf.split(param_alpha,[1,1],axis=-1)
            param_ps =  input_params[5]

            if use_global_shared_frame:
                #use the precomputed(gt) frame
                param_normal = global_shared_frame[0]
                param_tangent = global_shared_frame[1]
                param_binormal = global_shared_frame[2]
                param_pos = global_shared_frame[3]
            else:
                #use the predicted normal ,tangent and position 
                param_normal = input_params[0]
                param_tangent = input_params[1]
                param_binormal = tf.cross(param_normal,param_tangent)
                param_pos = input_params[2]

            t = param_tangent
            n = param_normal
            b = param_binormal

            position = param_pos
            position_broded = tf.tile(tf.expand_dims(position,axis=1),[1,self.lumitexel_size,1],name="expand_position")

            _,_,light_normals,light_poses,cam_pos = self.__net_load_constants(variable_scope_name)
            light_poses_broaded = tf.tile(tf.expand_dims(light_poses,axis=0),[self.fitting_batch_size,1,1],name="expand_light_poses")#shape is [batch,lightnum,3]
            light_normals_broaded = tf.tile(tf.expand_dims(light_normals,axis=0),[self.fitting_batch_size,1,1],name="expand_light_normals")#shape is [batch,lightnum,3]

            view_dir = cam_pos - position #shape=[batch,3]
            view_dir = self.normalized(view_dir)#shape=[batch,3]

            wi = light_poses_broaded-position_broded
            wi = self.normalized_nd(wi)#shape is [batch,lightnum,3]

            wi_local = tf.concat([self.dot_ndm_vector(wi,tf.expand_dims(t,axis=1)),
                                      self.dot_ndm_vector(wi,tf.expand_dims(b,axis=1)),
                                      self.dot_ndm_vector(wi,tf.expand_dims(n,axis=1))],axis=-1)#shape is [batch,lightnum,3]
                
            # view_dir_broaded = tf.tile(tf.expand_dims(view_dir,axis=1),[1,self.lumitexel_size,1])#shape is [batch,lightnum,3]
            wo_local = tf.concat([self.dot_ndm_vector(view_dir,t),
                                    self.dot_ndm_vector(view_dir,b),
                                    self.dot_ndm_vector(view_dir,n)],axis=-1)#shape is [batch,3]
            wo_local = tf.tile(tf.expand_dims(wo_local,axis=1),[1,self.lumitexel_size,1])#shape is [batch,lightnum,3]

            wi_z = tf.expand_dims(tf.gather(wi_local,indices=2,axis=2,name="wi_z"),axis=-1)#shape=[batch,lightnum,1]
            wo_z = tf.expand_dims(tf.gather(wo_local,indices=2,axis=2,name="wo_z"),axis=-1)#shape=[batch,lightnum,1]


            
            with tf.variable_scope("compute_scalar") as vs1:
                ldir = light_poses_broaded-tf.expand_dims(param_pos,axis=1)#[batch,lightnum,3]
                dist = tf.sqrt(tf.reduce_sum(tf.square(ldir),axis=-1,keepdims=True))#[batch,lightnum,1]
                tmp_b = dist*dist#[batch,lightnum,1]
                f =  4.0*tmp_b+1e-6#[batch,lightnum,1]
                ps = tf.expand_dims(param_ps,axis=1)#[batch,1,1]
            
            with tf.variable_scope("compute_G_i") as vs1:
                G_i = self.ggx_G1_aniso(wi_local,param_ax,param_ay,wi_z)#[batch,lightnum,1]
                # G_i = tf.squeeze(G_i,axis=-1)

            with tf.variable_scope("compute_G_o") as vs1:
                ldir = self.normalized_nd(ldir)
                wi_dot_nl = tf.maximum(self.dot_ndm_vector(ldir,light_normals_broaded),0.0)#[batch,lightnum,1]
                G_o = self.ggx_G1_aniso(wo_local,param_ax,param_ay,wo_z) 
                u =  wi_dot_nl#[batch,lightnum,1]
                b = wo_z#[batch,lightnum,1]

            with tf.variable_scope("compute_ps_frenel") as vs1:
                vhalf = self.normalized_nd(wi_local+wo_local)#[batch,lightnum,3]
                tmp = tf.minimum(tf.maximum(0.0,1-self.dot_ndm_vector(wi_local,vhalf)),1.0)#[batch,lightnum,1]
                F0 = 0.04
                F = F0+(1-F0)* tmp * tmp * tmp * tmp * tmp#[batch,lightnum,1]
                # F = tf.squeeze(F,axis=-1)

            with tf.variable_scope("compute_D") as vs1:
                axayaz = tf.expand_dims(tf.concat([param_ax,param_ay,tf.ones([self.fitting_batch_size,1],tf.float32)],axis=-1),axis=1)#[batch,1,3]
                vhalf = vhalf/(axayaz+1e-6)
                vhalf_norm = self.norm(vhalf)#[batch,lightnum,1]
                length = vhalf_norm*vhalf_norm##[batch,lightnum,1]
                D = 1.0/(math.pi*tf.expand_dims(param_ax,axis=1)*tf.expand_dims(param_ay,axis=1)*length*length)#[batch,lightnum,1]

            with tf.variable_scope("construct_spec_slice")as vs1:
                rendered_slice_spec = tf.ones([self.fitting_batch_size,self.lumitexel_size,1],tf.float32)#1e4*math.pi*1e-2#
                if "s" in component:
                    print("[CUSTOM RENDER] with ps!")
                    rendered_slice_spec = rendered_slice_spec * ps
                if "f" in component:
                    print("[CUSTOM RENDER] with squared distance!")
                    rendered_slice_spec = rendered_slice_spec / f
                if "i" in component:
                    print("[CUSTOM RENDER] with G_i!")
                    rendered_slice_spec = rendered_slice_spec * G_i
                if "o" in component:
                    print("[CUSTOM RENDER] with G_o!")
                    rendered_slice_spec = rendered_slice_spec * G_o
                if "u" in component:
                    print("[CUSTOM RENDER] with wi_dot_lightnormal!")
                    rendered_slice_spec = rendered_slice_spec * u
                if "b" in component:
                    print("[CUSTOM RENDER] with wo_dot_normal!")
                    rendered_slice_spec = rendered_slice_spec / b
                if "F" in component:
                    print("[CUSTOM RENDER] with Frenel!")
                    rendered_slice_spec = rendered_slice_spec * F
                if "D" in component:
                    print("[CUSTOM RENDER] with D!")
                    rendered_slice_spec = rendered_slice_spec * D            
                rendered_slice_spec = tf.squeeze(rendered_slice_spec,axis=-1)
        return rendered_slice_spec

    def compute_wo_dot_n(self,position,rotate_theta,n,new_cam_pos,batch_size):
        '''
        This is a bare function which means it doen't use any data of this class!
        
        position:[batchsize,3] global position of a point in guminyi frame
        rotate_theta:[batchsize,1] rotate theta
        n:[batchsize , 3] global normal in guminyi frame 
        new_cam_pos:[batchsize,3]or (3,) cam pos in guminyi frame
        '''
        shape_of_cam = new_cam_pos.get_shape().as_list()
        if len(shape_of_cam) == 1:
            new_cam_pos = tf.expand_dims(new_cam_pos,axis=0)#[1,3]
        else:
            assert len(shape_of_cam) == 2, "shape of cam should be in rank 2,now:{}".format(shape_of_cam)
            assert shape_of_cam[0] == batch_size and shape_of_cam[1] == 3, "shape of cam should be in rank 2 and first is batchsize,second is 3,now:{}".format(shape_of_cam)
        rotate_axis = tf.constant(np.repeat(np.array([0,0,1],np.float32).reshape([1,3]),batch_size,axis=0))
        
        view_mat_model = self.rotation_axis(rotate_theta,rotate_axis)#[batch,4,4]
        view_mat_model_t = tf.matrix_transpose(view_mat_model)
        view_mat_for_normal =tf.matrix_transpose(tf.matrix_inverse(view_mat_model))
        view_mat_for_normal_t = tf.matrix_transpose(view_mat_for_normal)

        pn = tf.expand_dims(tf.concat([n,tf.ones([n.shape[0],1],tf.float32)],axis=1),axis=1)
        
        n = tf.squeeze(tf.matmul(pn,view_mat_for_normal_t),axis=1)
        n,_ = tf.split(n,[3,1],axis=1)#shape=[batch,3]          
        
        position = tf.expand_dims(tf.concat([position,tf.ones([position.shape[0],1],tf.float32)],axis=1),axis=1)
        position = tf.squeeze(tf.matmul(position,view_mat_model_t),axis=1)#position@view_mat_model_t
        position,_ = tf.split(position,[3,1],axis=1)#shape=[batch,3]

        view_dir = new_cam_pos - position #shape=[batch,3]
        view_dir = self.normalized(view_dir)#shape=[batch,3]
        n_dot_views = self.dot_ndm_vector(view_dir,n)#[batch,1]

        return n_dot_views

    def draw_rendering_net(self,input_params,position,rotate_theta,variable_scope_name,with_cos = True,pd_ps_wanted="both",rotate_point = True,specular_component="D_F_G_B",global_custom_frame=None,use_custom_frame="",rotate_frame=True,new_cam_pos=None,use_new_cam_pos=False):
        '''
        input_params = (rendering parameters) shape = [self.fitting_batch_size,self.parameter_len] i.e.[24576,10]
        position = (rendering positions) shape=[self.fitting_batch_size,3]
        variable_scope_name = (for variable check a string like"rendering1") 
        rotate_theta = [self.fitting_batch_size,1]
        return shape = (rendered results)[batch,lightnum,1] or [batch,lightnum,3]
        specular_component means the degredient of brdf(B stands for bottom)
        "D_F_G_B"


        with_cos: if True,lumitexl is computed with cos and dir
        '''
        with tf.variable_scope(variable_scope_name):
            ###[STEP 0]
            #load constants
            view_mat_for_normal_t,view_mat_model_t,light_normals,light_poses,cam_pos = self.__net_load_constants(variable_scope_name)
            if use_new_cam_pos:
                cam_pos = new_cam_pos
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
            else:
                n_2d,theta,ax,ay,pd,ps = tf.split(input_params,[2,1,1,1,3,3],axis=1)

            #position shape=[bach,3]
            # n_2d = tf.clip_by_value(n_2d,0.0,1.0)
            if "n" in use_custom_frame:
                n = global_custom_frame[0]
                if "t" in use_custom_frame:
                    t = global_custom_frame[1]
                    b = global_custom_frame[2]
                else:
                    t,b = self.build_frame_f_z(n,None,with_theta=False)
            else:
                n_local = self.back_hemi_octa_map(n_2d)#[batch,3]
                self.endPoints[variable_scope_name+"normal_local"] = n_local
                t_local,_ = self.build_frame_f_z(n_local,theta,with_theta=True)
                n_local_x,n_local_y,n_local_z = tf.split(n_local,[1,1,1],axis=1)#[batch,1],[batch,1],[batch,1]
                n = n_local_x*frame_t+n_local_y*frame_b+n_local_z*frame_n#[batch,3]
                self.endPoints[variable_scope_name+"normal"]  = n
                t_local_x,t_local_y,t_local_z = tf.split(t_local,[1,1,1],axis=1)#[batch,1],[batch,1],[batch,1]
                t = t_local_x*frame_t+t_local_y*frame_b+t_local_z*frame_n#[batch,3]
                b = tf.cross(n,t)#[batch,3]
            
            if rotate_frame:
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
                if rotate_point:
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
                
                # view_dir_broaded = tf.tile(tf.expand_dims(view_dir,axis=1),[1,self.lumitexel_size,1])#shape is [batch,lightnum,3]
                wo_local = tf.concat([self.dot_ndm_vector(view_dir,t),
                                      self.dot_ndm_vector(view_dir,b),
                                      self.dot_ndm_vector(view_dir,n)],axis=-1)#shape is [batch,3]
                wo_local = tf.tile(tf.expand_dims(wo_local,axis=1),[1,self.lumitexel_size,1])#shape is [batch,lightnum,3]

                self.endPoints[variable_scope_name+"wi_local"] = wi_local
                self.endPoints[variable_scope_name+"wo_local"] = wo_local
                
                self.endPoints[variable_scope_name+"light_poses_broaded"] = light_poses_broaded
                self.endPoints[variable_scope_name+"light_normals_broaded"] = light_normals_broaded
                form_factors = self.compute_form_factors(position,n,light_poses_broaded,light_normals_broaded,variable_scope_name,with_cos)#[batch,lightnum,1]
                self.endPoints[variable_scope_name+"form_factors"] = form_factors
                lumi = self.calc_light_brdf(wi_local,wo_local,ax,ay,pd,ps,pd_ps_wanted,specular_component)#[batch,lightnum,channel]
                self.endPoints[variable_scope_name+"lumi_without_formfactor"] = lumi
                
                lumi = lumi*form_factors*1e4*math.pi*1e-2#[batch,lightnum,channel]

                wi_dot_n = self.dot_ndm_vector(wi,tf.expand_dims(n,axis=1))#[batch,lightnum,1]
                lumi = lumi*((tf.sign(wi_dot_n)+1.0)*0.5)
                # judgements = tf.less(wi_dot_n,1e-5)
                # lumi = tf.where(judgements,tf.zeros([self.fitting_batch_size,self.lumitexel_size,1]),lumi)

                n_dot_views = self.dot_ndm_vector(view_dir,n)#[batch,1]
                n_dot_view_dir = tf.tile(tf.expand_dims(n_dot_views,axis=1),[1,self.lumitexel_size,1])#[batch,lightnum,1]
                
                self.endPoints[variable_scope_name+"n_dot_view_dir"] = n_dot_views

                judgements = tf.less(n_dot_view_dir,0.0)
                if self.if_grey_scale:
                    rendered_results = tf.where(judgements,tf.zeros([self.fitting_batch_size,self.lumitexel_size,1]),lumi)#[batch,lightnum]
                else:
                    rendered_results = tf.where(tf.tile(judgements,[1,1,3]),tf.zeros([self.fitting_batch_size,self.lumitexel_size,3]),lumi)#[batch,lightnum]
                self.endPoints[variable_scope_name+"rendered_results"] = rendered_results

        return rendered_results

    def get_n_dot_view_penalty(self,variable_scope_name):
        return self.endPoints[variable_scope_name+"n_dot_view_penalty"],self.endPoints[variable_scope_name+"n_dot_view_dir"]
    
    def get_n(self,variable_scope_name):
        print("[RENDERER] WARNING! THIS FUNCTION IS NOT SAFE! PLEASE USE get_compute_node!")
        input()
        return self.endPoints[variable_scope_name+"normal_local"],self.endPoints[variable_scope_name+"n"]
    
    def get_pd(self,variable_scope_name):
        return self.endPoints[variable_scope_name+"pd"]
    
    def get_wi(self,variable_scope_name):
        return self.endPoints[variable_scope_name+"wi"]
    
    def get_compute_node(self,variable_scope_name,node_name):
        return self.endPoints[variable_scope_name+node_name]

if __name__ == "__main__":
    import cv2
    RENDER_SCALAR = 5*1e3/math.pi

    log_root = sys.argv[1]
    shrink_size = 1
    parameters = {}
    parameters["batch_size"] = 5
    parameters["lumitexel_size"] = 24576//shrink_size//shrink_size
    parameters["shrink_step"] = shrink_size
    parameters["is_grey_scale"] = True
    parameters["is_real_light"] = True
    parameters["config_dir"] = './tf_ggx_render_configs_{}x{}/'.format(shrink_size,shrink_size)#指向随包的config文件夹

    if parameters["is_grey_scale"]:  
        parameter_len = 7
    else:  
        parameter_len = 11
    
    renderer = tf_ggx_render(parameters)#实例化渲染器

    # test_params = np.array([
    #     [0.5,0.5,0.0,0.01,0.015,1,1,1,5,5,5],#n1 n2 t ax ay pd3 ps3
    #     [0.5,0.5,0.0,0.01,0.015,1,1,1,0,5,5],#n1 n2 t ax ay pd3 ps3
    #     [0.5,0.5,0.0,0.01,0.015,1,1,1,5,0,5],#n1 n2 t ax ay pd3 ps3
    #     [0.5,0.5,0.0,0.01,0.015,1,1,1,5,5,5],#n1 n2 t ax ay pd3 ps3
    #     [0.5,0.5,0.0,0.01,0.015,1,1,1,5,5,0],#n1 n2 t ax ay pd3 ps3
    # ],np.float32)

    test_params = np.array([
        [0.5,0.5,0.0,0.01,0.015,1,5],#n1 n2 t ax ay pd3 ps3
        [0.5,0.5,0.0,0.01,0.015,1,5],#n1 n2 t ax ay pd3 ps3
        [0.5,0.5,0.0,0.01,0.015,1,5],#n1 n2 t ax ay pd3 ps3
        [0.5,0.5,0.0,0.01,0.015,1,5],#n1 n2 t ax ay pd3 ps3
        [0.5,0.5,0.0,0.01,0.015,1,5],#n1 n2 t ax ay pd3 ps3
    ],np.float32)

    test_poses = np.array([
        [0.0,0.0,0.0],
        [0.0,0.0,0.0],
        [0.0,0.0,0.0],
        [0.0,0.0,0.0],
        [0.0,0.0,0.0],
    ],np.float32)

    with tf.Session() as sess:
        #define inputs
        input_params = tf.placeholder(tf.float32,shape=[parameters["batch_size"],parameter_len],name="render_params")
        input_positions = tf.placeholder(tf.float32,shape=[parameters["batch_size"],3],name="render_positions")

        rotate_theta_zero = tf.zeros([parameters["batch_size"],1],tf.float32)

        with tf.variable_scope("frame_definition_gt"):
            n_2d,theta,nnt_param = tf.split(input_params,[2,1,parameter_len-3],axis=1)
            normal = renderer.back_full_octa_map(n_2d)
            tangent,binormal = renderer.build_frame_f_z(normal,theta,with_theta=True)

            global_frame = [normal,tangent,binormal]

        #draw rendering net
        rendered_res = renderer.draw_rendering_net(input_params,input_positions,rotate_theta_zero,"my_little_render",global_custom_frame=global_frame,use_custom_frame="ntb",rotate_frame=True,rotate_point=True)
        
        #global variable init should be called before rendering
        init = tf.global_variables_initializer()
        sess.run(init)

        data_ptr = 0
        while True:
            tmp_params = test_params[data_ptr:data_ptr+parameters["batch_size"]]
            tmp_poses = test_poses[data_ptr:data_ptr+parameters["batch_size"]]
            if tmp_params.shape[0] == 0:
                break
            result = sess.run(rendered_res,feed_dict={
                input_params: tmp_params,
                input_positions :np.zeros([parameters["batch_size"],3],np.float32)
            })
            for idx in range(result.shape[0]):
                img = renderer.visualize_lumi_vector(result[idx],scalerf=255.0*RENDER_SCALAR)
                cv2.imwrite(log_root+"{}.png".format(data_ptr+idx),img)
            data_ptr+=tmp_params.shape[0]