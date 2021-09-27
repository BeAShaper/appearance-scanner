import torch
import math
import numpy as np

def hemi_octa_map(dir):
    '''
    a = [batch,3]
    return = [batch,2]
    '''
    # TODO ADD DIM ASSERT

    p = dir/torch.sum(torch.abs(dir),dim=1,keepdim=True)#[batch,3]
    result = torch.cat([p[:,[0]] - p[:,[1]],p[:,[0]] + p[:, [1]]],dim=1) * 0.5 + 0.5
    return result

def back_hemi_octa_map(a):
    '''
    a = [batch,2]
    return = [batch,3]
    '''
    # TODO ADD DIM ASSERT
    p = (a - 0.5)*2.0
    resultx = (p[:,[0]]+p[:,[1]])*0.5#[batch,1]
    resulty = (p[:,[1]]-p[:,[0]])*0.5#[batch,1]
    resultz = 1.0-torch.abs(resultx)-torch.abs(resulty)
    result = torch.cat([resultx,resulty,resultz],dim=1)#[batch,3]

    return torch.nn.functional.normalize(result,dim=1)

def full_octa_map(dir):
    '''
    dir = [batch,3]
    return=[batch,2]
    '''
    # TODO ADD DIM ASSERT
    p = dir/torch.sum(torch.abs(dir),dim=1,keepdim=True)
    px,py,pz = torch.split(p,[1,1,1],dim=1)
    x,y = px,py

    judgements1 = torch.ge(px,0.0)
    x_12 = torch.where(judgements1,1.0-py,-1.0+py)#[batch,1]
    y_12 = torch.where(judgements1,1.0-px,1.0+px)

    judgements2 = torch.le(px,0.0)
    x_34 = torch.where(judgements2,-1.0-py,1.0+py)
    y_34 = torch.where(judgements2,-1.0-px,-1.0+px)

    judgements3 = torch.ge(py,0.0)
    x_1234 = torch.where(judgements3,x_12,x_34)#[batch,1]
    y_1234 = torch.where(judgements3,y_12,y_34)#[batch,1]

    judgements4 = torch.lt(dir[:,[2]],0.0)
    x = torch.where(judgements4,x_1234,x)
    y = torch.where(judgements4,y_1234,y)

    return (torch.cat([x,y],dim=1)+1.0)*0.5

def back_full_octa_map(a):
    '''
    a = [batch,2]
    return = [batch,3]
    '''
    #TODO ADD DIM ASSERT
    p = a*2.0-1.0
    px,py = torch.split(p,[1,1],dim=1)#px=[batch,1] py=[batch,1]
    x = px#px=[batch,1]
    y = py#px=[batch,1]
    abs_px_abs_py = torch.abs(px)+torch.abs(py)
    
    judgements2 = torch.ge(py,0.0)
    judgements3 = torch.ge(px,0.0)
    judgements4 = torch.le(px,0.0)

    x_12 = torch.where(judgements3,1.0-py,-1.0+py)
    y_12 = torch.where(judgements3,1.0-px,1.0+px)

    x_34 = torch.where(judgements4,-1.0-py,1.0+py)
    y_34 = torch.where(judgements4,-1.0-px,-1.0+px)

    x_1234 = torch.where(judgements2,x_12,x_34)
    y_1234 = torch.where(judgements2,y_12,y_34)

    
    judgements1 = torch.gt(abs_px_abs_py,1)

    resultx = torch.where(judgements1,x_1234,px)#[batch,1]
    resulty = torch.where(judgements1,y_1234,py)#[batch,1]
    resultz = 1.0-torch.abs(resultx)-torch.abs(resulty)
    resultz = torch.where(judgements1,-1.0*resultz,resultz)

    result = torch.cat([resultx,resulty,resultz],dim=1)#[batch,3]

    return torch.nn.functional.normalize(result,dim=1)

def build_frame_f_z(n,theta,with_theta=True):
    '''
    n = [batch,3]
    return =t[batch,3] b[batch,3]
    '''
    #TODO ADD DIM ASSERT
    nz = n[:,[2]]
    batch_size = nz.size()[0]

    # try:
    #     constant_001 = build_frame_f_z.constant_001
    #     constant_100 = build_frame_f_z.constant_100
    # except AttributeError:
    #     build_frame_f_z.constant_001 = torch.from_numpy(np.expand_dims(np.array([0,0,1],np.float32),0).repeat(batch_size,axis=0)).to(device)#[batch,3]
    #     build_frame_f_z.constant_100 = torch.from_numpy(np.expand_dims(np.array([1,0,0],np.float32),0).repeat(batch_size,axis=0)).to(device)#[batch,3]
    #     constant_001 = build_frame_f_z.constant_001
    #     constant_100 = build_frame_f_z.constant_100
    
    constant_001 = torch.zeros_like(n)
    constant_001[:,2] = 1.0
    constant_100 = torch.zeros_like(n)
    constant_100[:,0] = 1.0

    nz_notequal_1 = torch.gt(torch.abs(nz-1.0),1e-6)
    nz_notequal_m1 = torch.gt(torch.abs(nz+1.0),1e-6)

    t = torch.where(nz_notequal_1&nz_notequal_m1,constant_001,constant_100)#[batch,3]

    t = torch.nn.functional.normalize(torch.cross(t,n),dim=1)#[batch,3]
    b = torch.cross(n,t)#[batch,3]

    if not with_theta:
        return t,b

    t = torch.nn.functional.normalize(t*torch.cos(theta)+b*torch.sin(theta),dim=1)

    b = torch.nn.functional.normalize(torch.cross(n,t),dim=1)#[batch,3]
    return t,b

def rotation_axis(t,v,isRightHand=True):
    '''
    t = [batch,1]#rotate rad
    v = [batch,3] or [3]#rotate axis(global) 
    return = [batch,4,4]#rotate matrix
    '''
    if isRightHand:
        theta = t
    else:
        print("[RENDERER]Error rotate system doesn't support left hand logic!")
        exit()
    
    batch_size = t.size()[0]

    c = torch.cos(theta)#[batch,1]
    s = torch.sin(theta)#[batch,1]

    v_x,v_y,v_z = torch.split(v,[1,1,1],dim=-1)

    m_11 = c + (1-c)*v_x*v_x
    m_12 = (1 - c)*v_x*v_y - s*v_z
    m_13 = (1 - c)*v_x*v_z + s*v_y

    m_21 = (1 - c)*v_x*v_y + s*v_z
    m_22 = c + (1-c)*v_y*v_y
    m_23 = (1 - c)*v_y*v_z - s*v_x

    m_31 = (1 - c)*v_z*v_x - s*v_y
    m_32 = (1 - c)*v_z*v_y + s*v_x
    m_33 = c + (1-c)*v_z*v_z

    tmp_zeros = torch.zeros_like(t)
    tmp_ones = torch.ones_like(t)


    res = torch.cat([
        m_11,m_12,m_13,tmp_zeros,
        m_21,m_22,m_23,tmp_zeros,
        m_31,m_32,m_33,tmp_zeros,
        tmp_zeros,tmp_zeros,tmp_zeros,tmp_ones
    ],dim=1)

    res = res.view(-1,4,4)
    return res

def compute_form_factors(position,n,light_poses,light_normals,end_points,with_cos=True):
    '''
    position = [batch,3]
    n = [batch,3]
    light_poses = [lightnum,3]
    light_normals = [lightnum,3]

    with_cos: if this is true, form factor adds cos(ldir.light_normals)  

    return shape=[batch,lightnum,1]
    '''
    ldir = torch.unsqueeze(light_poses,dim=0)-torch.unsqueeze(position,dim=1)#[batch,lightnum,3]
    dist = torch.sqrt(torch.sum(ldir**2,dim=2,keepdim=True))#[batch,lightnum,1]
    ldir = torch.nn.functional.normalize(ldir,dim=2)#[batch,lightnum,3]

    static_zero = torch.zeros(1,device=n.device)

    a = torch.max(torch.sum(ldir*torch.unsqueeze(n,dim=1),dim=2,keepdim=True),static_zero)#[batch,lightnum,1]
    if not with_cos:
        return a
    b = dist*dist#[batch,lightnum,1]
    c = torch.max(torch.sum(ldir*torch.unsqueeze(light_normals,dim=0),dim=2,keepdim=True),static_zero)#[batch,lightnum,1]
    # r_2_cos = b/(c+1e-6)
    # cos_r_2 = c/b
    # self.endPoints[variable_scope_name+"r_2_cos"] = r_2_cos
    # self.endPoints[variable_scope_name+"cos_r_2"] = cos_r_2
    # self.endPoints[variable_scope_name+"cos2"] = a
    return a/(b+1e-6)*c

def compute_form_factor_bc(position,light_poses,light_normals):
    '''
    position = (batchsize,3)
    light_poses = (lightnum,3)
    '''
    ldir = torch.unsqueeze(light_poses,dim=0)-torch.unsqueeze(position,dim=1)#[batch,lightnum,3]
    dist = torch.sqrt(torch.sum(ldir**2,dim=2,keepdim=True))#[batch,lightnum,1]
    ldir = torch.nn.functional.normalize(ldir,dim=2)#[batch,lightnum,3]
    b = dist*dist#[batch,lightnum,1]
    c = torch.max(torch.sum(ldir*torch.unsqueeze(light_normals,dim=0),dim=2,keepdim=True),torch.zeros(1,device=position.device))#[batch,lightnum,1]

    return c/(b+1e-6)

def ggx_G1_aniso_honntai(v,vz,ax,ay):
    axayaz = torch.cat([ax,ay,torch.ones_like(ax)],dim=1)#[batch,3]
    vv = v*torch.unsqueeze(axayaz,dim=1)#[batch,lightnum,3]
    # return 2.0/(1.0+(self.norm(vv)/(vz+1e-6)))
    return 2.0*vz/(vz+torch.norm(vv,dim=2,keepdim=True)+1e-6)#[batch,lightnum,1]

def ggx_G1_aniso(v,ax,ay,vz):
    '''
    v = [batch,lightnum,3]
    ax = [batch,1]
    ay = [batch,1]
    vz = [batch,lightnum,1] 
    return shape = [batch,lightnum,1]
    '''
    return torch.where(torch.le(vz,torch.zeros_like(vz)),torch.zeros_like(vz),ggx_G1_aniso_honntai(v,vz,ax,ay))#[batch,lightnum,1]

    # comparison = (tf.sign(vz)+1.0)*0.5
    # hontai = ggx_G1_aniso_honntai(v,vz,ax,ay)
    # return hontai*comparison

def ggx_brdf_aniso(wi,wo,ax,ay,specular_component):
    '''
    wi = [batch,lightnum,3]
    wo = [batch,3]
    ax = [batch,1]
    ay = [batch,1]
    return shape = [batch,lightnum,1]
    '''
    static_one = torch.ones(1,device=wi.device,dtype=torch.float32)
    static_zero = torch.zeros_like(static_one)

    wo = torch.unsqueeze(wo,dim=1).repeat(1,wi.size()[1],1)#[batch,lightnum,3]

    wi_z = wi[:,:,[2]]#tf.expand_dims(tf.gather(wi,indices=2,axis=2,name="wi_z"),axis=-1)#shape=[batch,lightnum,1]
    wo_z = wo[:,:,[2]]#tf.expand_dims(tf.gather(wo,indices=2,axis=2,name="wo_z"),axis=-1)#shape=[batch,lightnum,1]
    denom = 4*wi_z*wo_z#shape=[batch,lightnum,1]
    vhalf = torch.nn.functional.normalize(wi+wo,dim=2)#[batch,lightnum,3]
    tmp = torch.min(torch.max(1.0-torch.sum(wi*vhalf,dim=2,keepdim=True),static_zero),static_one)#[batch,lightnum,1]
    F0 = 0.04
    F = F0+(1-F0)* tmp * tmp * tmp * tmp * tmp#[batch,lightnum,1]
    
    axayaz = torch.unsqueeze(torch.cat([ax,ay,torch.ones_like(ax)],dim=1),dim=1)#[batch,1,3]
    vhalf = vhalf/(axayaz+1e-6)#[batch,lightnum,3]
    vhalf_norm = torch.norm(vhalf,dim=2,keepdim=True)#[batch,lightnum,1]
    length = vhalf_norm*vhalf_norm##[batch,lightnum,1]
    D = 1.0/(math.pi*torch.unsqueeze(ax,dim=1)*torch.unsqueeze(ay,dim=1)*length*length)#[batch,lightnum,1]

    judgement_wiz_less_equal_0 = torch.le(wi_z,static_zero)
    judgement_woz_less_equal_0 = torch.le(wo_z,static_zero)

    tmp_ones = torch.ones_like(denom)
    safe_denom = torch.where(judgement_wiz_less_equal_0,tmp_ones,denom)
    safe_denom = torch.where(judgement_woz_less_equal_0,tmp_ones,safe_denom)

    tmp = tmp_ones
    if "D" in specular_component:
        tmp = tmp * D
    if "F" in specular_component:
        tmp = tmp * F
    if "G" in specular_component:
        tmp = tmp * ggx_G1_aniso(wi,ax,ay,wi_z)*ggx_G1_aniso(wo,ax,ay,wo_z)
    if "B" in specular_component:
        tmp = tmp / (safe_denom+1e-6) 
    
    #[batch,lightnum,1]

    
    tmp_zeros = torch.zeros_like(tmp)


    res = torch.where(judgement_wiz_less_equal_0,tmp_zeros,tmp)
    res = torch.where(judgement_woz_less_equal_0,tmp_zeros,res)

    # wi_z_sign = (tf.sign(wi_z)+1.0)*0.5#shape=[batch,lightnum,1]
    # wo_z_sign = (tf.sign(wo_z)+1.0)*0.5#shape=[batch,lightnum,1]
    
    # # res = tmp*wi_z_sign*wo_z_sign
    # self.endPoints["4"] = wi_z_sign
    # self.endPoints["5"] = wo_z_sign
    # self.endPoints["6"] = denom+1e-6
    return res

def calc_light_brdf(wi_local,wo_local,ax,ay,pd,ps,pd_ps_wanted,specular_component):
    '''
    wi_local = [batch,lightnum,3]
    wo_local = [batch,3]
    ax = [batch,1]
    ay = [batch,1]
    pd = [batch,channel]
    ps = [batch,channel]
    return shape=[batch,lightnum,channel]
    '''
    if pd_ps_wanted == "both":
        b = ggx_brdf_aniso(wi_local,wo_local,ax,ay,specular_component)#[batch,lightnum,1]
        ps = torch.unsqueeze(ps,dim=1)#[batch,1,channel]
        a = torch.unsqueeze(pd/math.pi,dim=1)#[batch,1,1]
        return a+b*ps
    elif pd_ps_wanted =="pd_only":
        a = torch.unsqueeze(pd/math.pi,dim=1)#[batch,1,1]
        return a.repeat(1,wi_local.size()[1],1)
    elif pd_ps_wanted == "ps_only":
        b = ggx_brdf_aniso(wi_local,wo_local,ax,ay,specular_component)#[batch,lightnum,1]
        ps = torch.unsqueeze(ps,dim=1)#[batch,1,channel]
        return b*ps
    # return b*ps# return a+b*ps

def rotate_vector_along_axis(setup,rotate_theta,vector,is_list_input=False):
    batch_size = vector[0].size()[0] if is_list_input else vector.size()[0] 
    device = vector[0].device if is_list_input else vector.device
    
    view_mat_model = rotation_axis(rotate_theta,setup.get_rot_axis_torch(device))#[batch,4,4]
   
    view_mat_for_normal =torch.transpose(torch.inverse(view_mat_model),1,2)
    view_mat_for_normal_t = torch.transpose(view_mat_for_normal,1,2)

    if is_list_input:
        result_list = []
        static_tmp_ones = torch.ones(batch_size,1,dtype=vector[0].dtype,device=device)
        for a_vector in vector:
            pn = torch.unsqueeze(torch.cat([a_vector,static_tmp_ones],dim=1),1)#[batch,1,4]
            
            a_vector = torch.squeeze(torch.matmul(pn,view_mat_for_normal_t),1)[:,:3]
            result_list.append(a_vector)
        return result_list
    else:
        pn = torch.unsqueeze(torch.cat([vector,torch.ones(batch_size,1,dtype=vector.dtype,device=device)],dim=1),1)#[batch,1,4]
        
        vector = torch.squeeze(torch.matmul(pn,view_mat_for_normal_t),1)[:,:3]
        return vector

def rotate_point_along_axis(setup,rotate_theta,points,is_list_input=False):
    batch_size = points[0].size()[0] if is_list_input else points.size()[0] 
    device = points[0].device if is_list_input else points.device
    
    view_mat_model = rotation_axis(rotate_theta,setup.get_rot_axis_torch(device))#[batch,4,4]
    view_mat_model_t = torch.transpose(view_mat_model,1,2)
    
    if is_list_input:
        result_list = []
        static_tmp_ones = torch.ones(batch_size,1,dtype=points[0].dtype,device=device)
        for a_point in points:
            position = torch.unsqueeze(torch.cat([a_point,static_tmp_ones],dim=1),1)#[batch,1,4]#tf.expand_dims(tf.concat([position,tf.ones([position.shape[0],1],tf.float32)],axis=1),axis=1)
            position = torch.squeeze(torch.matmul(position,view_mat_model_t),1)[:,:3]#position@view_mat_model_t#shape=[batch,3]
            result_list.append(position)
        return result_list
    else:
        static_tmp_ones = torch.ones(batch_size,1,dtype=points[0].dtype,device=device)
        position = torch.unsqueeze(torch.cat([points,static_tmp_ones],dim=1),1)#[batch,1,4]#tf.expand_dims(tf.concat([position,tf.ones([position.shape[0],1],tf.float32)],axis=1),axis=1)
        position = torch.squeeze(torch.matmul(position,view_mat_model_t),1)[:,:3]#position@view_mat_model_t#shape=[batch,3]
            
        return position

def compute_wo_dot_n(setup,position,rotate_theta,n,new_cam_pos):
    '''
    This is a bare function which means it doen't use any data of this class!
    
    position:[batchsize,3] global position of a point in guminyi frame
    rotate_theta:[batchsize,1] rotate theta
    n:[batchsize , 3] global normal in guminyi frame 
    new_cam_pos:[batchsize,3]or (3,) cam pos in guminyi frame
    '''
    batch_size = position.size()[0]
    device = position.device
    shape_of_cam = new_cam_pos.size()
    if len(shape_of_cam) == 1:
        new_cam_pos = torch.unsqueeze(new_cam_pos,dim=0)#[1,3]
    else:
        assert len(shape_of_cam) == 2, "shape of cam should be in rank 2,now:{}".format(shape_of_cam)
        assert shape_of_cam[0] == batch_size and shape_of_cam[1] == 3, "shape of cam should be in rank 2 and first is batchsize,second is 3,now:{}".format(shape_of_cam)
    
    view_mat_model = rotation_axis(rotate_theta,setup.get_rot_axis_torch(device))#[batch,4,4]
    view_mat_model_t = torch.transpose(view_mat_model,1,2)
    view_mat_for_normal =torch.transpose(torch.inverse(view_mat_model),1,2)
    view_mat_for_normal_t = torch.transpose(view_mat_for_normal,1,2)

    static_tmp_ones = torch.ones(batch_size,1,dtype=n.dtype,device=device)
    pn = torch.unsqueeze(torch.cat([n,static_tmp_ones],dim=1),1)#[batch,1,4]
    
    n = torch.squeeze(torch.matmul(pn,view_mat_for_normal_t),1)[:,:3]
    
    position = torch.unsqueeze(torch.cat([position,static_tmp_ones],dim=1),1)#[batch,1,4]#tf.expand_dims(tf.concat([position,tf.ones([position.shape[0],1],tf.float32)],axis=1),axis=1)
    position = torch.squeeze(torch.matmul(position,view_mat_model_t),1)[:,:3]#position@view_mat_model_t#shape=[batch,3]

    view_dir = torch.nn.functional.normalize(new_cam_pos - position,dim=1)#shape=[batch,3]
    n_dot_views = torch.sum(view_dir*n,dim=1,keepdim=True)#[batch,1]

    return n_dot_views

def draw_rendering_net(setup,input_params,position,rotate_theta,variable_scope_name,global_custom_frame=None,
    use_custom_frame="",pd_ps_wanted="both",with_cos = True,rotate_point = True,specular_component="D_F_G_B",
    rotate_frame=True,new_cam_pos=None,use_new_cam_pos=False):
    '''
    setup is Setup_Config class
    input_params = (rendering parameters) shape = [self.fitting_batch_size,self.parameter_len] i.e.[24576,10]
    position = (rendering positions) shape=[self.fitting_batch_size,3]
    variable_scope_name = (for variable check a string like"rendering1") 
    rotate_theta = [self.fitting_batch_size,1]
    return shape = (rendered results)[batch,lightnum,1] or [batch,lightnum,3]
    specular_component means the degredient of brdf(B stands for bottom)
    "D_F_G_B"


    with_cos: if True,lumitexl is computed with cos and dir
    '''
    end_points = {}
    batch_size = input_params.size()[0]
    device = input_params.device
    ###[STEP 0]
    #load constants
    light_normals = setup.get_light_normal_torch(device)#[lightnum,3]
    light_poses = setup.get_light_poses_torch(device)#[lightnum,3],
    light_num = light_poses.size()[0]
    cam_pos = setup.get_cam_pos_torch(device)#[3]
    if use_new_cam_pos:
        cam_pos = new_cam_pos
    #rotate object           
    view_mat_model = rotation_axis(rotate_theta,setup.get_rot_axis_torch(device))#[batch,4,4]
    view_mat_model_t = torch.transpose(view_mat_model,1,2)#[batch,4,4]

    view_mat_for_normal =torch.transpose(torch.inverse(view_mat_model),1,2)#[batch,4,4]
    view_mat_for_normal_t = torch.transpose(view_mat_for_normal,1,2)#[batch,4,4]

    test_node = torch.cat([view_mat_model,view_mat_model_t,view_mat_for_normal,view_mat_for_normal_t],dim=0)

    ###[STEP 1] define frame
    view_dir = cam_pos - position #shape=[batch,3]
    view_dir = torch.nn.functional.normalize(view_dir,dim=1)#shape=[batch,3]

    ###[STEP 1.1]
    ###split input parameters into position and others
    if input_params.size()[1] == 7:
        n_2d,theta,ax,ay,pd,ps = torch.split(input_params,[2,1,1,1,1,1],dim=1)
    elif input_params.size()[1] == 11:
        n_2d,theta,ax,ay,pd,ps = torch.split(input_params,[2,1,1,1,3,3],dim=1)
    else:
        print("[RENDER ERROR] error param len!")
        exit(-1)
    #position shape=[bach,3]
    # n_2d = tf.clip_by_value(n_2d,0.0,1.0)
    if "n" in use_custom_frame:
        n = global_custom_frame[0]
        if "t" in use_custom_frame:
            t = global_custom_frame[1]
            b = global_custom_frame[2]
        else:
            t,b = build_frame_f_z(n,None,with_theta=False)
    else:
         #build local frame
        frame_t,frame_b = build_frame_f_z(view_dir,None,with_theta=False)#[batch,3]
        frame_n = view_dir#[batch,3]

        n_local = back_hemi_octa_map(n_2d)#[batch,3]
        t_local,_ = build_frame_f_z(n_local,theta,with_theta=True)
        n = n_local[:,[0]]*frame_t+n_local[:,[1]]*frame_b+n_local[:,[2]]*frame_n#[batch,3]
        t = t_local[:,[0]]*frame_t+t_local[:,[1]]*frame_b+t_local[:,[2]]*frame_n#[batch,3]
        b = torch.cross(n,t)#[batch,3]

    if rotate_frame:
        #rotate frame
        static_tmp_ones = torch.ones(batch_size,1,dtype=torch.float32,device=n.device)
        
        pn = torch.unsqueeze(torch.cat([n,static_tmp_ones],dim=1),1)#[batch,1,4]
        pt = torch.unsqueeze(torch.cat([t,static_tmp_ones],dim=1),1)#[batch,1,4]
        pb = torch.unsqueeze(torch.cat([b,static_tmp_ones],dim=1),1)#[batch,1,4]

        n = torch.squeeze(torch.matmul(pn,view_mat_for_normal_t),1)[:,:3]#[batch,1,4]
        t = torch.squeeze(torch.matmul(pt,view_mat_for_normal_t),1)[:,:3]
        b = torch.squeeze(torch.matmul(pb,view_mat_for_normal_t),1)[:,:3]
    
    if rotate_point:
        position = torch.unsqueeze(torch.cat([position,static_tmp_ones],dim=1),dim=1)#[batch,1,4]
        position = torch.squeeze(torch.matmul(position,view_mat_model_t),dim=1)[:,:3]#shape=[batch,3]
    end_points["n"] = n
    end_points["t"] = t
    end_points["b"] = b
    end_points["position"] = position
    ###[STEP 2]
    ##define rendering

    #get real view dir
    view_dir = torch.unsqueeze(cam_pos,dim=0) - position #shape=[batch,3]
    view_dir = torch.nn.functional.normalize(view_dir,dim=1)#shape=[batch,3]

    # light_poses_broaded = tf.tile(tf.expand_dims(light_poses,axis=0),[self.fitting_batch_size,1,1],name="expand_light_poses")#shape is [batch,lightnum,3]
    # light_normals_broaded = tf.tile(tf.expand_dims(light_normals,axis=0),[self.fitting_batch_size,1,1],name="expand_light_normals")#shape is [batch,lightnum,3]
    wi = torch.unsqueeze(light_poses,dim=0)-torch.unsqueeze(position,dim=1)#[batch,lightnum,3]
    wi = torch.nn.functional.normalize(wi,dim=2)#shape is [batch,lightnum,3]


    wi_local = torch.cat([  torch.sum(wi*torch.unsqueeze(t,dim=1),dim=2,keepdim=True),
                            torch.sum(wi*torch.unsqueeze(b,dim=1),dim=2,keepdim=True),
                            torch.sum(wi*torch.unsqueeze(n,dim=1),dim=2,keepdim=True)],dim=2)#shape is [batch,lightnum,3]
    
    wo_local = torch.cat([  torch.sum(view_dir*t,dim=1,keepdim=True),
                            torch.sum(view_dir*b,dim=1,keepdim=True),
                            torch.sum(view_dir*n,dim=1,keepdim=True)],dim=1)#shape is [batch,3]
    
    
    form_factors = compute_form_factors(position,n,light_poses,light_normals,end_points,with_cos)#[batch,lightnum,1]

    lumi = calc_light_brdf(wi_local,wo_local,ax,ay,pd,ps,pd_ps_wanted,specular_component)#[batch,lightnum,channel]

    end_points["lumi_noff"] = lumi.clone()
    end_points["form_factors"] = form_factors*1e4*math.pi*1e-2
    
    lumi = lumi*form_factors*1e4*math.pi*1e-2#[batch,lightnum,channel]

    wi_dot_n = torch.sum(wi*torch.unsqueeze(n,dim=1),dim=2,keepdim=True)#[batch,lightnum,1]
    # lumi = lumi*((tf.sign(wi_dot_n)+1.0)*0.5)
    lumi = torch.where(torch.lt(wi_dot_n,1e-5),torch.zeros_like(lumi),lumi)#[batch,lightnum,channel]

    n_dot_views = torch.sum(view_dir*n,dim=1,keepdim=True)#[batch,1]
    end_points["n_dot_view_dir"] = n_dot_views
    n_dot_view_dir = torch.unsqueeze(n_dot_views,dim=1).repeat(1,light_num,1)#tf.tile(tf.expand_dims(n_dot_views,axis=1),[1,self.lumitexel_size,1])#[batch,lightnum,1]

    rendered_results = torch.where(torch.lt(n_dot_view_dir,0.0),torch.zeros_like(lumi),lumi)#[batch,lightnum]

    return rendered_results,end_points

def visualize_lumi(lumi,setup_config,is_batch_lumi=True,resize=False):
    '''
    if is_batch_lumi:
        lumi=(batch,lumilen,channel_num) or (batch,lumilen)
        return=(batch,img_height,img_width,3)
    else:
        lumi=(lumilen,channel_num) or (lumilen)
        return=(img_height,img_width,3)
    '''
    if (is_batch_lumi and len(lumi.shape) == 2) or ((not is_batch_lumi) and len(lumi.shape) == 1):
        lumi = np.expand_dims(lumi,axis=-1)

    # tmp_img = np.repeat(np.expand_dims(np.zeros(setup_config.img_size,lumi.dtype),axis=-1),lumi.shape[-1],axis=-1)
    tmp_img = np.repeat(np.expand_dims(np.ones(setup_config.img_size,lumi.dtype),axis=-1),lumi.shape[-1],axis=-1) * 1.0#0.6
    
    if is_batch_lumi:
        tmp_img = np.repeat(np.expand_dims(tmp_img,axis=0),lumi.shape[0],axis=0)
        tmp_img[:,setup_config.visualize_map[:,1],setup_config.visualize_map[:,0]] = lumi
    else:
        tmp_img[setup_config.visualize_map[:,1],setup_config.visualize_map[:,0]] = lumi
        tmp_img = np.expand_dims(tmp_img,axis=0)
    #size=(img_num,originheight,originwidth,3) at this moment
    if tmp_img.shape[3] == 1:
        tmp_img = np.repeat(tmp_img,3,axis=3)
    if resize:
        ratio = setup_config.full_face_size * 3 //setup_config.img_size[0]
        tmp_img = np.expand_dims(np.expand_dims(tmp_img,axis=3),axis=2)#size=(img_num,originheight,1,originwidth,1,channel) at this moment
        tmp_img = np.repeat(np.repeat(tmp_img,ratio,axis=2),ratio,axis=4)
        tmp_img = np.reshape(tmp_img,[tmp_img.shape[0],setup_config.full_face_size*3,setup_config.full_face_size*4,tmp_img.shape[5]])

    if not is_batch_lumi:
        tmp_img = np.squeeze(tmp_img,axis=0)

    return tmp_img

def draw_vector_on_lumi(lumi_img,vector_to_draw,positions,setup_config,is_batch_lumi,color,resize=True,bold=3,length=7):
    '''
    if is_batch_lumi:
        lumi_img=(batch,lumi_img_height(full or not),lumi_img_width(full or not),channel_num)
        positions = (batch,3)
        vector_to_draw = (batch,3)
        return=(batch,lumi_img_height(full or not),lumi_img_width(full or not),3)
    else:
        lumi_img=(lumi_img_height(full or not),lumi_img_width(full or not),channel_num)
        positions = (3,)
        vector_to_draw = (3,)
        return=(lumi_img_height(full or not),lumi_img_width(full or not),3)
    '''

    if not is_batch_lumi:
        lumi_img = np.expand_dims(lumi_img,axis=0)
        positions = np.expand_dims(positions,axis=0)
        vector_to_draw = np.expand_dims(vector_to_draw,axis=0)

    if lumi_img.shape[3] == 1:
        lumi_img = np.repeat(lumi_img,3,axis=3)
    if hasattr(color,"__len__"):
        color = np.ones(3,dtype=lumi_img.dtype)*color
    batch_size = lumi_img.shape[0]
    #lumi_img = (batch,lumi_img_height(full or not),lumi_img_width(full or not),3) at this moment
    #positions = (batch,3)
    #vector_to_draw = (batch,3)
    #color = (3,)

    #get closest light
    light_dir = np.expand_dims(setup_config.light_poses,axis=0) - np.expand_dims(positions,axis=1) #(batch,lightnum,3)
    light_dir = light_dir/np.linalg.norm(light_dir,axis=2,keepdims=True)#(batch,lightnum,3)

    dot_res = np.sum(light_dir*np.expand_dims(vector_to_draw,axis=1),axis=2)#(batch,lightnum)
    max_idx = np.argmax(dot_res,axis=1)#(batch,)
    
    #draw vector in shrinked img
    tmp_img = np.repeat(np.expand_dims(np.zeros([setup_config.img_size[0],setup_config.img_size[1],3],dtype=lumi_img.dtype),axis=0),batch_size,axis=0)#(batch,shrinked_height,shrinked_width,3)
    mask = np.zeros_like(tmp_img)
    light_idxes = setup_config.visualize_map[max_idx]#(batch,2)
    for img_id in range(batch_size):
        tmp_img[img_id][max(0,light_idxes[img_id,1]-length//2):light_idxes[img_id,1]+length//2+1,max(0,light_idxes[img_id,0]-bold//2):light_idxes[img_id,0]+bold//2+1] = color#horizontal
        tmp_img[img_id][max(0,light_idxes[img_id,1]-bold//2):light_idxes[img_id,1]+bold//2+1,max(0,light_idxes[img_id,0]-length//2):light_idxes[img_id,0]+length//2+1] = color#vertical
        mask[img_id][max(0,light_idxes[img_id,1]-length//2):light_idxes[img_id,1]+length//2+1,max(0,light_idxes[img_id,0]-bold//2):light_idxes[img_id,0]+bold//2+1] = 1.0#horizontal
        mask[img_id][max(0,light_idxes[img_id,1]-bold//2):light_idxes[img_id,1]+bold//2+1,max(0,light_idxes[img_id,0]-length//2):light_idxes[img_id,0]+length//2+1] = 1.0#horizontal
    
    if resize:
        ratio = setup_config.full_face_size * 3 //setup_config.img_size[0]
        tmp_img = np.expand_dims(np.expand_dims(tmp_img,axis=3),axis=2)#size=(img_num,originheight,1,originwidth,1,channel) at this moment
        tmp_img = np.repeat(np.repeat(tmp_img,ratio,axis=2),ratio,axis=4)
        tmp_img = np.reshape(tmp_img,[tmp_img.shape[0],setup_config.full_face_size*3,setup_config.full_face_size*4,tmp_img.shape[5]])

        mask = np.expand_dims(np.expand_dims(mask,axis=3),axis=2)#size=(img_num,originheight,1,originwidth,1,channel) at this moment
        mask = np.repeat(np.repeat(mask,ratio,axis=2),ratio,axis=4)
        mask = np.reshape(mask,[mask.shape[0],setup_config.full_face_size*3,setup_config.full_face_size*4,mask.shape[5]])


    lumi_img = np.where(mask > 0.0,tmp_img,lumi_img) 

    if not is_batch_lumi:
        lumi_img = np.squeeze(lumi_img,axis=0)
    
    return lumi_img

def rotate_rowwise(matrix,shifts):
    '''
    matrix = (m,n,d) or (m,n) torch_tensor (I think higher dimension is ok, but to be tested. If you find it goes well, please notify me :))

    shifts = (m,) torch_tensor
    
    shift to left for example:
    
    matrix:
    [[0.74728148 0.76232882 0.91310868 0.86849492]
    [0.36226688 0.3736157  0.37672    0.1656204 ]
    [0.0099235  0.24112559 0.37998685 0.44215475]
    [0.37583682 0.08553133 0.82055228 0.58087278]
    [0.56478736 0.64550805 0.13207896 0.06440142]]
    
    shifts:
    [0 1 1 0 2]
    
    return:
    [[0.74728148 0.76232882 0.91310868 0.86849492]
    [0.3736157  0.37672    0.1656204  0.36226688]
    [0.24112559 0.37998685 0.44215475 0.0099235 ]
    [0.37583682 0.08553133 0.82055228 0.58087278]
    [0.13207896 0.06440142 0.56478736 0.64550805]]
    '''
    # get shape of the input matrix
    shape = matrix.size()

    # compute and stack the meshgrid to get the index matrix of shape (m,n,2)
    ind = torch.stack(torch.meshgrid(torch.arange(shape[0],device=shifts.device,dtype=shifts.dtype),torch.arange(shape[1],device=shifts.device,dtype=shifts.dtype))).permute(1,2,0)
    
    # add the value from shifts to the corresponding row and devide modulo shape[1]
    # this will effectively introduce the desired shift, but at the level of indices
    shifted_ind = torch.fmod(((ind[:,:,1]).permute(1,0) + shifts).permute(1,0),shape[1])
    
    # convert the shifted indices to the right shape
    new_ind = torch.stack([ind[:,:,0],shifted_ind],dim=2).long()
    
    # return the resliced tensor
    return matrix[new_ind[:,:,0],new_ind[:,:,1]]

class Setup_Config(object):
    
    def __init__(self,args):
        self.config_dir = args["config_dir"]
        self.slice_width = 32
        self.slice_height = 16
        
        self.load_constants_from_bin(self.config_dir)

    def load_constants_from_bin(self,config_file_dir):
        #load configs
        self.cam_pos = np.fromfile(config_file_dir+"cam_pos.bin",np.float32)
        assert self.cam_pos.shape[0] == 3
        print("[SETUP]cam_pos:",self.cam_pos)
        
        tmp_data = np.fromfile(config_file_dir+"lights.bin",np.float32).reshape([2,-1,3])
        self.light_poses = tmp_data[0]
        self.light_normals = tmp_data[1]

        self.rot_axis = np.array([0.0,0.0,1.0],np.float32)# TODO read from calibration file

        try:
            with open(self.config_dir+"visualize_config_torch.bin","rb") as pf:
                self.img_size = np.fromfile(pf,np.int32,2)
                self.visualize_map = np.fromfile(pf,np.int32).reshape([-1,2])
                self.full_face_size = 64
        except FileNotFoundError  as identifier:
            print("Lightstage visualize_config_torch doesn't exist")

        try:
            print("Load color tensor from color_tensor_hand.bin")

            self.color_tensor = np.fromfile(config_file_dir+"color_tensor_hand.bin",np.float32).reshape([3,3,3])
        except FileNotFoundError  as identifier:
            print("Color tensor doesn't exist, using Identity Tensor")
            self.color_tensor = np.zeros([3,3,3],np.float32)
            for i in range(3):
                self.color_tensor[i][i][i] = 1.0

        try:
            self.visualize_map = np.fromfile(self.config_dir+"visualize_config_torch_plane_{}x{}.bin".format(self.slice_height,self.slice_width),np.int32).reshape([-1,2])
            self.visualize_map = self.visualize_map[:,[1,0]]
            self.img_size = np.array([self.slice_height,self.slice_width])
            self.light_normals *= -1
        
            ###### load camera extrinsic
            self.camera_extrinsic = np.fromfile(config_file_dir+"extrinsic.bin",np.float32)
            self.cameraR = self.camera_extrinsic[:9].reshape(3,3)
            self.cameraT = self.camera_extrinsic[9:].reshape(1,3)

            self.light_poses = np.matmul(self.light_poses,self.cameraR) + self.cameraT
            self.light_normals = np.matmul(self.light_normals,self.cameraR)

            self.cam_pos = np.matmul(self.cam_pos,self.cameraR) + self.cameraT
            self.cam_pos = np.squeeze(self.cam_pos,axis=0)

            print("[SETUP]cam_pos:",self.cam_pos)

            # origin_pos = np.array([[-75,-75,-75],[-75,-75,75],[-75,75,-75],[-75,75,75],[75,-75,-75],[75,-75,75],[75,75,-75],[75,75,75]],np.float32)
            # # min_ = np.min(origin_pos,axis=0)
            # # max_ = np.max(origin_pos,axis=0)
            # # print(min_, max_)
            # trans_pos = np.matmul(origin_pos,self.cameraR) + self.cameraT
            # min_ = np.min(trans_pos,axis=0)
            # max_ = np.max(trans_pos,axis=0)
            # print(min_, max_)
            # exit()
            print("USE PART LIGHT STAGE")

        except FileNotFoundError  as identifier:
            print("USE FULL LIGHT STAGE")

            ######

    def get_color_tensor(self,custom_device):
        
        return torch.from_numpy(self.color_tensor).to(custom_device)
    
    def get_light_num(self):
        return self.light_poses.shape[0]

    # def get_cam_pos_torch(self):
    #     try:
    #         return self.cam_pos_torch
    #     except AttributeError:
    #         self.cam_pos_torch = torch.from_numpy(self.cam_pos).to(self.device)
    #         return self.cam_pos_torch
    
    def get_cam_pos_torch(self,custom_device):
        return torch.from_numpy(self.cam_pos).to(custom_device)

    def set_cam_pos(self,new_cam_pos):
        self.cam_pos = new_cam_pos

    # def get_light_normal_torch(self):
    #     try:
    #         return self.light_normals_torch
    #     except AttributeError:
    #         self.light_normals_torch = torch.from_numpy(self.light_normals).to(self.device)
    #         return self.light_normals_torch

    def get_light_normal_torch(self,custom_device):
        return torch.from_numpy(self.light_normals).to(custom_device)

    # def get_light_poses_torch(self):
    #     try:
    #         return self.light_poses_torch
    #     except AttributeError:
    #         self.light_poses_torch = torch.from_numpy(self.light_poses).to(self.device)
    #         return self.light_poses_torch
    
    def get_light_poses_torch(self,custom_device):
        return torch.from_numpy(self.light_poses).to(custom_device)

    # def get_rot_axis_torch(self):
    #     try:
    #         return self.rot_axis_torch
    #     except AttributeError:
    #         self.rot_axis_torch = torch.from_numpy(self.rot_axis).to(self.device)
    #         return self.rot_axis_torch
    
    def get_rot_axis_torch(self,custom_device):
        return torch.from_numpy(self.rot_axis).to(custom_device)

    def set_rot_axis_torch(self,axis):
        if axis == 0:
            self.rot_axis = np.array([1.0,0.0,0.0],np.float32)
        elif axis == 1:
            self.rot_axis = np.array([0.0,1.0,0.0],np.float32)
        elif axis == 2:
            self.rot_axis = np.array([0.0,0.0,1.0],np.float32)
        else:
            print("[ERROR] Wrong axis")

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
        new_pos = sub_light_pos
        # new_pos = np.zeros((0,3),np.float32)
        # for row in range(self.slice_height):
        #     for col in range(self.slice_width):
        #         tmp_pos = sub_light_pos[0] - 10 * row * np.array([0.0,0.0,1.0]) - 10 * col * np.array([0.0,1.0,0.0])
        #         tmp_pos = tmp_pos.reshape([1,3])
        #         new_pos = np.concatenate([new_pos,tmp_pos],axis=0)
        return new_pos.astype(sub_light_pos.dtype), sub_light_normals