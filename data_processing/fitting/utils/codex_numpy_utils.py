import numpy as np

def normalize_vector_matrix(a):
    '''
    nd is nd vector
    a = [batch,nd]
    return = [batch,nd]
    '''
    norms = np.linalg.norm(a,axis=-1,keepdims=True)
    a_normalized = a / norms
    return a_normalized

def back_hemi_octa_map(a):
    '''
    a = [batch,2]
    return = [batch,3]
    '''
    p = (a - 0.5)*2.0
    px = p[:,0]#px=[batch]
    py = p[:,1]#py=[batch]
    resultx = np.expand_dims((px+py)*0.5,axis=-1)#[batch,1]
    resulty = np.expand_dims((py-px)*0.5,axis=-1)#[batch,1]
    resultz = 1.0-np.abs(resultx)-np.abs(resulty)#[batch,1]
    result = np.concatenate([resultx,resulty,resultz],axis=-1)#[batch,3]

    return normalize_vector_matrix(result)

def build_frame_f_z(n,theta,without_theta = False):
    '''
    n = [batch,3]
    theta = [batch,1]
    return =t[batch,3] b[batch,3]
    '''
    batch_size = n.shape[0]
    nz = n[:,[2]]#[batch,1]
    constant_001 = np.expand_dims(np.array([0,0,1],np.float32),0).repeat(batch_size,axis=0)#[batch,3]
    constant_100 = np.expand_dims(np.array([1,0,0],np.float32),0).repeat(batch_size,axis=0)#[batch,3]
    nz_notequal_1 = np.abs(nz-1.0)>1e-4
    nz_notequal_m1 = np.abs(nz+1.0)>1e-4
    judgements = np.tile(np.logical_and(nz_notequal_1,nz_notequal_m1),[1,3])#[batch,3]
    
    t = np.where(judgements,constant_001,constant_100)#[batch,3]
    t = np.cross(t,n)
    t = normalize_vector_matrix(t)#[batch,3]
    b = np.cross(n,t)#[batch,3]
    if without_theta:#no theta,no rotation
        return t,b
    t = t*np.cos(theta)+b*np.sin(theta)

    b = normalize_vector_matrix(np.cross(n,t))#[batch,3]
    return t,b