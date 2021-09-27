import numpy as np

def normalized(a):
    norm = np.sqrt(np.sum(np.square(a), axis=-1,keepdims=True))
    a = a / (norm+1e-6)

    return a

def full_octa_map(dir):
    '''
    dir = [batch,3]
    return=[batch,2]
    '''
    dirz = np.expand_dims(dir[:,2],1)#[batch,1]
    p = dir/np.sum(np.abs(dir),axis=1,keepdims=True)
    px,py,pz = np.split(p,3,axis=1)
    x = px
    y = py

    judgements1 = np.greater_equal(px,0.0)
    x_12 = np.where(judgements1,1.0-py,-1.0+py)#[batch,1]
    y_12 = np.where(judgements1,1.0-px,1.0+px)

    judgements2 = np.less_equal(px,0.0)
    x_34 = np.where(judgements2,-1.0-py,1.0+py)
    y_34 = np.where(judgements2,-1.0-px,-1.0+px)

    judgements3 = np.greater_equal(py,0.0)
    x_1234 = np.where(judgements3,x_12,x_34)#[batch,1]
    y_1234 = np.where(judgements3,y_12,y_34)#[batch,1]

    judgements4 = np.less(dirz,0.0)
    x = np.where(judgements4,x_1234,x)
    y = np.where(judgements4,y_1234,y)

    return (np.concatenate([x,y],axis=1)+1.0)*0.5

def back_full_octa_map(a):
    '''
    a = [batch,2]
    return = [batch,3]
    '''
    p = a*2.0-1.0
    px = np.expand_dims(p[:,0],axis=-1)
    py = np.expand_dims(p[:,1],axis=-1)#px=[batch,1] py=[batch,1]
    x = px #px=[batch,1]
    y = py #px=[batch,1]
    abs_px_abs_py = np.abs(px)+np.abs(py)
    
    judgements2 = np.greater_equal(py,0.0)
    judgements3 = np.greater_equal(px,0.0)
    judgements4 = np.less_equal(px,0.0)

    x_12 = np.where(judgements3,1.0-py,-1.0+py)
    y_12 = np.where(judgements3,1.0-px,1.0+px)

    x_34 = np.where(judgements4,-1.0-py,1.0+py)
    y_34 = np.where(judgements4,-1.0-px,-1.0+px)

    x_1234 = np.where(judgements2,x_12,x_34)
    y_1234 = np.where(judgements2,y_12,y_34)

    
    judgements1 = np.greater(abs_px_abs_py,1)

    resultx = np.where(judgements1,x_1234,px)#[batch,1]
    resulty = np.where(judgements1,y_1234,py)#[batch,1]
    resultz = 1.0-np.abs(resultx)-np.abs(resulty)
    resultz = np.where(judgements1,-1.0*resultz,resultz)

    result = np.concatenate([resultx,resulty,resultz],axis=-1)#[batch,3]

    return normalized(result)


def orthogonalize(matrix):
    orthogonal_matrix = np.zeros(matrix.shape)
    x = normalized(matrix[:,0])
    
    y = normalized(matrix[:,1])
    z = normalized(matrix[:,2])

    x0 = x
    x1 = x

    z1 = normalized(np.cross(x,y))
    y1 = normalized(np.cross(z1,x1))

    z0 = z1
    y0 = y1
    y0 += y

    x1 = normalized(np.cross(y,z))
    z1 = normalized(np.cross(x1,y))

    x0 += x1
    z0 += z1
    z0 += z

    y1 = normalized(np.cross(z,x))
    x1 = normalized(np.cross(y1,z))

    y0 = normalized(y1+y0)
    x0 = normalized(x1+x0)
    z0 = normalized(np.cross(x0,y0))

    y0 = np.cross(z0,x0)

    x0 = np.squeeze(x0)
    y0 = np.squeeze(y0)
    z0 = np.squeeze(z0)
    
    orthogonal_matrix[:,0] = x0
    orthogonal_matrix[:,1] = y0
    orthogonal_matrix[:,2] = z0

    return orthogonal_matrix


def build_transform_matrix(motion,angular):
    pattern_num = angular[0].shape[0]

    transform_matrix_for_model = np.repeat(np.expand_dims(np.eye(4),axis=0),pattern_num,axis=0)

    alpha = angular[0]
    beta = angular[1]
    gamma = angular[2]

    transform_matrix_for_model[:,0,1] = -gamma
    transform_matrix_for_model[:,0,2] = beta
    transform_matrix_for_model[:,1,0] = gamma
    transform_matrix_for_model[:,1,2] = -alpha
    transform_matrix_for_model[:,2,0] = -beta
    transform_matrix_for_model[:,2,1] = alpha
    transform_matrix_for_model[:,:3,3] = motion

    transform_matrix_for_model[:,:3,:3] = orthogonalize(transform_matrix_for_model[:,:3,:3])
    

    return transform_matrix_for_model