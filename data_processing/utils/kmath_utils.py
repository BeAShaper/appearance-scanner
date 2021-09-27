import numpy as np

def rotation_axis_numpy(t, v,isRightHand=True):
    '''
    t = [batch,1]
    v = [batch,3]
    return = [batch,4,4]#rotate matrix
    caution! this is not a elegant implementation!
    '''
    if isRightHand:
        theta = t
    else:
        print("[RENDERER]Error rotate system doesn't support left hand logic!")
        exit()
    
    c = np.cos(theta)
    s = np.sin(theta)

    v_x = v[:,[0]]
    v_y = v[:,[1]]
    v_z = v[:,[2]]

    m_11 = c + (1-c)*v_x*v_x
    m_12 = (1 - c)*v_x*v_y - s*v_z
    m_13 = (1 - c)*v_x*v_z + s*v_y

    m_21 = (1 - c)*v_x*v_y + s*v_z
    m_22 = c + (1-c)*v_y*v_y
    m_23 = (1 - c)*v_y*v_z - s*v_x

    m_31 = (1 - c)*v_z*v_x - s*v_y
    m_32 = (1 - c)*v_z*v_y + s*v_x
    m_33 = c + (1-c)*v_z*v_z

    tmp_zeros = np.zeros([theta.shape[0],1],np.float32)
    tmp_ones = np.ones([theta.shape[0],1],np.float32)

    res = np.concatenate([
        m_11,m_12,m_13,tmp_zeros,
        m_21,m_22,m_23,tmp_zeros,
        m_31,m_32,m_33,tmp_zeros,
        tmp_zeros,tmp_zeros,tmp_zeros,tmp_ones
    ],axis=-1)

    res = np.reshape(res,[theta.shape[0],4,4])
    return res