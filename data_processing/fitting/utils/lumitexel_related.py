import numpy as np

def plane_visualize_init(utils_path,lumitexel_size=4096,shrink_step=1):
    global shrink_size
    shrink_size = shrink_step
    path = utils_path+"visualize_configs/plane_visualize_idx_new.bin"
    pf = open(path,"rb")
    print("initing idx....")
    global visualize_idxs
    visualize_idxs = np.fromfile(pf,dtype = np.int32).reshape([-1,2])
    print(lumitexel_size)
    print(visualize_idxs.shape[0])
    if visualize_idxs.shape[0] != lumitexel_size :
        print("[VISUALIZE]:error dimension here")
        exit()
    pf.close()
    print("done.")

# def plane_visualize_init_big(utils_path,lumitexel_size=4096):
#     path = utils_path+"visualize_configs/plane_visualize_idx_big_slice_1x1_96.bin"
#     pf = open(path,"rb")
#     print("initing idx....")
#     global visualize_idxs_big
#     visualize_idxs_big = np.fromfile(pf,dtype = np.int32).reshape([-1,2])
#     print(lumitexel_size)
#     print(visualize_idxs_big.shape[0])
#     if visualize_idxs_big.shape[0] != lumitexel_size :
#         print("[VISUALIZE]:error dimension in init big")
#         exit()
#     pf.close()
#     print("done.")

def plane_visualize_big_slice_init(utils_path,width=128,height=64,ratio=4):
    path = utils_path+"visualize_configs/plane_visualize_idx_big_slice_{}x{}_{}_new.bin".format(ratio,ratio,width)
    pf = open(path,"rb")
    print("initing idx....")
    if not 'visualize_cube_slice_idxs_collector' in globals():
        global visualize_cube_slice_idxs_collector
        visualize_cube_slice_idxs_collector = {}
    visualize_cube_slice_idxs = np.fromfile(pf,dtype = np.int32).reshape([-1,2])
    if visualize_cube_slice_idxs.shape[0] != width//ratio * height//ratio :
        print("[VISUALIZE]:error dimension in init")
        # print(visualize_cube_slice_idxs.shape[0])
        # print(width)
        # print(height)
        exit()
    pf.close()
    visualize_cube_slice_idxs_collector["big_{}_{}".format(width,ratio)] = visualize_cube_slice_idxs
    print("done.")

def get_big_visualize_idxs(width,ratio):
    global visualize_cube_slice_idxs_collector
    return visualize_cube_slice_idxs_collector["big_{}_{}".format(width,ratio)]

# def plane_visualize_slice_init(utils_path,width=64,height=32,ratio=4):
#     path = utils_path+"visualize_configs/plane_visualize_idx_slice_{}x{}.bin".format(ratio,ratio)
#     pf = open(path,"rb")
#     print("initing idx....")
#     if not 'visualize_cube_slice_idxs_collector' in globals():
#         global visualize_cube_slice_idxs_collector
#         visualize_cube_slice_idxs_collector = {}
#     visualize_cube_slice_idxs = np.fromfile(pf,dtype = np.int32).reshape([-1,2])
#     if visualize_cube_slice_idxs.shape[0] != width//ratio * height//ratio :
#         print("[VISUALIZE]:error dimension")
#         exit()
#     pf.close()
#     visualize_cube_slice_idxs_collector["normal_{}".format(ratio)] = visualize_cube_slice_idxs
#     print("done.")

def plane_visualize_new_rgb(data,len = 16,scalerf=1.0):
    global shrink_size
    len = len//shrink_size
    img = np.zeros([len * 1, len * 2,3], data.dtype)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            img[visualize_idxs[j][0]][visualize_idxs[j][1]][i] = data[i][j] * scalerf
    return img


def plane_visualize_new_slice_rgb(data,width=128,height=64,ratio=4,scalerf=1.0):
    if data.shape[1] != width//ratio * height//ratio :
        print("[VISUALIZE]:error dimension in new")
        exit()
    if width > 32:
        idxs = visualize_cube_slice_idxs_collector["big_{}_{}".format(width,ratio)]
    else:
        idxs = visualize_cube_slice_idxs_collector["normal_{}".format(ratio)]

    img = np.zeros([height//ratio, width//ratio,3], data.dtype)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            img[idxs[j][0]][idxs[j][1]][i] = data[i][j] * scalerf
    return img # expand_slice(img,ratio)

def plane_visualize_new(data,len = 16, scalerf=1.0):
    global shrink_size
    len = len//shrink_size
    img = np.zeros([len * 1, len * 2],data.dtype)
    for i in range(data.shape[0]):
        img[visualize_idxs[i][0]][visualize_idxs[i][1]] = data[i] * scalerf
    return img


def visualize_init(utils_path,shrink_step=1,lumitexel_size=24576):
    global shrink_size
    shrink_size = shrink_step
    path = utils_path+"visualize_configs/visualize_idxs_{}x{}.bin".format(shrink_step,shrink_step)
    pf = open(path,"rb")
    print("initing idx....")
    global visualize_idxs
    visualize_idxs = np.fromfile(pf,dtype = np.int32).reshape([-1,2])
    if visualize_idxs.shape[0] != lumitexel_size :
        print("[VISUALIZE]:error dimension here")
        exit()
    pf.close()
    print("done.")

def visualize_new(data,len = 64,scalerf=1.0):
    global shrink_size
    len = len//shrink_size
    img = np.zeros([len * 3, len * 4],data.dtype)#*192
    for i in range(data.shape[0]):
        img[visualize_idxs[i][1]][visualize_idxs[i][0]] = data[i] * scalerf
    return img

# def visualize_cube_slice_init(utils_path,slice_height=16,slice_width=32):
#     print(utils_path)
#     path = utils_path+"visualize_configs/visualize_config_torch_plane_{}x{}.bin".format(slice_height,slice_width)
#     pf = open(path,"rb")
#     print("initing idx....")
#     if not 'visualize_cube_slice_idxs_collector' in globals():
#         global visualize_cube_slice_idxs_collector
#         visualize_cube_slice_idxs_collector = {}
#     visualize_cube_slice_idxs = np.fromfile(pf,dtype = np.int32).reshape([-1,2])
    
#     # if visualize_cube_slice_idxs.shape[0] != sample_num*sample_num*6 :
#     #     print("[VISUALIZE]:error dimension")
#     #     exit()
#     pf.close()
#     visualize_cube_slice_idxs_collector["{}x{}".format(slice_height,slice_width)] = visualize_cube_slice_idxs[:,[1,0]]
#     print("done.")

def visualize_cube_slice_init(utils_path,sample_num=32):
    path = utils_path+"visualize_configs/visualize_idxs_cube_slice_{}.bin".format(sample_num)
    pf = open(path,"rb")
    print("initing idx....")
    if not 'visualize_cube_slice_idxs_collector' in globals():
        global visualize_cube_slice_idxs_collector
        visualize_cube_slice_idxs_collector = {}
    visualize_cube_slice_idxs = np.fromfile(pf,dtype = np.int32).reshape([-1,2])
    if visualize_cube_slice_idxs.shape[0] != sample_num*sample_num*6 :
        print("[VISUALIZE]:error dimension")
        exit()
    pf.close()
    visualize_cube_slice_idxs_collector[sample_num] = visualize_cube_slice_idxs
    print("done.")

def expand_slice(img, ratio):
    if len(img.shape) == 3:
        height,width,channel = img.shape
    else:
        height,width = img.shape
        channel = 1
    height *= ratio
    width *= ratio
    new_img = np.zeros([height,width,channel],img.dtype)
    for i in range(channel):
        for j in range(height):
            for k in range(width):
                new_img[j][k][i] = img[j//ratio][k//ratio][i]
    return new_img

def blur_lumi_init(utils_path,step):
    global reorder_idx
    global invert_idx
    reorder_idx = np.fromfile(utils_path+"shrink_order/reorder_lumi_to_{}x{}.bin".format(step,step),np.int32).reshape([-1])
    invert_idx = np.fromfile(utils_path+"shrink_order/reorder_lumi_to_{}x{}_revert.bin".format(step,step),np.int32).reshape([-1])

def blur_brdf_not_slice_init(utils_path,step):
    global reorder_brdf_not_slice_idx
    global invert_brdf_not_slice_idx
    reorder_brdf_not_slice_idx = np.fromfile(utils_path+"shrink_order/reorder_brdf_not_slice_to_{}x{}.bin".format(step,step),np.int32).reshape([-1])
    invert_brdf_not_slice_idx = np.fromfile(utils_path+"shrink_order/reorder_brdf_not_slice_to_{}x{}_revert.bin".format(step,step),np.int32).reshape([-1])

def get_blur_method(method):
    if method == "mean":
        return np.mean
    elif method == "median":
        return np.median
    else:
        print("[get blur lumi] unsupport method!")
        exit()

def blur_lumi(data,step,method="mean",with_invert = True):
    '''
    input can be tensor
    '''
    origin_shape = data.shape
    data = np.reshape(data,[-1,24576])
    data = data[:,reorder_idx]
    blur_method = get_blur_method(method)
    if not with_invert:
        return blur_method(data.reshape([-1,step*step]),axis=1,keepdims=True).reshape([-1,24576//step//step])
    blured_lumi = np.repeat(blur_method(data.reshape([-1,step*step]),axis=1,keepdims=True),step*step,axis=1).reshape([-1,24576])
    blured_lumi = blured_lumi[:,invert_idx]
    return blured_lumi.reshape(origin_shape)

def blur_brdf_not_slice(data,step,method="mean",with_invert = True):
    '''
    input can be tensor
    '''
    origin_shape = data.shape
    data = np.reshape(data,[-1,6144])
    data = data[:,reorder_brdf_not_slice_idx]
    blur_method = get_blur_method(method)
    if not with_invert:
        return blur_method(data.reshape([-1,step*step]),axis=1,keepdims=True).reshape([-1,6144//step//step])
    blured_lumi = np.repeat(blur_method(data.reshape([-1,step*step]),axis=1,keepdims=True),step*step,axis=1).reshape([-1,6144])
    blured_lumi = blured_lumi[:,invert_brdf_not_slice_idx]
    return blured_lumi.reshape(origin_shape)

def reorder_to_block(data,step):
    data = np.reshape(data,[-1,24576])
    data = data[:,reorder_idx]
    return data

def get_reorder_idx():
    return reorder_idx

def get_brdf_not_slice_reorder_idx():
    return reorder_brdf_not_slice_idx


def visualize_cube_slice(data,sample_num=32,scalerf=1.0):
    img = np.zeros([sample_num * 3, sample_num * 4],np.float32)#*192.0
    visualize_cube_slice_idxs = visualize_cube_slice_idxs_collector[sample_num]
    for i in range(data.shape[0]):
        img[visualize_cube_slice_idxs[i][1]][visualize_cube_slice_idxs[i][0]] = data[i] * scalerf
    return img

def visualize_plane(data,slice_height=16,slice_width=32,scalerf=1.0):
    
    img = np.zeros([slice_height,slice_width],np.float32)
    visualize_cube_slice_idxs = visualize_cube_slice_idxs_collector["{}x{}".format(slice_height,slice_width)]
    
    for i in range(data.shape[0]):
        img[visualize_cube_slice_idxs[i][1]][visualize_cube_slice_idxs[i][0]] = data[i] * scalerf
    return img

def unvisualize_new(img,len=64):
    data = np.zeros([24576],np.float32)
    for i in range(data.shape[0]):
        data[i] = img[visualize_idxs[i][1]][visualize_idxs[i][0]]
    return data

def get_visualize_idxs():
    global visualize_idxs
    return visualize_idxs

def get_cube_slice_visualize_idxs(sample_num=32):
    return visualize_cube_slice_idxs_collector[sample_num]

def init_sub_to_full_lumitexel(utils_path):
    global idx_map
    idxpath=utils_path+"config/kkz/idx_map_8x8.txt"
    idx_map = {}
    print("[EXPANDER] init expander with:",idxpath)
    with open(idxpath,"r") as f:
        maps = [line for line in f.readlines() if line.strip()]
        # maps = f.read().split('\n')
        maps = [pair.split() for pair in maps]
        for pairs in maps:
            idx_map[int(pairs[0])] = int(pairs[1])

def get_sub_to_full_idxmap():
    global idx_map
    return idx_map

def sub_to_full_lumitexel(origin,theMap=None,lumitexel_size=24576):
    if theMap == None:
        global idx_map
        theMap = idx_map     
    full = np.zeros(lumitexel_size,np.float32)
    for i in range(origin.shape[0]):
        full[theMap[i]] = origin[i]
    return full

def shrink(img_origin,step,method="max"):

    if method=="max":
        f = shrink_withmax
    elif method=="mean":
        f = shrink_withmean
    elif method == "sum":
        f = shrink_withsum
    else:
        print("[ERROR]error when shrink. unknown method:",method)

    block_size = 64
    block_size_shrinked = int(block_size/step)
    res = np.zeros([int(img_origin.shape[0]/step),int(img_origin.shape[1]/step)],np.float32)

    line_num = 3
    col_num = 2
    img1 = img_origin[block_size*(line_num-1):block_size*line_num,block_size*(col_num-1):block_size*(col_num)].copy()
    img1 = np.flipud(img1)
    img1 = f(img1,step)#[::step,::step]
    img1 = np.flipud(img1)
    res[block_size_shrinked*(line_num-1):block_size_shrinked*line_num,block_size_shrinked*(col_num-1):block_size_shrinked*(col_num)] = img1

    line_num = 2
    col_num = 1
    img2 = img_origin[block_size*(line_num-1):block_size*line_num,block_size*(col_num-1):block_size*(col_num)].copy()
    img2 = f(img2,step)#[::step,::step]
    res[block_size_shrinked*(line_num-1):block_size_shrinked*line_num,block_size_shrinked*(col_num-1):block_size_shrinked*(col_num)] = img2

    line_num = 1
    col_num = 2
    img3 = img_origin[block_size*(line_num-1):block_size*line_num,block_size*(col_num-1):block_size*(col_num)]
    img3 = np.fliplr(img3)
    img3 = f(img3,step)#[::step,::step]
    img3 = np.fliplr(img3)
    res[block_size_shrinked*(line_num-1):block_size_shrinked*line_num,block_size_shrinked*(col_num-1):block_size_shrinked*(col_num)] = img3

    line_num = 2
    col_num = 4
    img4 = img_origin[block_size*(line_num-1):block_size*line_num,block_size*(col_num-1):block_size*(col_num)]
    img4 = np.fliplr(img4)
    img4 = np.flipud(img4)
    img4 = f(img4,step)#[::step,::step]
    img4 = np.flipud(img4)
    img4 = np.fliplr(img4)
    res[block_size_shrinked*(line_num-1):block_size_shrinked*line_num,block_size_shrinked*(col_num-1):block_size_shrinked*(col_num)] = img4

    line_num = 2
    col_num = 3
    img5 = img_origin[block_size*(line_num-1):block_size*line_num,block_size*(col_num-1):block_size*(col_num)]
    img5 = np.fliplr(img5)
    img5 = np.flipud(img5)
    img5 = f(img5,step)#[::step,::step]
    img5 = np.flipud(img5)
    img5 = np.fliplr(img5)
    res[block_size_shrinked*(line_num-1):block_size_shrinked*line_num,block_size_shrinked*(col_num-1):block_size_shrinked*(col_num)] = img5

    line_num = 2
    col_num = 2
    img6 = img_origin[block_size*(line_num-1):block_size*line_num,block_size*(col_num-1):block_size*(col_num)]
    img6 = np.flipud(img6)
    img6 = f(img6,step)#[::step,::step]
    img6 = np.flipud(img6)
    res[block_size_shrinked*(line_num-1):block_size_shrinked*line_num,block_size_shrinked*(col_num-1):block_size_shrinked*(col_num)] = img6

    return res

def shrink_withmean(img_origin,step):
    M, N = img_origin.shape
    K = step
    L = step

    MK = M // K
    NL = N // L
    res = img_origin[:MK*K, :NL*L].reshape(MK, K, NL, L).mean(axis=(1, 3))
    return res

def shrink_withsum(img_origin,step):
    M, N = img_origin.shape
    K = step
    L = step

    MK = M // K
    NL = N // L
    res = img_origin[:MK*K, :NL*L].reshape(MK, K, NL, L).sum(axis=(1, 3))
    return res

def shrink_withmax(img_origin,step):
    M, N = img_origin.shape
    K = step
    L = step

    MK = M // K
    NL = N // L
    res = img_origin[:MK*K, :NL*L].reshape(MK, K, NL, L).max(axis=(1, 3))
    return res

def expand_withcopy(img_origin,step):
    img_new = img_origin.repeat(step,axis=0).repeat(step,axis=1)
    res = img_new
    # print(res)
    res = img_new/(step*step)
    return res
def expand_withcopy_only(img_origin,step):
    img_new = img_origin.repeat(step,axis=0).repeat(step,axis=1)
    res = img_new
    return res

def expand_img(img_shrinked,step,method="copy",back_zero = False):
    if back_zero:  
        res = np.zeros([img_shrinked.shape[0]*step,img_shrinked.shape[1]*step],np.float32)
    else:
        res = np.ones([img_shrinked.shape[0]*step,img_shrinked.shape[1]*step],np.float32)#*192.0

    if method == "copy":
        f = expand_withcopy
    elif method =="copy_only":
        f = expand_withcopy_only
    else:
        print("[EXPAND]unknown method")
        exit()

    block_size = 64
    block_size_shrinked = int(block_size/step)
    if(block_size_shrinked*3 != img_shrinked.shape[0]):
        print("[EXPAND] error dimension")

    line_num = 3
    col_num = 2
    img1 = img_shrinked[block_size_shrinked*(line_num-1):block_size_shrinked*line_num,block_size_shrinked*(col_num-1):block_size_shrinked*(col_num)].copy()
    img1 = np.flipud(img1)
    img1 = f(img1,step)#[::step,::step]
    img1 = np.flipud(img1)
    res[block_size*(line_num-1):block_size*line_num,block_size*(col_num-1):block_size*(col_num)] = img1

    line_num = 2
    col_num = 1
    img2 = img_shrinked[block_size_shrinked*(line_num-1):block_size_shrinked*line_num,block_size_shrinked*(col_num-1):block_size_shrinked*(col_num)].copy()
    img2 = f(img2,step)#[::step,::step]
    res[block_size*(line_num-1):block_size*line_num,block_size*(col_num-1):block_size*(col_num)] = img2

    line_num = 1
    col_num = 2
    img3 = img_shrinked[block_size_shrinked*(line_num-1):block_size_shrinked*line_num,block_size_shrinked*(col_num-1):block_size_shrinked*(col_num)]
    img3 = np.fliplr(img3)
    img3 = f(img3,step)#[::step,::step]
    img3 = np.fliplr(img3)
    res[block_size*(line_num-1):block_size*line_num,block_size*(col_num-1):block_size*(col_num)] = img3

    line_num = 2
    col_num = 4
    img4 = img_shrinked[block_size_shrinked*(line_num-1):block_size_shrinked*line_num,block_size_shrinked*(col_num-1):block_size_shrinked*(col_num)]
    img4 = np.fliplr(img4)
    img4 = np.flipud(img4)
    img4 = f(img4,step)#[::step,::step]
    img4 = np.flipud(img4)
    img4 = np.fliplr(img4)
    res[block_size*(line_num-1):block_size*line_num,block_size*(col_num-1):block_size*(col_num)] = img4

    line_num = 2
    col_num = 3
    img5 = img_shrinked[block_size_shrinked*(line_num-1):block_size_shrinked*line_num,block_size_shrinked*(col_num-1):block_size_shrinked*(col_num)]
    img5 = np.fliplr(img5)
    img5 = np.flipud(img5)
    img5 = f(img5,step)#[::step,::step]
    img5 = np.flipud(img5)
    img5 = np.fliplr(img5)
    res[block_size*(line_num-1):block_size*line_num,block_size*(col_num-1):block_size*(col_num)] = img5

    line_num = 2
    col_num = 2
    img6 = img_shrinked[block_size_shrinked*(line_num-1):block_size_shrinked*line_num,block_size_shrinked*(col_num-1):block_size_shrinked*(col_num)]
    img6 = np.flipud(img6)
    img6 = f(img6,step)#[::step,::step]
    img6 = np.flipud(img6)
    res[block_size*(line_num-1):block_size*line_num,block_size*(col_num-1):block_size*(col_num)] = img6

    return res

def expand_lumitexel(short_lumi,step):
    full_lumitexel = sub_to_full_lumitexel(short_lumi,idx_map)
    full_lumitexel_img = visualize_new(full_lumitexel)
    full_lumitexel_img = shrink(full_lumitexel_img,step)
    full_lumitexel_img = expand_img(full_lumitexel_img,step)
    full_lumitexel = unvisualize_new(full_lumitexel_img)
    return full_lumitexel


def stretch_img(img,step):
    res_len = 24576//step//step
    block_len = 64//step
    res = np.zeros(res_len,np.float32)

    row_num = 1
    col_num = 1
    a = img[:block_len,(col_num-1)*block_len:col_num*block_len]
    a = a.reshape([-1])

    row_num = 2
    b = img[(row_num-1)*block_len:row_num*block_len]
    b = b.reshape([-1])

    row_num=3
    col_num=1
    c = img[(row_num-1)*block_len:row_num*block_len,(col_num-1)*block_len:col_num*block_len]
    c = c.reshape([-1])

    res = np.concatenate([a,b,c],axis=-1)
    if res.shape[0] != res_len:
        print(res.shape)
        print(res_len)
        print("[ERROR] Dimension error when stretching.")
        exit()
    return res

def get_cloest_light(light_poses,normal,obj_pos):
    '''
    light_poses = (light_num,3)
    normal = (3,)
    obj_pos = (3,)
    '''
    light_dir = light_poses - obj_pos.reshape([1,3])
    light_dir = light_dir/np.linalg.norm(light_dir,axis=1,keepdims=True)#[light_num,3]

    dot_res = np.sum(light_dir*normal,axis=1)
    max_idx = np.argmax(dot_res)
    return max_idx

default_axis_size = 9//2
default_axis_bold = 3//2

def draw_point(img,uv,color,axis_size = default_axis_size,axis_bold = default_axis_bold):
    img[uv[1]-axis_size:uv[1]+axis_size+1,uv[0]-axis_bold:uv[0]+axis_bold+1,:] = color
    img[uv[1]-axis_bold:uv[1]+axis_bold+1,uv[0]-axis_size:uv[0]+axis_size+1,:] = color

def draw_vector(light_poses,the_vector,obj_pos,color,sample_num,img):
    '''
    light_poses = (light_num,3)
    the_vector = (3,)
    obj_pos = (3,)
    color = (3,)
    img = [img_height,img_width,3]
    '''
    max_idx = get_cloest_light(light_poses,the_vector,obj_pos)
    if sample_num == -1:
        max_uv = visualize_idxs[max_idx]
    else:
        max_uv = visualize_cube_slice_idxs_collector[sample_num][max_idx]
    draw_point(img,max_uv,color)