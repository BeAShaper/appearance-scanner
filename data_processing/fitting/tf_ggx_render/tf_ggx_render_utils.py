import numpy as np
import os

def make_dir(path):
    if(os.path.exists(path) == False):
        os.makedirs(path)

def visualize_init(path="visualize_idxs_full.bin",lumitexel_size=24576):
    pf = open(path,"rb")
    print("initing idx....")
    global visualize_idxs
    visualize_idxs = np.fromfile(pf,dtype = np.int32).reshape([-1,2])
    if visualize_idxs.shape[0] != lumitexel_size :
        print("[VISUALIZE]:error dimension")
        exit()
    pf.close()
    print("done.")

def visualize_new(data,len = 64,scalerf=1.0):
    img = np.zeros([len * 3, len * 4],np.float32)
    for i in range(data.shape[0]):
        img[visualize_idxs[i][1]][visualize_idxs[i][0]] = data[i] * scalerf
    return img

def print_loss(loss_evaled, vector_evaled):
    pass
    # print("[OPTIMIZE LOG]",loss_evaled, vector_evaled[0])

def print_step(vector_evaled):
    global step_counter
    step_counter += 1
    # print("[STEP LOG]",vector_evaled)

def init_step():
    global step_counter
    step_counter = 0

def report_step():
    # print("[STEP COST]:",step_counter)
    return step_counter
