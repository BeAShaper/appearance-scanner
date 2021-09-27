import numpy as np
import sys
sys.path.append("../utils/")
from dir_folder_and_files import make_dir,del_file
from codex_numpy_utils import build_frame_f_z,back_hemi_octa_map
from subprocess import Popen
import cv2

def load_cam_pos(config_file_dir):
    cam_pos = np.fromfile(config_file_dir+"cam_pos.bin",np.float32)
    assert cam_pos.shape[0] == 3
    mat_model = np.fromfile(config_file_dir+"mat_model.bin",np.float32).reshape([4,4])
    mat_for_normal = np.fromfile(config_file_dir+"mat_for_normal.bin",np.float32).reshape([4,4])

    return cam_pos,mat_model,mat_for_normal

if __name__ == "__main__":
    data_root = sys.argv[1]
    img_height = int(sys.argv[2])
    img_width = int(sys.argv[3])
    machine_config_dir = sys.argv[4]
    is_rgb = int(sys.argv[5]) == 1
    file_name=sys.argv[6]
    config_file = sys.argv[7]
    is_for_server = sys.argv[8]

    ######################
    ######old
    #####################
    # if is_rgb:
    #     raw_data = np.fromfile(data_root+file_name,np.float32).reshape([-1,11])
    # else:
    #     raw_data = np.fromfile(data_root+file_name,np.float32).reshape([-1,7])


    # with open(data_root+"cam00_index_nocc.bin","rb") as f:
    #     num = np.fromfile(f,np.int32,1)
    #     idxs = np.fromfile(f,np.int32).reshape([-1,2])
    
    # if(raw_data.shape[0] != idxs.shape[0]):
    #     print(raw_data.shape[0])
    #     print(idxs.shape[0])
    #     print("[ERROR]error num")
    #     exit(0)
    
    # cam_pos,_,_ = load_cam_pos(machine_config_dir)#[3]

    # positions = np.zeros([raw_data.shape[0],3],np.float32)#[batch,3]
    # cam_pos_broaded = np.tile(np.expand_dims(cam_pos,axis=0),[raw_data.shape[0],1])#[batch,3]
    # view_dirs = cam_pos_broaded-positions

    # frame_t,frame_b = build_frame_f_z(view_dirs,None,without_theta=True)#[batch,3]
    # frame_n = view_dirs #[batch,3]

    # ns = raw_data[:,:2]#[batch,2]
    # ts = np.expand_dims(raw_data[:,2],axis=-1)#[batch,1]
    # axay = raw_data[:,3:5]#[batch,2]
    # axay_min = 0.006
    # axay_max = 0.503
    # axay = (axay-axay_min)/(axay_max-axay_min)
    # axay = np.concatenate([axay,np.zeros([raw_data.shape[0],1],np.float32)],axis=-1)#[batch,3]

    # ns_local_3d = back_hemi_octa_map(ns)#[batch,3]
    # global_n = ns_local_3d[:,[0]]*frame_t+\
    #             ns_local_3d[:,[1]]*frame_b+\
    #                 ns_local_3d[:,[2]]*frame_n#[batch,3]

    # global_t,global_b = build_frame_f_z(global_n,ts)#[batch,3]

    # img_n       = np.zeros([img_height,img_width,3],np.float32)
    # img_t       = np.zeros([img_height,img_width,3],np.float32)
    # img_axay    = np.zeros([img_height,img_width,3],np.float32)
    # img_pd      = np.zeros([img_height,img_width,3],np.float32)
    # img_ps      = np.zeros([img_height,img_width,3],np.float32)

    # for i in range(idxs.shape[0]):
    #     img_n[idxs[i][1],idxs[i][0]] = ns_local_3d[i]*0.5+0.5
    #     img_t[idxs[i][1],idxs[i][0]] = global_t[i]*0.5+0.5
    #     img_axay[idxs[i][1],idxs[i][0]] = axay[i]
    #     if is_rgb:
    #         img_pd[idxs[i][1],idxs[i][0]] = raw_data[i][5:8]
    #         img_ps[idxs[i][1],idxs[i][0]] = raw_data[i][8:]
    #     else:
    #         img_pd[idxs[i][1],idxs[i][0]] = raw_data[i][5]
    #         img_ps[idxs[i][1],idxs[i][0]] = raw_data[i][6]

    # img_root = data_root+"imgs/"
    # make_dir(img_root)

    # img_n       = img_n[:,:,::-1]   
    # img_t       = img_t[:,:,::-1]   
    # img_axay    = img_axay[:,:,::-1]
    # img_pd      = img_pd[:,:,::-1]  
    # img_ps      = img_ps[:,:,::-1]
    # cv2.imwrite(img_root+"normal.png",img_n*255)
    # cv2.imwrite(img_root+"tangent.png",img_t*255)
    # cv2.imwrite(img_root+"axay.png",img_axay*255)
    # cv2.imwrite(img_root+"pd.png",img_pd*15)
    # cv2.imwrite(img_root+"ps.png",img_ps*15)


    ##########################################
    ######new
    ##########################################
    # theProcess = Popen(
    #     [
    #         "python",
    #         "n_local_to_global.py",
    #         data_root,
    #         config_file,
    #         is_for_server
    #     ]
    # )
    # exit_code = theProcess.wait()
    # print("n local to global:",exit_code)

    root = "images/data_for_server/" if is_for_server == "true" else "images/data_for_local/"
    params = np.fromfile(data_root+root+"gathered_all/params_gathered_gloabal_normal.bin",np.float32).reshape([-1,12])

    # ###n 
    # param_name = "n_global_single_view"
    # ns = params[:,:3]*0.5+0.5
    # ns.astype(np.float32).tofile(data_root+root+"gathered_all/{}.bin".format(param_name))

    # theProcess = Popen(
    #     [
    #         "../exe/float_to_exr_with_output_name.exe",
    #         data_root+root+"gathered_all/",
    #         "{}.bin".format(param_name),
    #         "2048",
    #         "2448",
    #         "{}.exr".format(param_name)
    #     ]
    # )
    # theProcess.wait()

    # ###pd
    # param_name = "pd"
    # ns = params[:,6:9]
    # ns.astype(np.float32).tofile(data_root+root+"gathered_all/{}.bin".format(param_name))

    # theProcess = Popen(
    #     [
    #         "../exe/float_to_exr_with_output_name.exe",
    #         data_root+root+"gathered_all/",
    #         "{}.bin".format(param_name),
    #         "2048",
    #         "2448",
    #         "{}.exr".format(param_name)
    #     ]
    # )
    # theProcess.wait()

    ###ps
    param_name = "ps"
    ns = params[:,9:]
    ns.astype(np.float32).tofile(data_root+root+"gathered_all/{}.bin".format(param_name))

    theProcess = Popen(
        [
            "../exe/float_to_exr_with_output_name.exe",
            data_root+root+"gathered_all/",
            "{}.bin".format(param_name),
            "2048",
            "2448",
            "{}.exr".format(param_name)
        ]
    )
    theProcess.wait()