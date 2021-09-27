import cv2
import numpy as np
import argparse
import sys
import shutil

# from rotate_normal_back import rotate_normal_global_siga19,rotate_centre_normal_to_global

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("shot_root")
    parser.add_argument("feature_task")
    parser.add_argument("material_task")
    parser.add_argument("udt_folder_name")
    parser.add_argument("texture_folder_name")
    parser.add_argument("texture_resolution",type=int)
    parser.add_argument("sub_folder_name")

    args = parser.parse_args()

    tex_folder_root = args.shot_root+args.feature_task+"/"+args.udt_folder_name+"/"+args.texture_folder_name+"/"
    material_root = args.shot_root+args.material_task+"/images_{}/data_for_server/gathered_all/".format(args.sub_folder_name)
    
    tex_uvs = np.fromfile(tex_folder_root+"texturemap_uv.bin",np.int32).reshape([-1,2])
    

    sub_folder_name = args.sub_folder_name
    fitted_spec_params = np.fromfile(material_root+"params_gathered_gloabal_normal.bin",np.float32).reshape([-1,12])

    data_file_names = [
        "params_gathered_gloabal_normal.bin",
        "nn_normals.bin",
        "params_gathered.bin",
        "gathered_global_tangent.bin"
    ]
    for data_file_name in data_file_names:
        shutil.copyfile(material_root+data_file_name,tex_folder_root+sub_folder_name+data_file_name)

    # infer_img = cv2.imread(tex_folder_root+"valid_check.exr",cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    # texture_map_uv_map = {(tex_uvs[i][0],tex_uvs[i][1]):i for i in range(tex_uvs.shape[0]) }
    # test_id_list = []
    
    # for y in range(infer_img.shape[0]):
    #     for x in range(infer_img.shape[1]):
    #         if infer_img[y,x,0] == 1.0:
    #             test_id_list.append(texture_map_uv_map[(x,y)])
    # print(len(test_id_list))
    
    # fitting_sequence = np.array(test_id_list)
    fitting_sequence = np.arange(fitted_spec_params.shape[0])

    # fitted normal local
    normal = fitted_spec_params[:,:3]
    img = np.zeros([args.texture_resolution,args.texture_resolution,3],np.float32)
    img[tex_uvs[fitting_sequence,1],tex_uvs[fitting_sequence,0]] = normal*0.5+0.5
    img = img[:,:,::-1]
    cv2.imwrite(tex_folder_root+sub_folder_name+"/normal_fitted_centre.exr",img)

    #pd
    pd_data = fitted_spec_params[:,6:9]
    img = np.zeros([args.texture_resolution,args.texture_resolution,3],np.float32)
    img[tex_uvs[fitting_sequence,1],tex_uvs[fitting_sequence,0]] = pd_data
    img = img[:,:,::-1]
    cv2.imwrite(tex_folder_root+sub_folder_name+"/pd_fitted.exr",img)
    
    #ps
    ps_data = fitted_spec_params[:,9:12]
    img = np.zeros([args.texture_resolution,args.texture_resolution,3],np.float32)
    img[tex_uvs[fitting_sequence,1],tex_uvs[fitting_sequence,0]] = ps_data
    img = img[:,:,::-1]
    cv2.imwrite(tex_folder_root+sub_folder_name+"/ps_fitted.exr",img)

    #axay
    axay_data = fitted_spec_params[:,4:6]
    axay_data = np.concatenate([axay_data,np.zeros([axay_data.shape[0],1],np.float32)],axis=-1)
    img = np.zeros([args.texture_resolution,args.texture_resolution,3],np.float32)
    img[tex_uvs[fitting_sequence,1],tex_uvs[fitting_sequence,0]] = axay_data
    img = img[:,:,::-1]
    cv2.imwrite(tex_folder_root+sub_folder_name+"/axay_fitted.exr",img)
    
    # normal fitted global
    normal = np.fromfile(tex_folder_root+sub_folder_name+"/nn_normal_global.bin",np.float32).reshape([-1,3])
    img = np.zeros([args.texture_resolution,args.texture_resolution,3],np.float32)
    img[tex_uvs[fitting_sequence,1],tex_uvs[fitting_sequence,0]] = normal*0.5+0.5
    img = img[:,:,::-1]
    cv2.imwrite(tex_folder_root+sub_folder_name+"/nn_normals_global.exr",img)

    # normal fitted global
    normal = np.fromfile(tex_folder_root+sub_folder_name+"/normal_fitted_global.bin",np.float32).reshape([-1,3])
    img = np.zeros([args.texture_resolution,args.texture_resolution,3],np.float32)
    img[tex_uvs[fitting_sequence,1],tex_uvs[fitting_sequence,0]] = normal*0.5+0.5
    img = img[:,:,::-1]
    cv2.imwrite(tex_folder_root+sub_folder_name+"/normal_fitted_global.exr",img)

    # normal fitted global
    tangent = np.fromfile(tex_folder_root+sub_folder_name+"/tangent_fitted_global.bin",np.float32).reshape([-1,3])
    img = np.zeros([args.texture_resolution,args.texture_resolution,3],np.float32)
    img[tex_uvs[fitting_sequence,1],tex_uvs[fitting_sequence,0]] = tangent*0.5+0.5
    img = img[:,:,::-1]
    cv2.imwrite(tex_folder_root+sub_folder_name+"/tangent_fitted_global.exr",img)

    dot_res = np.sum(normal*tangent,axis=-1)
    print(dot_res)
    # [ 5.9604645e-08  2.9802322e-08 -3.5762787e-07 ... -2.6822090e-07
    #  -3.0547380e-07  1.8626451e-07]


    # # axay rejected
    # axay = np.fromfile(tex_folder_root+"camera_sigmoid_geo"+"/axay.bin",np.float32).reshape([-1,2])
    # axay = np.concatenate([axay,np.zeros([axay.shape[0],1],np.float32)],axis=-1)
    # img = np.zeros([args.texture_resolution,args.texture_resolution,3],np.float32)
    # img[tex_uvs[:,1],tex_uvs[:,0]] = axay
    # img = img[:,:,::-1]
    # cv2.imwrite(tex_folder_root+sub_folder_name+"/axay_rejected.exr",img)

    # #tangent global
    # tangent = np.fromfile(tex_folder_root+sub_folder_name+"tangent_fitted_local.bin",np.float32).reshape([-1,3])
    # img = np.zeros([args.texture_resolution,args.texture_resolution,3],np.float32)
    # img[tex_uvs[:,1],tex_uvs[:,0]] = tangent*0.5+0.5
    # img = img[:,:,::-1]
    # cv2.imwrite(tex_folder_root+sub_folder_name+"tangent_fitted_local.exr",img)

    # #tamgemt global
    # tamgemt = np.fromfile(tex_folder_root+sub_folder_name+"shading_tangent_global.bin",np.float32).reshape([-1,3])
    # img = np.zeros([args.texture_resolution,args.texture_resolution,3],np.float32)
    # img[tex_uvs[:,1],tex_uvs[:,0]] = tamgemt*0.5+0.5
    # img = img[:,:,::-1]
    # cv2.imwrite(tex_folder_root+sub_folder_name+"global_shading_tangent.exr",img)

    # global_shading_normal = np.fromfile(tex_folder_root+"normals_geo_fulllocalview.bin",np.float32).reshape([-1,60,3])[:,0,:]
    # img = np.zeros([args.texture_resolution,args.texture_resolution,3],np.float32)
    # img[tex_uvs[:,1],tex_uvs[:,0]] = global_shading_normal*0.5+0.5
    # img = img[:,:,::-1]
    # cv2.imwrite(tex_folder_root+sub_folder_name+"/global_shading_normal.exr",img)

    # normal = geometry_normals
    # img = np.zeros([args.texture_resolution,args.texture_resolution,3],np.float32)
    # img[tex_uvs[:,1],tex_uvs[:,0]] = normal*0.5+0.5
    # img = img[:,:,::-1]
    # cv2.imwrite(tex_folder_root+sub_folder_name+"normal_geometry.exr",img)

    # # fitted normal local
    # normal = np.fromfile(tex_folder_root+sub_folder_name+"normal_fitted_local.bin",np.float32).reshape([-1,3])
    # img = np.zeros([args.texture_resolution,args.texture_resolution,3],np.float32)
    # img[tex_uvs[:,1],tex_uvs[:,0]] = normal*0.5+0.5
    # img = img[:,:,::-1]
    # cv2.imwrite(tex_folder_root+sub_folder_name+"normal_fitted_local.exr",img)

    # # nn normal local
    # normal = np.fromfile(tex_folder_root+sub_folder_name+"nn_normal_local.bin",np.float32).reshape([-1,3])
    # img = np.zeros([args.texture_resolution,args.texture_resolution,3],np.float32)
    # img[tex_uvs[:,1],tex_uvs[:,0]] = normal*0.5+0.5
    # img = img[:,:,::-1]
    # cv2.imwrite(tex_folder_root+sub_folder_name+"nn_normal_local.exr",img)