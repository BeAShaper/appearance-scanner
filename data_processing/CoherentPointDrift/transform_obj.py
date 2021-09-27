import numpy as np
import open3d as o3d
import os
import cv2
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_obj_file")
    parser.add_argument("aln_file")
    parser.add_argument("aln_file_matlab")

    parser.add_argument("save_obj_file")
    args = parser.parse_args()

    print("WARNING this program won't take care of normal and uv coord!!!!!")
    
    #######################################
    ###read in transfer data
    #######################################
    with open(args.aln_file,"r") as pf:
        obj_file_num = int(pf.readline().strip("\n"))
        assert obj_file_num == 2,"There are more than 2 objects in:{}".format(args.aln_file)
        this_ply_name = pf.readline().strip("\n")
        if "fused.ply" in this_ply_name:
            for i in range(6):
                pf.readline()
        pf.readline()#read in #
        collector = []
        for i in range(4):
            tmp_data = pf.readline().strip("\n").strip(" ").split(" ")
            collector.extend(tmp_data)
        trans_1 = np.asarray(collector,np.float32).reshape((4,4))
    # assert not np.allclose(trans_1,np.eye(4)),"{} is wierd, it looks like an identity matrix.".format(args.aln_file)

    with open(args.aln_file_matlab,"r") as pf:
        obj_file_num = int(pf.readline().strip("\n"))
        assert obj_file_num == 2,"There are more than 2 objects in:{}".format(args.aln_file)
        this_ply_name = pf.readline().strip("\n")
        if "X.ply" in this_ply_name:
            for i in range(6):
                pf.readline()
        pf.readline()#read in #
        collector = []
        for i in range(4):
            tmp_data = pf.readline().strip("\n").strip(" ").split(" ")
            collector.extend(tmp_data)
        trans_2 = np.asarray(collector,np.float32).reshape((4,4))
    # assert not np.allclose(trans_2,np.eye(4)),"{} is wierd, it looks like an identity matrix.".format(args.aln_file)

    trans = np.matmul(trans_2,trans_1)
    #######################################
    ###read in origin data
    #######################################

    scan_mesh_rawdata = open(args.scan_obj_file,"r").read().strip("\n").split("\n")
    face_row_collector = []
    v_collector = []
    vn_collector = []
    vt_collector = []
    with open(args.save_obj_file,"w") as pf:
        for a_line in scan_mesh_rawdata:
            if "#" in a_line:
                pf.write(a_line+"\n")
            elif 'mtllib' in a_line:
                continue
                # pf.write(a_line+"\n")
            elif "v" in a_line:
                data = np.array(a_line.split(" ")[1:4],np.float32)
                if "vn" in a_line:#normal
                    vn_collector.append(data)
                elif "vt" in a_line:
                    vt_collector.append(a_line)
                else:#point
                    v_collector.append(data)
            elif "f" in a_line:#face
                data = " ".join([a.split('/')[0] for a in a_line.split(" ")])
                face_row_collector.append(data)
        
        #######################################
        ###transfer here
        #######################################
        v_collector = np.asarray(v_collector)
        v_collector = np.concatenate((v_collector,np.ones((v_collector.shape[0],1),v_collector.dtype)),axis=1)
        print(trans.shape,v_collector.T.shape)
        v_collector = np.matmul(trans,v_collector.T).T#(n,4)
        
        #######################################
        ###output here
        #######################################
        v_collector = v_collector.astype(np.str)
        v_collector = np.char.add(" ",v_collector)
        v_head = np.repeat(np.array(['v ']).reshape(1,1),v_collector.shape[0],axis=0)
        v_collector = np.concatenate((v_head,v_collector[:,:3]),axis=1)
        for a_line_v in ["".join(i) for i in v_collector]:
            pf.write("{}\n".format(a_line_v))
    
        for a_line_f in face_row_collector:
            pf.write("{}\n".format(a_line_f))