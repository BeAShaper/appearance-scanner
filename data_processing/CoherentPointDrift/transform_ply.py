import numpy as np
import open3d as o3d
import os
import cv2
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_ply_file")
    parser.add_argument("aln_file")

    parser.add_argument("save_root")
    parser.add_argument("--matlab",action="store_true",default=False)
    args = parser.parse_args()

    data = o3d.io.read_point_cloud(args.scan_ply_file)
    aln_file_path = args.aln_file

    points = np.asarray(data.points)
    colors = np.asarray(data.colors)
    normals = np.asarray(data.normals)

    trans_mat = []
    with open(aln_file_path, 'r') as pf:
        for i in range(9):
            lines = pf.readline()
            print(lines)
        for i in range(4):
            lines = pf.readline()
            lines = lines[:-1].split(' ')

            if not args.matlab:
                lines = lines[:-1]

            print(lines)
            lines = np.array([eval(item) for item in lines]).reshape([-1,4])
            trans_mat.append(lines)
    
    trans_mat = np.concatenate(trans_mat,axis=0)

    points = np.concatenate([points,np.ones([points.shape[0],1],np.float32)],axis=1)
    points = np.matmul(trans_mat,points.T)
    points = points.T
    points = points[:,:3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    o3d.io.write_point_cloud(args.save_root, pcd)