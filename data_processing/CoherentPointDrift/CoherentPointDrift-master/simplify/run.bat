SET NAME=Cheongsam
SET IN_PATH=../../../../data/%NAME%/undistort_feature/
SET POINTCLOUD1=fused.ply
SET POINTCLOUD2=%NAME%_rec.ply

SET SIMPLIFY_PATH=../../mesh/

SET SCRIPT=simplify.mlx

mkdir %SIMPLIFY_PATH%

SET MESH1=X.ply
SET MESH2=Y.ply


::simplify the mesh
meshlabserver -i %IN_PATH%%POINTCLOUD1% -o %SIMPLIFY_PATH%%MESH1% -s %SCRIPT%
meshlabserver -i %IN_PATH%%POINTCLOUD2% -o %SIMPLIFY_PATH%%MESH2% -s %SCRIPT%