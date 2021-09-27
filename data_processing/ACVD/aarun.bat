SET DATA_ROOT=F:/Turbot/appearance-scanner/data/Cheongsam/undistort_feature/
SET FILE=%DATA_ROOT%meshed-poisson_obj.ply
SET FILE_O=meshed-poisson_obj_remeshed.ply
SET NVERTICES=100000
SET GRADATION=0

ACVD.exe %FILE% %NVERTICES% %GRADATION% -of %FILE_O% -o %DATA_ROOT%