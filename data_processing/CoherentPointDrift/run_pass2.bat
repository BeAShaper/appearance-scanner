SET NAME=Cheongsam
SET DATA_ROOT=F:/Turbot/appearance-scanner/data/%NAME%

SET TRANS_PLY_FILE=%DATA_ROOT%/undistort_feature/%NAME%_rec.ply
SET SAVE_ROOT=%DATA_ROOT%/undistort_feature/meshed-poisson_obj.ply
SET ALN_FILE=trans.aln
python transform_ply.py %TRANS_PLY_FILE% %ALN_FILE% %SAVE_ROOT% --matlab