SET NAME=Cheongsam
SET DATA_ROOT=F:/Turbot/appearance-scanner/data/%NAME%

SET SAVE_ROOT=%DATA_ROOT%/undistort_feature/%NAME%_rec.ply
SET SCAN_PLY_FILE=%DATA_ROOT%/scan/%NAME%.ply
SET ALN_FILE=%DATA_ROOT%/scan/%NAME%.aln

python transform_ply.py %SCAN_PLY_FILE% %ALN_FILE% %SAVE_ROOT% 

@REM SET SAVE_ROOT=%DATA_ROOT%/undistort_feature/meshed-poisson.obj
@REM SET OBJ_FILE=%DATA_ROOT%/scan/%NAME%.obj

@REM python transform_obj.py %OBJ_FILE% %ALN_FILE% %ALN_FILE_MATLAB% %SAVE_ROOT%



@REM @REM SET SCAN_PLY_FILE=%DATA_ROOT%/undistort_feature/%NAME%_rec.ply
@REM @REM SET SAVE_ROOT=%DATA_ROOT%/undistort_feature/meshed-poisson_obj.ply
@REM @REM SET ALN_FILE=trans.aln
@REM @REM python transform_ply.py %SCAN_PLY_FILE% %ALN_FILE% %SAVE_ROOT% --matlab

@REM @REM SET SCAN_PLY_FILE=%DATA_ROOT%/undistort_feature/%NAME%_obj_rec.ply
@REM @REM SET SAVE_ROOT=%DATA_ROOT%/undistort_feature/meshed-poisson_obj.ply
@REM @REM SET ALN_FILE=trans.aln
@REM @REM python transform_ply.py %SCAN_PLY_FILE% %ALN_FILE% %SAVE_ROOT% --matlab


@REM SET SAVE_ROOT=%DATA_ROOT%/undistort_feature/meshed-poisson.obj
@REM SET OBJ_FILE=%DATA_ROOT%/scan/%NAME%.obj
@REM SET ALN_FILE_MATLAB=trans.aln

@REM python transform_obj.py %OBJ_FILE% %ALN_FILE% %ALN_FILE_MATLAB% %SAVE_ROOT%

