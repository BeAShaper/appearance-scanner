@echo off
SET ROOT=F:/SIG21/appearance-scanner/data/
SET FEATURE_PATH=Cheongsam/

:::::::::::::::::::::::::
::: convert mesh from colmap frame to custom frame
:::::::::::::::::::::::::

SET DATA_ROOT=%ROOT%%FEATURE_PATH%
SET UDT_FOLDER_NAME=undistort_feature/
SET CONFIG_DIR="../device_configuration/"

SET COL_OBJ_NAME=meshed-poisson_obj_remeshed.obj
SET W_OBJ_NAME=meshed-poisson_obj_remeshed_w.obj
SET SAMPLE_VIEW_NUM=1206
::SAMPLE_VIEW_NUM must be set to total number of photos send to colmap

@REM SCALAR
SET COLMAP_L=1.13287
SET REAL_L=100 

@echo on
TextureGenerator.exe --dataroot %DATA_ROOT% --udt %UDT_FOLDER_NAME% --config %CONFIG_DIR% --sampleviewnum %SAMPLE_VIEW_NUM% --trans --iobj %COL_OBJ_NAME% --oobj %W_OBJ_NAME% --real_l %REAL_L% --colmap_l %COLMAP_L%
@echo off


SET UV_ATLAS_PATH=../uv/
cd /d %UV_ATLAS_PATH%
@echo on
uv_generator.bat %DATA_ROOT%%UDT_FOLDER_NAME%/ %~dp0
@echo off