@echo off
SET ROOT=../../data/
SET FEATURE_PATH=Cheongsam/

:::::::::::::::::::::::::
::: generate texture map 
:::::::::::::::::::::::::

SET DATA_ROOT=%ROOT%%FEATURE_PATH%
SET UDT_FOLDER_NAME=undistort_feature
SET CONFIG_DIR=../device_configuration/

SET SAMPLE_VIEW_NUM=1206

SET COLMAP_L=1.13287
SET REAL_L=100 

SET TEXTURE_RESOLUTION=1024
SET MESH_PATH=mesh_%TEXTURE_RESOLUTION%/
SET TEX_PATH=texture_%TEXTURE_RESOLUTION%/

@echo on
TextureGenerator.exe --dataroot %DATA_ROOT% --udt %UDT_FOLDER_NAME% --config %CONFIG_DIR% --sampleviewnum %SAMPLE_VIEW_NUM% --texgen --meshpath %MESH_PATH% --texturepath %TEX_PATH%  --real_l %REAL_L% --colmap_l %COLMAP_L% --textureres %TEXTURE_RESOLUTION%


@echo off