
SET SHOT_ROOT=F:/Turbot_freshmeat/12_19/
SET DATA_ROOT=%SHOT_ROOT%egypt/
SET /A M_LEN=1
SET /A LIGHTING_PATTERN_NUM=1
SET /A THREAD_NUM=12
SET IS_FOR_SERVER=true
SET CONFIG_FILE=tf_ggx_render_configs_plane_new
SET IF_FIX_NORMAL=false

SET MODEL_PATH="F:/Turbot/runs/10/1206_64_random_sigmoid_handheld_1p/"
SET MODEL_NAME="model_state_3000000.pkl"

SET SELECT_VIEW_NUM=128
SET SCALAR=200
REM 377.67454703
SET TEXTURE_MAP_SIZE=512
SET SUB_FOLDER_NAME=selected_views_test_128_ba/
SET /A PD_SAMPLE_NUM = 8
SET /A PS_SAMPLE_NUM = 32
SET ORDER=1
SET CAMERA_EXTRINSIC_FILE="F:/Turbot/torch_renderer/wallet_of_torch_renderer/handheld_device_render_config_16x32/extrinsic.bin"

python select_views_for_each_texel.py %DATA_ROOT% %SUB_FOLDER_NAME% %LIGHTING_PATTERN_NUM% %M_LEN% %SCALAR% %SELECT_VIEW_NUM% %TEXTURE_MAP_SIZE% %CAMERA_EXTRINSIC_FILE% %MODEL_PATH%

python prepare_fitting_for_server.py %DATA_ROOT% %SUB_FOLDER_NAME% %MODEL_PATH% %MODEL_NAME% %PD_SAMPLE_NUM% %PS_SAMPLE_NUM% %SELECT_VIEW_NUM% %M_LEN% %LIGHTING_PATTERN_NUM% %THREAD_NUM% %IF_FIX_NORMAL% %TEXTURE_MAP_SIZE% --add_normal

::::::::::::::::::::::::::::::::::::::::::::::::::
@REM SET FEATURE_TASK=egypt2
@REM SET MATERIAL_TASK=egypt2
@REM SET UDT_FOLDER_NAME=undistort_feature
@REM SET TEX_FOLDER_NAME=/texture_%TEXTURE_MAP_SIZE%
@REM @REM REM SET
@REM python n_local_to_global.py %DATA_ROOT% %TEX_FOLDER_NAME% %CONFIG_FILE% %IS_FOR_SERVER% %SUB_FOLDER_NAME%

@REM SET TEX_FOLDER_NAME=/texture_%TEXTURE_MAP_SIZE%
@REM python rotate_normal_back.py %SHOT_ROOT% %FEATURE_TASK% %MATERIAL_TASK% %UDT_FOLDER_NAME% %TEX_FOLDER_NAME% %TEXTURE_MAP_SIZE% %SUB_FOLDER_NAME%
@REM python texture_map_draw.py %SHOT_ROOT% %FEATURE_TASK% %MATERIAL_TASK% %UDT_FOLDER_NAME% %TEX_FOLDER_NAME% %TEXTURE_MAP_SIZE% %SUB_FOLDER_NAME%
