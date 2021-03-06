SET SHOT_ROOT=../../data/
SET DATA_ROOT=%SHOT_ROOT%Cheongsam/
SET /A M_LEN=3
SET /A LIGHTING_PATTERN_NUM=1
SET /A THREAD_NUM=24
SET IS_FOR_SERVER=true
SET IF_FIX_NORMAL=false

SET MODEL_PATH="../../models/"
SET MODEL_NAME="model_state_1000000.pkl"

SET SCALAR=184.0
SET TEXTURE_MAP_SIZE=512
SET SUB_FOLDER_NAME=512/
SET /A PD_SAMPLE_NUM = 8
SET /A PS_SAMPLE_NUM = 32
SET CAMERA_EXTRINSIC_FILE=../device_configuration/extrinsic.bin

python gather_data.py %DATA_ROOT% %SUB_FOLDER_NAME% %SCALAR% %TEXTURE_MAP_SIZE% %CAMERA_EXTRINSIC_FILE% %MODEL_PATH%


python prepare_fitting.py %DATA_ROOT% %MODEL_PATH% %MODEL_NAME% %PD_SAMPLE_NUM% %PS_SAMPLE_NUM% %M_LEN% %LIGHTING_PATTERN_NUM% %THREAD_NUM% %IF_FIX_NORMAL% %TEXTURE_MAP_SIZE% 


