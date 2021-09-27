SET PNG_ROOT=../../data/Cheongsam

SET /A LIGHTING_PATTERM_NUM=1

SET /A VIEW_NUM=1206

SET MODEL_PATH="../../models/"

SET WHICH_CAM=0
SET INTRINSIC_FILE_pAN=../device_configuration/intrinsic%WHICH_CAM%.yml

python extract_measurements.py %PNG_ROOT% %VIEW_NUM% %LIGHTING_PATTERM_NUM% %MODEL_PATH%
