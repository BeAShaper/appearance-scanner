SET SHOT_RESULT_ROOT=../../data/Cheongsam/
SET VIEW_NUM=1206

SET INTRINSIC_PATH=../device_configuration/

python prepare_images_for_sfm.py %SHOT_RESULT_ROOT% %VIEW_NUM%
