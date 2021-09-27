@echo off
SET MODEL_ROOT="../../models/"
SET MODEL_FILE_NAME="model_state_1000000.pkl"
SET NODE_NAME="linear_projection_pointnet_pipeline."
SET SAMPLE_VIEW_NUM=1
SET ALL_POS=1
@echo on
::[STEP1]get pattern
python get_lighting_pattern_restore.py %MODEL_ROOT% %MODEL_FILE_NAME% %NODE_NAME% %SAMPLE_VIEW_NUM%
::
