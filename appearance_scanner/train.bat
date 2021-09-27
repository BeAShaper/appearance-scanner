SET data_root=../training_data/
SET sample_view_num=64
SET lighting_pattern_num=1
SET training_gpu=0

SET sample_num=200000000
SET train_ratio="0.9"
SET learning_rate=1e-4
SET log_dir=../runs/
set slice_width=256
set slice_height=192
set PRETRAINED_MODEL_PAN=""

python data_utils/origin_parameter_generator_n2d.py %data_root% %Sample_num% %train_ratio%

python train.py %data_root% %slice_width% %slice_height% --training_gpu %training_gpu% --sample_view_num %sample_view_num% --lighting_pattern_num %lighting_pattern_num% --log_dir %log_dir% --pretrained_model_pan %PRETRAINED_MODEL_PAN%
pause
