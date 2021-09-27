data_root=../training_data/
sample_view_num=64
lighting_pattern_num=1
training_gpu=0

sample_num=200000000
train_ratio="0.9"
learning_rate=1e-4
log_dir=../runs/
slice_width=256
slice_height=192
PRETRAINED_MODEL_PAN=""

python data_utils/origin_parameter_generator_n2d.py %data_root% %Sample_num% %train_ratio%

python train.py %data_root% %slice_width% %slice_height% --training_gpu %training_gpu% --sample_view_num %sample_view_num% --lighting_pattern_num %lighting_pattern_num% --log_dir %log_dir% --pretrained_model_pan %PRETRAINED_MODEL_PAN%
pausetrain