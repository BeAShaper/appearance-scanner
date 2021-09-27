# appearance-scanner

## About

This repository is an implementation of the neural network proposed in [Free-form Scanning of Non-planar Appearance with Neural Trace Photography](https://svbrdf.github.io/publications/scanner/project.html)

For any questions, please email xiaohema98 at gmail.com 
## Usage
### System Requirement

+ Windows or Linux(The codes are validated on Win10, Ubuntu 18.04 and Ubuntu 16.04)
+ Python >= 3.6.0
+ Pytorch >= 1.6.0
+ tensorflow>=1.11.0, meshlab and matlab are needed if you process the test data we provide  


### Training

1. move to appearance_scanner
2. run train.bat or train.sh according to your own platform 

Notice that the data generation step
```
python data_utils/origin_parameter_generator_n2d.py %data_root% %Sample_num% %train_ratio%
``` 

should be run only once.

### Training Visulization

When training is started, you can open tensorboard to observe the training process.
There will be two log images of a certain training sample, one is the sampled lumitexels from 64 views and the other is an composite image from six images in the order of groundtruth lumitexel, groundtruth diffuse lumitexel, groundtruth specular lumitexel, predicted lumitexel, predicted diffuse lumitexel and predicted specular lumitexel.

<table>
    <tr>
    <td><img src='./imgs/input.png'></td> 
    <td><img src='./imgs/output.png' height=146></td> </td></tr>
</table>



Trained lighting pattern will also be showed. Trained model will be found in the `log_dir` set in train.bat/train.sh.

### License
Our source code is released under the GPL-3.0 license for acadmic purposes. The only requirement for using the code in your research is to cite our paper: 

```
@article{Ma:2021:Scanner,
author = {Ma, Xiaohe and Kang, Kaizhang and Zhu, Ruisheng and Wu, Hongzhi and Zhou, Kun},
title = {Free-Form Scanning of Non-Planar Appearance with Neural Trace Photography},
year = {2021},
issue_date = {August 2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {40},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3450626.3459679},
doi = {10.1145/3450626.3459679},
journal = {ACM Trans. Graph.},
month = jul,
articleno = {124},
numpages = {13},
keywords = {illumination multiplexing, SVBRDF, optimal lighting pattern}
}
```
For commercial licensing options, please email hwu at acm.org.
See COPYING for the open source license.

The third party remeshing tool ACVD and alignment tool CPD are not under our protection.



## Reconstruction process

Download our [Cheongsam](https://drive.google.com/file/d/1KOAJcUrDdWgEWN_obL5ccwNYJRrgFdua/view?usp=sharing) test data and unzip it in appearance_scanner/data/.


<table>
    <tr>
        <td>
            <img src="./imgs/pd_predicted_118_0.png" width="100%" title="shot photo">
        </td>
        <td>
            <img src="./imgs/pd_predicted_370_0.png" width="100%" title="shot photo">
        </td>
        <td>
            <img src="./imgs/pd_predicted_616_0.png" width="100%" title="shot photo">
        </td>
    </tr>
    <tr>
        Sample photographs captured from the Cheongsam object. The brightness of the original images has been doubled for a better visualization.
    </tr>
</table>



 
Download  our [model](https://drive.google.com/file/d/1qWEusL4JfGcp2Wqt64clsBWOAae-SHU0/view?usp=sharing) and unzip it in appearance_scanner/.

### 1. Camera Registration 

#### 1.1 Run `SFM/run.bat` first to brighten the raw images

#### 1.2 Open Colmap and do the following steps

**1.2.1** New project

<img src="./imgs/project.png" title="project">

**1.2.2** Feature extraction

Copy the parameters of our camera in Cheongsam/cam.txt to `Custom parameters`.

<img src="./imgs/feature_extraction.png" title="feature extraction">


**1.2.3** Feature matching

Tick `guided_matching` and run.

<img src="./imgs/feature_matching.png" title="feature matching">

**1.2.4**  Reconstruction options

Do **not** tick `multiple_models` in the General sheet.

<img src="./imgs/reconstruction0.png" title="reconstruction">

Do **not** tick `refine_focal_length/refine_extra_params/use_pba` in the Bundle sheet.

<img src="./imgs/reconstruction.png" title="reconstruction">

Start reconstruction.

**1.2.5** Bundle adjustment

Do **not** tick `refine_focal_length/refine_principal_point/refine_extra_params`.

<img src="./imgs/bundle_adjustment.png" title="bundle adjustment">

**1.2.6** 

Make a folder named **undistort_feature** in Cheongsam/ and export model as text in undistort_feature folder. Three files including cameras.txt, images.txt and point3D.txt will be saved.



**1.2.7** 

Dense reconstruction -> select undistort_feature folder -> Undistortion -> Stereo

Since we upload all the photos we taken,  it will take a long time to run this step.
We strongly recommend you run

```
colmap stereo_fusion --workspace_path path --input_type photometric --output_path path/fused.ply

//change path to undistort_feature folder
``` 


when the files amount in undistort_feature/stereo/normal_maps arise to around 200-250. 
It will output a coarse point cloud in undistort_feature/ .

<img src="./imgs/sfm_model.png" title="sfm point cloud">

Delete the noise points and the table plane.

<img src="./imgs/sfm_model_clean.png" title="sfm point cloud">


Save fused.ply.


### 2. Extract measurements

move your own model to `models/` and run `appearance_scanner/test_files/prepare_pattern.bat`

run `extract_measurements/run.bat`

### 3. Align mesh

#### 3.1 Use meshlab to align mesh roughly

Open fused.ply and Cheongsam/scan/Cheongsam.ply in the same meshlab window.

<img src="./imgs/meshlab_align.png" title="meshlab align">

Align two mesh and save project file in Cheongsam/scan/Cheongsam.aln, which records the transform matrix between two meshes.

run `CoherentPointDrift/run.bat` to align Cheongsam.ply to fused.ply. 

<img src="./imgs/aln1.png" title="after meshlab align">

#### 3.2 Further Alignment

run `CoherentPointDrift/CoherentPointDrift-master/simplify/run.bat` to simplify two meshes.

Open the CPD project in Matlab and run `main.m`.
<img src="./imgs/aln2.png" title="matlab align">

After alignment done, run `CoherentPointDrift/run_pass2.bat`. meshed-poisson_obj.ply will be saved in undistort_feature/ .

You should open fused.ply and meshed-poisson_obj.ply in the same meshlab window to check the quality of alignment. It is a key factor in the final result.

### 4. Generate view information from registrated cameras

#### 4.1

 run `ACVD/aarun.bat`  
 
 save undistort_feature/meshed-poisson_obj_remeshed.ply as undistort_feature/meshed-poisson_obj_remeshed.obj

#### 4.2

copy data_processing/device_configuration/extrinsic.bin to undistort_feature/
copy Cheongsam/512.exr and 1024.exr to undistort_feature/

run `generate_texture/trans.bat` to transform mesh from colmap frame to world frame in our system and generate uv maps.

We recommend that you generate uv maps with resolution of 512x512 because it will save a lot of time and retain most details. The resolution of the results in our paper is 1024x1024. 

You can set `UVMAP_WIDTH` and `UVMAP_HEIGHT` to 1024 in `uv/uv_generator.bat` if you pursue higher quality.

#### 4.3

in `generate_texture/texgen.bat`, set `TEXTURE_RESOLUTION` to the certain resolution

choose the same line or the other reference on meshed-poisson_obj_remeshed.obj and on the physical object, then meature the lengths of both. Set the results to `COLMAP_L` and `REAL_L`. `REAL_L` in mm.

<img src="./imgs/SCALAR.png" width=60% title="scalar">

The marker cylinder's diameter is 10cm, so we set `REAL_L` to 100.


run `generate_texture/texgen.bat` to output view information of all registrated cameras

### 5. Gather data

run `gather_data/run.bat` to gather the inputs to the network for each valid pixel on the texture map. A folder named images_{resolution} will be made in Cheongsam/.

### 6. Fitting

1. Change %DATA_ROOT% and %TEXTURE_MAP_SIZE% in `fitting/tf_ggx_render/run.bat`. Then run `fitting/tf_ggx_render/run.bat`.  
2. A folder named `fitting_folder_for_server` will be generated under texture_{resolution}.  
3. Upload the entire folder generated in previous step to a linux server.  
4. Change current path of terminal to `fitting_folder_for_server\fitting_temp\tf_ggx_render`, then run `split.sh` or `split1024.sh` according to the resolution you chosen. (split.sh is for 512. If you want to use custom texture map resolution, you may need to modify the $TEX_RESOLUTION in split.sh)
5. When the fitting procedure finished, a folder named `Cheongsam/images_{resolution}/data_for_server/data/images/data_for_server/fitted_grey` will be generated. It contains the final texture maps, including normal_fitted_global.exr, tangent_fitted_global.exr, axay_fitted.exr, pd_fitted.exr and ps_fitted.exr.  
Note: If you find the split.sh cannot run properly and complain about abscent which_server argument, it's probably caused by the difference of linux and windows. Reading in the sh file and writing it with no changing of content on sever can fix this issue.## Reference 

<table>
    <tr>
        <td align="center"><img src='./imgs/pd_fitted.png' width="100%"></td> 
        <td align="center"><img src='./imgs/ps_fitted.png' width="100%"></td> 
        <td align="center"><img src='./imgs/axay_fitted.png' width="100%"></td> 
    </tr>
    <tr>
        <td align="center">diffuse</td>
        <td align="center">specular(with exposure -3 in photoshop)</td>
        <td align="center">roughness</td>
    </tr>
    <tr>
        <td align="center"><img src='./imgs/normal_fitted_global.png' width="100%"></td> 
        <td align="center"><img src='./imgs/tangent_fitted_global.png' width="100%"></td> 
    </tr>
    <tr>
        <td align="center">normal</td>
        <td align="center">tangent</td>
    </tr>
    
</table>

### 7. Render results
<table>
    <tr>
        <td align="center"><img src='./imgs/render1.png'></td> 
        <td align="center"><img src='./imgs/render2.png'></td> 
    </tr>
    
</table>

## Reference & Third party tools
Colmap: https://demuc.de/colmap/

Coherent Point Drift: https://ieeexplore.ieee.org/document/5432191

ACVD: https://github.com/valette/ACVD



