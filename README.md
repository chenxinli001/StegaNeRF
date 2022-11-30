# StegaNeRF: Embedding Invisible Information within Neueral Radiance Fields
[[Paper]](https://github.com/XGGNet/StegaNeRF) [[Website]](https://xggnet.github.io/StegaNeRF/)
<div>

  <div style="display: flex; padding: 0px">
  <div style="max-width: 15%">
    <!-- <h5 align="center">Rendering Views</h5> -->
          <video class="video" autoplay="true" loop="true" style="margin-top: -1rem;" autoplay muted>
            <source src="doc/lego_rensu_.mp4">
          </video>
  </div>
  <div style="max-width: 15%">
    <!-- <h5 align="center">Residual Error (x25)</h5> -->
          <video class="video" autoplay="true" loop="true" style="margin-top: -1rem;" autoplay muted>
            <source src="videos/lego_ressu_.mp4">
          </video>
  </div>

  <div style="max-width: 15%">
    <!-- <h5 align="center">Recovered Information</h5> -->
          <video class="video" autoplay="true" loop="true" style="margin-top: -1rem;" autoplay muted>
            <source src="videos/lego_recsu_resize.mp4">
          </video>
  </div>  

  <div style="width: 10%"></div>

  <div style="max-width: 15%">
    <!-- <h5 align="center">Rendering Views</h5> -->
          <video class="video" autoplay="true" loop="true" style="margin-top: -1rem;" autoplay muted>
            <source src="doc/drums_rensu_.mp4">
          </video>
  </div>
  <div style="max-width: 15%">
    <!-- <h5 align="center">Residual Error (x25)</h5> -->
          <video class="video" autoplay="true" loop="true" style="margin-top: -1rem;" autoplay muted>
            <source src="videos/drums_ressu_.mp4">
          </video>
  </div>

  <div style="max-width: 15%">
    <!-- <h5 align="center">Recovered Information</h5> -->
          <video class="video" autoplay="true" loop="true" style="margin-top: -1rem;" autoplay muted>
            <source src="videos/drums_recsu_resize.mp4">
          </video>
  </div>  

</div>	


<!-- <img src="doc/lego_ren.gif" height="120"/>
<img src="doc/lego_res.gif" height="120"/>
<img src="doc/lego_rec.gif" height="120"/>
<img src="doc/drums_ren.gif" height="120"/>
<img src="doc/drums_res.gif" height="120"/>
<img src="doc/drums_rec.gif" height="120"/> 
</div> -->
<!-- ## Pipeline -->
<!-- <div align=center><img src="doc/MOTIVATION_V2.svg" height = "70%" width = "70%"/></div> -->
<!-- ![](doc/MOTIVATION_V2.svg) -->

## Method
<div align=center><img src="doc/METHOD_V2.svg" height = "100%" width = "90%"/></div>
<!-- ![](doc/MOTIVATION_V2.svg) -->

## Quick start
### Environment
```
. ./create_env.sh
```
<!-- Note: some errors may happen when installing the main library of [svox2](https://github.com/sxyu/svox2) including a CUDA extension, which is mainly due to the imcompatable   -->
### Dataset
Please download the datasets from these links:

- NeRF synthetic: Download `nerf_synthetic.zip` from https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
- LLFF: Download `nerf_llff_data.zip` from https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
<!-- - NeRF-W: Download `brandenburg_gate (4.0G)` from https://www.cs.ubc.ca/~kmyi/imw2020/data.html. More details to use this dataset can be found [here](https://github.com/kwea123/nerf_pl/tree/nerfw). -->

<!-- - DTU: Download the preprocessed DTU training data from https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view
Please download the depth from here: https://drive.google.com/drive/folders/13Lc79Ox0k9Ih2o0Y9e_g_ky41Nx40eJw?usp=sharing -->

### Training

```
cd opt && . ./stega_{llff/syn}.sh [scene_name] [style_id]
```
* At the first stage, a photorealistic radiance field will first be reconstructed if it doesn't exist on disk. Then the steganographic training at the second stage ends up with the steganographic NeRF and decoder.
* Select ```{llff/syn}``` according to your data type. For example, use ```llff``` for ```flower``` scene, ```syn``` for ```lego``` scene. 
* ```[style_id].jpg``` is the style image inside ```./data/watermarks```. 
<!-- * For example, ```14.jpg``` is the starry night painting. -->
<!-- * Note that a photorealistic radiance field will first be reconstructed for each scene, if it doesn't exist on disk. This will take extra time. -->


### Evaluation & Rendering

View the results by tensorboard. 

You can also obtain the results and rendering the videos from the saved checkpoints.

Use `opt/render_imgs.py` for the scenes on LLFF: `python render_imgs.py <CHECKPOINT.npz> <Decoder.pt> <data_dir>`

Use `opt/render_imgs_circle.py` to render a spiral for the scenes on NeRF synthetic: `python render_imgs_circle.py <CHECKPOINT.npz> <Decoder.pt> <data_dir>`

<!-- 
Render use `opt/render_imgs.py` for 
To render the 
Usage,
(in opt/)
`python render_imgs.py <CHECKPOINT.npz> <data_dir>`
By default this saves all frames, which is very slow. Add `--no_imsave` to avoid this.
## Rendering a spiral
Use `opt/render_imgs_circle.py`
Usage,
(in opt/)
`python render_imgs_circle.py <CHECKPOINT.npz> <data_dir>` -->



<!-- If you meet OOM issue, try:

1. enable `precision=16`
2. reduce the patch size `--patch_size` (or `--patch_size_x`, `--patch_size_y`) and enlarge the stride size `--sH`, `--sW` -->

<!-- <details>
  <summary>NeRF synthetic</summary>



    
          
            
    

          
          
            
    

          
    
    @@ -128,9 +86,9 @@ Usage,
  
- Step 1
  ```
  python train.py  --dataset_name blender_ray_patch_1image_rot3d  --root_dir  ../../dataset/nerf_synthetic/lego   --N_importance 64 --img_wh 400 400 --num_epochs 2000 --batch_size 1  --optimizer adam --lr 2e-4  --lr_scheduler steplr --decay_step 500 1000 --decay_gamma 0.5  --exp_name lego_s6 --with_ref --patch_size 64 --sW 6 --sH 6 --proj_weight 1 --depth_smooth_weight 0  --dis_weight 0 --num_gpus 4 --load_depth --depth_type nerf --model sinnerf --depth_weight 8 --vit_weight 10 --scan 4
  ```
- Step 2
  ```
  python train.py  --dataset_name blender_ray_patch_1image_rot3d  --root_dir  ../../dataset/nerf_synthetic/lego   --N_importance 64 --img_wh 400 400 --num_epochs 2000 --batch_size 1  --optimizer adam --lr 5e-5  --lr_scheduler steplr --decay_step 500 1000 --decay_gamma 0.5  --exp_name lego_s6_4ft --with_ref --patch_size 64 --sW 4 --sH 4 --proj_weight 1 --depth_smooth_weight 0  --dis_weight 0.01 --num_gpus 4 --load_depth --depth_type nerf --model sinnerf --depth_weight 8 --vit_weight 0 --pt_model xxx.ckpt --nerf_only  --scan 4
  ```
</details>
<details>
  <summary>LLFF</summary>
- Step 1
  ```
  python train.py  --dataset_name llff_ray_patch_1image_proj  --root_dir  ../../dataset/nerf_llff_data/room   --N_importance 64 --img_wh 504 378 --num_epochs 2000 --batch_size 1  --optimizer adam --lr 2e-4  --lr_scheduler steplr --decay_step 500 1000 --decay_gamma 0.5  --exp_name llff_room_s4 --with_ref --patch_size_x 63 --patch_size_y 84 --sW 4 --sH 4 --proj_weight 1 --depth_smooth_weight 0  --dis_weight 0 --num_gpus 4 --load_depth --depth_type nerf --model sinnerf --depth_weight 8 --vit_weight 10
  ```
- Step 2
  ```
  python train.py  --dataset_name llff_ray_patch_1image_proj  --root_dir  ../../dataset/nerf_llff_data/room   --N_importance 64 --img_wh 504 378 --num_epochs 2000 --batch_size 1  --optimizer adam --lr 5e-5  --lr_scheduler steplr --decay_step 500 1000 --decay_gamma 0.5  --exp_name llff_room_s4_2ft --with_ref --patch_size_x 63 --patch_size_y 84 --sW 2 --sH 2 --proj_weight 1 --depth_smooth_weight 0  --dis_weight 0.01 --num_gpus 4 --load_depth --depth_type nerf --model sinnerf --depth_weight 8 --vit_weight 0 --pt_model xxx.ckpt --nerf_only
  ```
</details>

<details>
  <summary>DTU</summary> -->

<!-- - Step 1
  ```
  python train.py  --dataset_name dtu_proj  --root_dir  ../../dataset/mvs_training/dtu   --N_importance 64 --img_wh 640 512 --num_epochs 2000 --batch_size 1  --optimizer adam --lr 2e-4  --lr_scheduler steplr --decay_step 500 1000 --decay_gamma 0.5  --exp_name dtu_scan4_s8 --with_ref --patch_size_y 70 --patch_size_x 56 --sW 8 --sH 8 --proj_weight 1 --depth_smooth_weight 0  --dis_weight 0 --num_gpus 4 --load_depth --depth_type nerf --model sinnerf --depth_weight 8 --vit_weight 10 --scan 4
  ```

    
        
          
    

        
    
    @@ -142,29 +100,26 @@ Usage,
  
- Step 2
  ```
  python train.py  --dataset_name dtu_proj  --root_dir  ../../dataset/mvs_training/dtu   --N_importance 64 --img_wh 640 512 --num_epochs 2000 --batch_size 1  --optimizer adam --lr 5e-5  --lr_scheduler steplr --decay_step 500 1000 --decay_gamma 0.5  --exp_name dtu_scan4_s8_4ft --with_ref --patch_size_y 70 --patch_size_x 56 --sW 4 --sH 4 --proj_weight 1 --depth_smooth_weight 0  --dis_weight 0.01 --num_gpus 4 --load_depth --depth_type nerf --model sinnerf --depth_weight 8 --vit_weight 0 --pt_model xxx.ckpt --nerf_only  --scan 4
  ```

More finetuning with smaller strides benefits reconstruction quality.

</details> -->

<!-- 
### Testing

```
python eval.py  --dataset_name llff  --root_dir /dataset/nerf_llff_data/room --N_importance 64 --img_wh 504 378 --model nerf --ckpt_path ckpts/room.ckpt --timestamp test
``` -->

<!-- Please use `--split val` for NeRF synthetic dataset. -->
## Experiments on NeRF-W
* Dataset: Download `brandenburg_gate (4.0G)` from https://www.cs.ubc.ca/~kmyi/imw2020/data.html. More details to use this dataset can be found [here](https://github.com/kwea123/nerf_pl/tree/nerfw).
- [ ] Code to be released; stays tuned.

## Acknowledgement

We would like to thank [Plenoxel](https://github.com/sxyu/svox2) authors for open-sourcing their implementations.

## Citation

If you find this repo is helpful, please cite:

<!-- ```
@InProceedings{Xu_2022_SinNeRF,
author = {Xu, Dejia and Jiang, Yifan and Wang, Peihao and Fan, Zhiwen and Shi, Humphrey and Wang, Zhangyang},

    
        
          
    

        
    
    @@ -173,4 +128,4 @@ journal={arXiv preprint arXiv:2204.00928},
  
title = {SinNeRF: Training Neural Radiance Fields on Complex Scenes from a Single Image},
journal={arXiv preprint arXiv:2204.00928},
year={2022}
}