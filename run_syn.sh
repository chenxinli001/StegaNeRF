'''
快速验证 mask, hard+ranking, soft+energy, ema


调整baseline 更偏向水印， 加大水印 trade-off; 如此便于后续的AWM
    python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 0 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 1 --w_wm_ctr 1 --w_dis_wm 0.05 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_base --C_input_S --wm_resize 128 

    这个做baseline 可以 1-0.5-0.5-0.05
    python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 1 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.05 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_base --C_input_S --wm_resize 128
    
    baseline重跑，看看能不能低点...  再不行就把baseline 换回去。。。 >>换回去吧。。。
    python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 0 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.05 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_base_RE1 --C_input_S --wm_resize 128 

    
        >> tune n_pow=5
        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 1 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.05 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_energy+soft-pow5 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --gpu 0

        >> tune n_pow=3
        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 1 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.05 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_energy+soft-pow3 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 3 --gpu 0

        >> tune n_pow=7
        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 1 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.05 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_energy+soft-pow7 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 7 --gpu 1

        >> tune n_pow=2
        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 1 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.05 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_energy+soft-pow2 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 2 --gpu 0

        >> tune n_pow=1
        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 1 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.05 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_energy+soft-pow1 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 1 --gpu 1

baseline 换回去
python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 0 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_poor_base --C_input_S --wm_resize 128

python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 0 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_poor-base_energy+soft-pow5 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --gpu 1

re-run to certify
python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 0 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_poor-base_energy+soft-pow5_RE1 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --gpu 1

python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 0 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_poor-base_energy+soft-pow7 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 7 --gpu 2


python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 0 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_poor-base_energy+soft-pow1 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 1 --gpu 1

python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 0 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_poor-base_energy+soft-pow2 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 2 --gpu 1

python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 0 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_poor-base_energy+soft-pow3 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 3 --gpu 1


python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 0 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1026_poor-base_energy+soft-pow0.5 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 0.5 --gpu 1














>>baseline

    >>LEGO
        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 2 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_base --C_input_S --wm_resize 128 

    >>SHIP
        python opt_style_cp_init.py ../data/synthetic/ship/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/ship/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 2 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_base --C_input_S --wm_resize 128

    >>CHAIR
        python opt_style_cp_init.py ../data/synthetic/chair/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/chair/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 2 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_base --C_input_S --wm_resize 128

        python opt_style_cp_init.py ../data/synthetic/chair/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/chair/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 6 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.05 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_base --C_input_S --wm_resize 128
    
    >>DRUM
        python opt_style_cp_init.py ../data/synthetic/drums/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/drums/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 2 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_base --C_input_S --wm_resize 128

        python opt_style_cp_init.py ../data/synthetic/drums/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/drums/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 5 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.05 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_base --C_input_S --wm_resize 128

>>ranking+hard-0.75
    >> LEGO
        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+hard-0.75pre --C_input_S --wm_resize 128  --unmask_value_mode ranking --unmask_reweight_mode hard --sh_unmask_ratio 0.75 --unmask_every_color True --gpu 2

        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+hard-0.75pre_ema0.8 --C_input_S --wm_resize 128  --unmask_value_mode ranking --unmask_reweight_mode hard --sh_unmask_ratio 0.75 --unmask_every_color True --ema_ratio 0.8 --gpu 2

        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+hard-0.75pre_ema0.95 --C_input_S --wm_resize 128  --unmask_value_mode ranking --unmask_reweight_mode hard --sh_unmask_ratio 0.75 --unmask_every_color True --ema_ratio 0.95 --gpu 2

    试各种不同的masking ratio

    python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+hard-0.5pre --C_input_S --wm_resize 128  --unmask_value_mode ranking --unmask_reweight_mode hard --sh_unmask_ratio 0.5 --unmask_every_color True --gpu 1

    python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+hard-0.25pre --C_input_S --wm_resize 128  --unmask_value_mode ranking --unmask_reweight_mode hard --sh_unmask_ratio 0.25 --unmask_every_color True --gpu 1

    python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+hard-0.35pre --C_input_S --wm_resize 128  --unmask_value_mode ranking --unmask_reweight_mode hard --sh_unmask_ratio 0.35 --unmask_every_color True --gpu 1

    python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+hard-0.15pre --C_input_S --wm_resize 128  --unmask_value_mode ranking --unmask_reweight_mode hard --sh_unmask_ratio 0.15 --unmask_every_color True --gpu 1






>>energy+soft-n_pow=5
    >> LEGO
        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --gpu 2

        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5_RE1 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --gpu 2

        调参n_pow
        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow3 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 3 --gpu 0

        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow3_RE1 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 3 --gpu 1

        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow2 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 2 --gpu 1

        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow1 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 1 --gpu 2




        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5_ema0.8 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --ema_ratio 0.8 --gpu 2

        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5_ema0.95 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --ema_ratio 0.95 --gpu 2


        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5_ema0.5 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --ema_ratio 0.5 --gpu 2

        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5_ema0.25 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --ema_ratio 0.25 --gpu 1

        很低的EMA
        python opt_style_cp_init.py ../data/synthetic/lego/ -c configs/syn_fixgeom_wm.json --init_ckpt ckpt_svox2/synthetic_256_to_512_fasttv/lego/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5_ema0.1 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --ema_ratio 0.1 --gpu 1








'''