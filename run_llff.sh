'''
>> trex
    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix 1014_image+audio+text --clip_grad True --max_norm 1 --gpuid 0 



>> horns

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/horns/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/horns_low/ckpt.npz --style ../data/multi-modal/horns.webp --audio_wave ../data/multi-modal/horns.wav --text_scene horns --suffix 1012_image+audio+text --gpuid 0


>> leaves

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/leaves/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/leaves_low/ckpt.npz --style ../data/multi-modal/leaves.jfif --audio_wave ../data/multi-modal/leaves.wav --text_scene leaves --suffix 1012_image+audio+text --gpuid 0



1015  去除artifacts的尝试
>> trex

    恢复TV via using temp.json >> 还是很严重的色偏
    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm_temp.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_recover-TV --gpuid 1

    只加图像水印试一下会不会色偏太大 >> 竟然也是色偏。。。。
    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0 --w_text_wm_ctr 0 --w_audio_wm 0 --w_audio_wm_ctr 0 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_onlyimg --gpuid 1

        降采样试一下
        python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0 --w_text_wm_ctr 0 --w_audio_wm 0 --w_audio_wm_ctr 0 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_onlyimg-128 --gpuid 1 --wm_resize_h 128 --wm_resize_w 128



    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_gradient-clip-1 --gpuid 2 --clip_grad True --max_norm 1


    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_gradient-clip-0.01 --gpuid 2 --clip_grad True --max_norm 0.01

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_gradient-clip-0.001 --gpuid 0 --clip_grad True --max_norm 0.001
    
    gradient_clip 似乎有效,接下来尝试 
        python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_gradient-clip-0.0001 --gpuid 3 --clip_grad True --max_norm 0.0001

        加大RGB分量
        python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_gradient-clip-0.0001 --gpuid 0 --clip_grad True --max_norm 0.0001

        以及更严格的max_norm
        python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_gradient-clip-5e-5 --gpuid 0 --clip_grad True --max_norm 5e-5


    约束模型距离
    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_reg-sh-0.01 --gpuid 2 --weight_reg_sh 0.01

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_reg-sh-0.1 --gpuid 3 --weight_reg_sh 0.1

    EMA  0.95比较好
        python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_ema-ratio-0.99 --gpuid 0 --ema_ratio 0.99

        python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_ema-ratio-0.95 --gpuid 2 --ema_ratio 0.95

        python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_ema-ratio-0.90 --gpuid 0 --ema_ratio 0.90


    sh_mask
    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_unmask-id-0~1 --gpuid 2 --sh_unmask_id_start 0 --sh_unmask_id_end 1


    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_unmask-id-0~19 --gpuid 2 --sh_unmask_id_start 0 --sh_unmask_id_end 19

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_unmask-id-20~26 --gpuid 0 --sh_unmask_id_start 20 --sh_unmask_id_end 26

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_unmask-id-15~26 --gpuid 1 --sh_unmask_id_start 15 --sh_unmask_id_end 26

    >> unmask every color
    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_unmask-every-color-id-7~8 --gpuid 0 --sh_unmask_id_start 7 --sh_unmask_id_end 8 --unmask_every_color True

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_unmask-every-color-id-8 --gpuid 2 --sh_unmask_ids 8 --unmask_every_color True

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_unmask-every-color-id-0~1 --gpuid 1 --sh_unmask_id_start 0 --sh_unmask_id_end 1 --unmask_every_color True

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_unmask-every-color-id-[0,2,6] --gpuid 0 --sh_unmask_ids 0 2 6 --unmask_every_color True 

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_unmask-every-color-id-[1,3,4,5,7,8] --gpuid 1 --sh_unmask_ids 1 3 4 5 7 8 --unmask_every_color True 
    
    >pre-defined static
    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_unmask-every-color-id-0.75-pre --gpuid 0 --sh_unmask_ratio 0.75 --unmask_every_color True 

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_unmask-every-color-id-0.8-pre --gpuid 2 --sh_unmask_ratio 0.8 --unmask_every_color True 

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_unmask-every-color-id-0.6-pre --gpuid 0 --sh_unmask_ratio 0.6 --unmask_every_color True 


    10-19 对机制进一步探究， 加大RGB_W=2

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_unmask-every-color-id-[1,3,4,5,7,8] --gpuid 0 --sh_unmask_ids 1 3 4 5 7 8 --unmask_every_color True

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/trex/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/trex_low/ckpt.npz --style ../data/multi-modal/trex.jfif --audio_wave ../data/multi-modal/trex.wav --text_scene trex --suffix image+audio+text_1015temp_unmask-every-color-id-[1,3,4,5,7,8]_RE1 --gpuid 0 --sh_unmask_ids 1 3 4 5 7 8 --unmask_every_color True  






>> horns via unmask

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/horns/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/horns_low/ckpt.npz --style ../data/multi-modal/horns.webp --audio_wave ../data/multi-modal/horns.wav --text_scene horns --suffix image+audio+text_1015temp_unmask-id-0~1 --gpuid 4  --sh_unmask_id_start 0 --sh_unmask_id_end 1


    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/horns/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/horns_low/ckpt.npz --style ../data/multi-modal/horns.webp --audio_wave ../data/multi-modal/horns.wav --text_scene horns --suffix image+audio+text_1015temp_unmask-every-color-id-0.75-pre --gpuid 1 --sh_unmask_ratio 0.75 --unmask_every_color True 

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/horns/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/horns_low/ckpt.npz --style ../data/multi-modal/horns.webp --audio_wave ../data/multi-modal/horns.wav --text_scene horns --suffix image+audio+text_1015temp_unmask-every-color-id-0.8-pre --gpuid 1 --sh_unmask_ratio 0.8 --unmask_every_color True 


    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/horns/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/horns_low/ckpt.npz --style ../data/multi-modal/horns.webp --audio_wave ../data/multi-modal/horns.wav --text_scene horns --suffix image+audio+text_1015temp_unmask-every-color-id-0.6-pre --gpuid 1 --sh_unmask_ratio 0.6 --unmask_every_color True 

    10-19 对机制进一步探究， 加大RGB_W=2
    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/horns/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/horns_low/ckpt.npz --style ../data/multi-modal/horns.webp --audio_wave ../data/multi-modal/horns.wav --text_scene horns --suffix image+audio+text_1015temp_unmask-every-color-id-[1,3,4,5,7,8] --gpuid 1 --sh_unmask_ids 1 3 4 5 7 8 --unmask_every_color True 

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/horns/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/horns_low/ckpt.npz --style ../data/multi-modal/horns.webp --audio_wave ../data/multi-modal/horns.wav --text_scene horns --suffix image+audio+text_1015temp_unmask-every-color-id-[0,2,6] --gpuid 1 --sh_unmask_ids 0 2 6 --unmask_every_color True 

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/horns/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/horns_low/ckpt.npz --style ../data/multi-modal/horns.webp --audio_wave ../data/multi-modal/horns.wav --text_scene horns --suffix image+audio+text_1015temp_unmask-every-color-id-0 --gpuid 1 --sh_unmask_ids 0 2 6 --unmask_every_color True 

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/horns/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/horns_low/ckpt.npz --style ../data/multi-modal/horns.webp --audio_wave ../data/multi-modal/horns.wav --text_scene horns --suffix image+audio+text_1015temp_unmask-every-color-id-[1]_RE1 --gpuid 0 --sh_unmask_ids 1 --unmask_every_color True 

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/horns/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/horns_low/ckpt.npz --style ../data/multi-modal/horns.webp --audio_wave ../data/multi-modal/horns.wav --text_scene horns --suffix image+audio+text_1015temp_unmask-every-color-id-0~1_RE2 --gpuid 1 --sh_unmask_ids 0 1 --unmask_every_color True 


    10-20 新图像
    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/horns/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/horns_low/ckpt.npz --style ../data/multi-modal/horns.jfif --audio_wave ../data/multi-modal/horns.wav --text_scene horns --suffix image+audio+text_1015temp_new-img-wm --gpuid 0

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/horns/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/horns_low/ckpt.npz --style ../data/multi-modal/horns.jfif --audio_wave ../data/multi-modal/horns.wav --text_scene horns --suffix image+audio+text_1015temp_new-img-wm_unmask-every-color-id-[1,3,4,5,7,8] --gpuid 1 --sh_unmask_ids 1 3 4 5 7 8 --unmask_every_color True 






>>leaves via unmask
    
    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 2 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/leaves/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/leaves_low/ckpt.npz --style ../data/multi-modal/leaves.jfif --audio_wave ../data/multi-modal/leaves.wav --text_scene leaves --suffix image+audio+text_1015temp_unmask-id-0~1 --gpuid 5  --sh_unmask_id_start 0 --sh_unmask_id_end 1

    python opt_style_cp_init.py --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_img_wm 0.5 --w_img_wm_ctr 0.5 --w_text_wm 0.5 --w_text_wm_ctr 0.5 --w_audio_wm 0.5 --w_audio_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 100 --w_rgb 1 --w_rgb_lowf 0 --gaussian_ks 3 --gaussian_sigma 2 --mse_cp init ../data/llff/leaves/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/leaves_low/ckpt.npz --style ../data/multi-modal/leaves.jfif --audio_wave ../data/multi-modal/leaves.wav --text_scene leaves --suffix image+audio+text_1015temp_unmask-every-color-id-0.75-pre --gpuid 2 --sh_unmask_ratio 0.75 --unmask_every_color True 



>>FLOWER

    python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 2 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 65 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-id-0.75-pre --C_input_S --wm_resize 128 --sh_unmask_ratio 0.75 --unmask_every_color True 

  
    python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --gpuid 2 --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-id-0.75-pre_EPOCH55 --C_input_S --wm_resize 128 --sh_unmask_ratio 0.75 --unmask_every_color True 

    baseline

    python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-base --C_input_S --wm_resize 128  --unmask_value_mode ranking --unmask_reweight_mode hard --sh_unmask_ratio 1 --unmask_every_color True --gpu 0


    ranking  hard
        [0, 1, 0, 1, 1, 1, 0, 1, 1]
        python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+hard-0.75pre --C_input_S --wm_resize 128  --unmask_value_mode ranking --unmask_reweight_mode hard --sh_unmask_ratio 0.75 --unmask_every_color True --gpu 0

        [0, 1, 0, 1, 1, 0, 0, 0, 1]
        python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+hard-0.5pre --C_input_S --wm_resize 128  --unmask_value_mode ranking --unmask_reweight_mode hard --sh_unmask_ratio 0.5 --unmask_every_color True --gpu 0

        python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+hard-0.5pre_RE1 --C_input_S --wm_resize 128  --unmask_value_mode ranking --unmask_reweight_mode hard --sh_unmask_ratio 0.5 --unmask_every_color True --gpu 0

        python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+hard-0.25pre --C_input_S --wm_resize 128  --unmask_value_mode ranking --unmask_reweight_mode hard --sh_unmask_ratio 0.25 --unmask_every_color True --gpu 1




    ranking   soft
        python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+soft --C_input_S --wm_resize 128 --unmask_value_mode ranking --unmask_reweight_mode soft --unmask_every_color True --gpu 0
        
        >>n_power_softw = 2
        python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+soft-pow2 --C_input_S --wm_resize 128 --unmask_value_mode ranking --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 2 --gpu 1 

        >>n_power_softw = 5
        python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+soft-pow5 --C_input_S --wm_resize 128 --unmask_value_mode ranking --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --gpu 1 

    energy  soft
        python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --gpu 0

        >> n_power_softw = 2
        python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow2 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 2 --gpu 1

        >> n_power_softw = 5   这个效果好
            python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --gpu 1

            python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5_RE1 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --gpu 2

            python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5_RE2 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --gpu 2

            >>soft_energy + 不同的EMA
                python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5_ema0.95 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --ema_ratio 0.95 --gpu 2

                python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5_ema0.90 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --ema_ratio 0.90 --gpu 2

                python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5_ema0.99 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --ema_ratio 0.99 --gpu 2

                python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5_ema0.8 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --ema_ratio 0.8 --gpu 2

                python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5_ema0.5 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --ema_ratio 0.5 --gpu 2

                python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5_ema0.25 --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --ema_ratio 0.25 --gpu 2
            
            >> hard_ranking + 不同的EMA
                python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+hard-0.75pre_ema0.8 --C_input_S --wm_resize 128  --unmask_value_mode ranking --unmask_reweight_mode hard --sh_unmask_ratio 0.75 --unmask_every_color True --gpu 2 --ema_ratio 0.8

                python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg  --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-ranking+hard-0.75pre_ema0.95 --C_input_S --wm_resize 128  --unmask_value_mode ranking --unmask_reweight_mode hard --sh_unmask_ratio 0.75 --unmask_every_color True --gpu 2 --ema_ratio 0.95

        

        >> n_power_softw = 5  +  dynamic_unmask  >> dynamic的效果不会更好
        python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+soft-pow5-dynamic --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode soft --unmask_every_color True --n_power_softw 5 --dynamic_unmask --gpu 1

    
    energy hard
        [1, 1, 0, 1, 1, 1, 0, 1, 1]    
        python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+hard-0.5pre --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode hard --sh_unmask_ratio 0.5 --unmask_every_color True  --gpu 0

        [0, 1, 0, 1, 1, 0, 0, 0, 1]
        python opt_style_cp_init.py ../data/llff/flower/ -c configs/llff_fixgeom_wm.json --init_ckpt ckpt_svox2/llff/flower_low/ckpt.npz --style ../data/watermarkers/logo.jpg --no_pre_ct --no_post_ct --reset_basis_dim 0 --w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --total_epoch 55 --w_rgb 1 --mse_cp init --suffix 1021_unmask-every-color-energy+hard-0.25pre --C_input_S --wm_resize 128 --unmask_value_mode energy --unmask_reweight_mode hard --sh_unmask_ratio 0.125 --unmask_every_color True  --gpu 0




'''