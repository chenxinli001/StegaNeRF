SCENE=$1
STYLE=$2
GPU_ID=$3

dataset=llff
ckpt_svox2=ckpt_svox2/${dataset}/${SCENE}
ckpt_ste=ckpt_ste/${dataset}/${SCENE}_${STYLE}
data_dir=../data/${dataset}/${SCENE}
style_img=../data/watermarks/${STYLE}


if [[ ! -f "${ckpt_svox2}_low/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2}_low  ../data/${dataset}/${SCENE} --gpuid ${GPU_ID}\
                    -c configs/${dataset}_low.json
fi


python opt_stega.py \
-t ${ckpt_ste} ${data_dir} -c configs/${dataset}_fixgeom_wm.json \
--init_ckpt ckpt_svox2/${dataset}/${SCENE}_low/ckpt.npz \
# --no_pre_ct --no_post_ct 
--reset_basis_dim 0 --mse_cp init \
--w_wm 0.5 --w_wm_ctr 0.5 --w_dis_wm 0.01 --w_rgb 1 \
--total_epoch 55 --wm_resize 128 \
--mask --n_power_softw 3 --gpu ${GPU_ID}\
--style ./data/watermarks/${STYLE}
--root ckpt --suffix mask-pow

