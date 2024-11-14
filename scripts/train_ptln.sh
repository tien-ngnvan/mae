CUDA_VISIBLE_DEVICES=0,1 python train_ptln.py \
        --devices 0 1 \
        --batch_size 140 \
        --accum_iter 1 \
        --epochs 150 \
        --model mae_vit_base_patch16 \
        --input_size 224 \
        --weight_decay 0.05 \
        --blr 1.5e-4 \
        --warmup_epochs 4 \
        --dataset_name_train "presencesw/general_20_data_remove_v0" \
        --image_folder \
            "/mnt/Datadrive/tiennv/data/final" \
            "/mnt/Datadrive/tiennv/data/data_remove_v0/train" \
            "/mnt/Datadrive/datasets/ade20k/ade20k" \
            "/mnt/Datadrive/datasets/ade20k/pascal-context" \
            "/mnt/Datadrive/datasets/coco2017/train" \
            "/mnt/Datadrive/datasets/coco2017/val" \
        --do_train \
        --do_eval \
        --mask_ratio 0.5 \
        --mask_min 0 \
        --mask_max 1 \
        --cache_dir .cache \
        --output_dir outputs_rand_4_bitwise_3_semi_objmask_150/files \
        --log_dir outputs_rand_4_bitwise_3_semi_objmask_150/logs \
        --weights checkpoints/mae_visualize_vit_base.pth \
        --mask_mode 'rand' 'bitwise' 'semi_objmask' \
        --dis_mask 0.4 0.3 0.3

# torch.distributed.launch