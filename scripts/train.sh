python train.py \
        --norm_pix_loss \
        --batch_size 256 \
        --epochs 100 \
        --model mae_vit_base_patch16 \
        --input_size 224 \
        --norm_pix_loss \
        --weight_decay 0.05 \
        --blr 1e-3 \
        --warmup_epochs 5 \
        --dataset_name_train "inpaint-context/train-mae-update-furniture" \
        --image_folder \
            "/mnt/Datadrive/tiennv/data/final" \
            "/mnt/Datadrive/datasets/ade20k/ade20k" \
            "/mnt/Datadrive/datasets/ade20k/pascal-context" \
            "/mnt/Datadrive/datasets/coco2017/train" \
            "/mnt/Datadrive/datasets/coco2017/val" \
        --do_train \
        --do_eval \
        --mask_ratio 0.75 \
        --mask_min 0 \
        --mask_max 1 \
        --cache_dir .cache \
        --output_dir outputs/files \
        --log_dir outputs/logs \
        --world_size 2 \
        --weights checkpoints/mae_pretrain_vit_base_full.pth \
        --mask_mode 'objmask' 
