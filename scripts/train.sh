python train.py \
        --norm_pix_loss \
        --batch_size 32 \
        --epochs 3 \
        --model mae_vit_base_patch16 \
        --input_size 224 \
        --warmup_epochs 1 \
        --dataset_name_train inpaint-context/test \
        --image_folder \
            /mnt/d/Work/Inpainting/data/data_folder/ade20k \
            /mnt/d/Work/Inpainting/data/data_folder/pascal-context \
            /mnt/d/Work/Inpainting/data/data_folder/bnb \
        --do_train \
        --do_eval \
        --max_train_samples 128 \
        --max_val_samples 32 \
        --mask_ratio 0.25 \
        --mask_min 0 \
        --mask_max 1 \
        --cache_dir .cache \
        --uptrain checkpoints/mae_pretrain_vit_base.pth \
        --output_dir outputs/files \
        --log_dir outputs/logs \
        --blr 1.5e-4 \
        --weight_decay 0.05 \
