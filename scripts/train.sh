CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --norm_pix_loss \
    --batch_size 8 \
    --epochs 5 \
    --model mae_vit_base_patch16 \
    --input_size 224 \
    --norm_pix_loss \
    --weight_decay 0.05 \
    --blr 1e-3 \
    --warmup_epochs 1 \
    --dataset_name_train "inpaint-context/test" \
    --image_folder \
        "/mnt/d/Work/Inpainting/data/data_folder/ade20k" \
        "/mnt/d/Work/Inpainting/data/data_folder/bnb" \
        "/mnt/d/Work/Inpainting/data/data_folder/pascal-context" \
    --do_train \
    --max_train_samples 128 \
    --do_eval \
    --mask_ratio 0.75 \
    --mask_min 0 \
    --mask_max 1 \
    --cache_dir .cache \
    --output_dir outputs/files \
    --log_dir outputs/logs \
    --weights "/mnt/d/Data/mae_pretrain_vit_base_full.pth" \
    --mask_mode 'objmask' \