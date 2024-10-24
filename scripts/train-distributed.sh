CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
    python -m torch.distributed.launch --nproc_per_node=2 \
        submitit_pretrain.py \
            --nodes 1 \
            --ngpus 2 \
            --partition local \
            --norm_pix_loss \
            --batch_size 64 \
            --epochs 100 \
            --model mae_vit_base_patch16 \
            --input_size 224 \
            --norm_pix_loss \
            --weight_decay 0.05 \
            --blr 1e-3 \
            --warmup_epochs 5 \
            --dataset_name_train "inpaint-context/opa-fix" \
            --image_folder \
                "/home/tiennv/hoang/mae/opa" \
            --do_train \
            --do_eval \
            --mask_ratio 0.75 \
            --mask_min 0 \
            --mask_max 1 \
            --cache_dir .cache \
            --output_dir outputs-distributed/files \
            --log_dir outputs-distributed/logs \
            --weights checkpoints/mae_pretrain_vit_base_full.pth \
            --mask_mode 'objmask'