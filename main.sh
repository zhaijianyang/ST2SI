#     --not_debug \
#--gradient_checkpointing_enable \
# allenai/longformer-base-4096

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main.py \
    --not_debug \
    --data_root /userhome/dataset/Recformer/finetune_data \
    --dataset Scientific \
    --output_dir ./log \
    --model_name_or_path facebook/opt-125m \
    --model_cache_dir /userhome/cache_models \
    --max_item_num 50 \
    --max_token_num 1024 \
    --gradient_accumulation_steps 4 \
    --train_attr title \
    --base_lr 5e-5 \
    --base_weight_decay 0 \
    --num_train_epochs 30 \
    --batch_size 8 \
    --query_token_mode 1 \
    --query_token_num 1 \
    --suffix debug \
    --index_alignment
