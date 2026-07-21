# This script is used to train TabICL for the first stage of the curriculum learning

# Choose ICL backbone: graph or encoder
ICL_BACKEND=${ICL_BACKEND:-graph}
# Enable wandb logging by setting WAND_LOG=True (and optionally WAND_MODE=online)
WAND_LOG=${WAND_LOG:-True}
WAND_MODE=${WAND_MODE:-online}
# GPU selection controls
DEVICE=${DEVICE:-cuda}
NUM_GPUS=${NUM_GPUS:-2}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
# Confusion matrix logging interval
LOG_CONF_MAT_EVERY=${LOG_CONF_MAT_EVERY:-2000}

export CUDA_VISIBLE_DEVICES

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------

torchrun --standalone --nproc_per_node=${NUM_GPUS} /workspace/src/tabicl/train/_run.py \
            --wandb_log ${WAND_LOG} \
            --wandb_project TabICL \
            --wandb_name Stage1 \
            --wandb_dir /workspace/wandb/ \
            --wandb_mode ${WAND_MODE} \
            --device ${DEVICE} \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 10000 \
            --batch_size 512 \
            --micro_batch_size 3 \
            --log_conf_mat_every ${LOG_CONF_MAT_EVERY} \
            --lr 8e-4 \
            --weight_decay 1e-4 \
            --supcon_weight 0.1 \
            --entropy_weight 0.0 \
            --icl_decoder_type soft_kmeans \
            --scheduler cosine_warmup \
            --warmup_proportion 0.02 \
            --gradient_clipping 2.0 \
            --prior_type nanotabicl \
            --prior_device cpu \
            --normalization robust \
            --batch_size_per_gp 8 \
            --min_features 2 \
            --max_features 10 \
            --max_classes 10 \
            --max_seq_len 1024 \
            --min_train_size 0.1 \
            --max_train_size 0.6 \
            --embed_dim 256 \
            --col_num_blocks 3 \
            --col_nhead 8 \
            --col_num_inds 128 \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 1 \
            --row_rope_base 100000 \
            --icl_num_blocks 4 \
            --icl_nhead 8 \
            --icl_backend ${ICL_BACKEND} \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_dir /workspace/checkpoints_d=4_supcon=0.1/stage1/ \
            --save_temp_every 1000 \
            --save_perm_every 5000 \
            --icl_soft_kmeans_temperature 0.5 \
            --label_smoothing 0.1 \
            --graph_min_train_neighbors 3 \
            --graph_max_train_neighbors 6 \
            --graph_test_k_per_class 2 \
            #--recompute True \
            #--scheduled_loader_steps 0,300,600,1000,2000 \
            #--scheduled_loader_sizes 64,256,1024,2048,inf
            #--model_type nanotabicl


# ------------------------------------------------------
# Save prior datasets to disk and load them for training
# ------------------------------------------------------

# Saving to disk
#python /workspace/src/tabicl/prior/_genload.py \
#    --save_dir /workspace/prior/stage1/ \
#    --np_seed 42 \
#    --torch_seed 42 \
#    --num_batches 100000 \
#    --resume_from 0 \
#    --batch_size 512 \
#    --batch_size_per_gp 4 \
#    --prior_type mix_scm \
#    --min_features 2 \
#    --max_features 100 \
#    --max_classes 10 \
#    --max_seq_len 1024 \
#    --min_train_size 0.1 \
#    --max_train_size 0.9 \
#    --n_jobs -1 \
#    --num_threads_per_generate 1 \
#    --device cpu

# Loading from disk and training
#torchrun --standalone --nproc_per_node=${NUM_GPUS} /workspace/src/tabicl/train/_run.py \
#            --wandb_log ${WAND_LOG} \
#            --wandb_project TabICL \
#            --wandb_name Stage1 \
#            --wandb_dir /workspace/wandb/ \
#            --wandb_mode ${WAND_MODE} \
#            --device ${DEVICE} \
#            --dtype float32 \
#            --np_seed 42 \
#            --torch_seed 42 \
#            --max_steps 100000 \
#            --batch_size 512 \
#            --micro_batch_size 4 \
#            --log_conf_mat_every ${LOG_CONF_MAT_EVERY} \
#            --lr 1e-4 \
#            --scheduler cosine_warmup \
#            --warmup_proportion 0.02 \
#            --gradient_clipping 1.0 \
#            --prior_dir /workspace/prior/stage1/ \
#            --load_prior_start 0 \
#            --delete_after_load False \
#            --prior_device cpu \
#            --embed_dim 128 \
#            --col_num_blocks 3 \
#            --col_nhead 8 \
#            --col_num_inds 128 \
#            --row_num_blocks 3 \
#            --row_nhead 8 \
#            --row_num_cls 4 \
#            --row_rope_base 100000 \
#            --icl_num_blocks 12 \
#            --icl_nhead 8 \
#            --icl_backend ${ICL_BACKEND} \
#            --ff_factor 2 \
#            --norm_first True \
#            --checkpoint_dir /workspace/checkpoints/stage1/ \
#            --save_temp_every 50 \
#            --save_perm_every 5000