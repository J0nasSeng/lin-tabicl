# This script is used to train TabICL for the first stage of the curriculum learning

# Choose ICL backbone: graph or encoder
ICL_BACKEND=${ICL_BACKEND:-graph}
# Enable wandb logging by setting WAND_LOG=True (and optionally WAND_MODE=online)
WAND_LOG=${WAND_LOG:-True}
WAND_MODE=${WAND_MODE:-offline}
# GPU selection controls
DEVICE=${DEVICE:-cuda}
NUM_GPUS=${NUM_GPUS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5}
# Confusion matrix logging interval
LOG_CONF_MAT_EVERY=${LOG_CONF_MAT_EVERY:-200000} # effectively disables confusion matrix logging if set to a large number

# Stage 1.5 always starts from a Stage 1 checkpoint and writes its output to a
# separate directory so the Stage 1 checkpoints remain untouched.
STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-/workspace/checkpoints_dyngraph_intraclass=0.25/stage1/step-10000.ckpt}
STAGE1_5_CHECKPOINT_DIR=${STAGE1_5_CHECKPOINT_DIR:-/workspace/checkpoints_dyngraph_intraclass=0.25/stage1_5/}

if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
    echo "Stage 1 checkpoint not found: ${STAGE1_CHECKPOINT}" >&2
    exit 1
fi
mkdir -p "${STAGE1_5_CHECKPOINT_DIR}"

export CUDA_VISIBLE_DEVICES

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------

torchrun --standalone --nproc_per_node=${NUM_GPUS} /workspace/src/tabicl/train/_run.py \
            --wandb_log ${WAND_LOG} \
            --wandb_project TabICL \
            --wandb_name Stage1_5 \
            --wandb_dir /workspace/wandb/ \
            --wandb_mode ${WAND_MODE} \
            --device ${DEVICE} \
            --dtype bfloat16 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 1000 \
            --batch_size 128 \
            --micro_batch_size 2 \
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
            --normalization std \
            --batch_size_per_gp 2 \
            --min_features 2 \
            --max_features 512 \
            --max_classes 10 \
            --max_seq_len 1024 \
            --min_train_size 0.1 \
            --max_train_size 0.9 \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 8 \
            --col_num_inds 128 \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --icl_num_blocks 12 \
            --icl_nhead 8 \
            --icl_backend ${ICL_BACKEND} \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_path ${STAGE1_CHECKPOINT} \
            --checkpoint_dir ${STAGE1_5_CHECKPOINT_DIR} \
            --save_temp_every 1000 \
            --save_perm_every 5000 \
            --icl_soft_kmeans_temperature 0.5 \
            --label_smoothing 0.1 \
            --graph_min_train_neighbors 4 \
            --graph_max_train_neighbors 4 \
            --graph_train_neighbors_per_test 2 \
            --graph_cross_label_fraction 0.25 \
            --graph_num_graphs 6 \
            --recompute False \
            --train_col_embed_only True \
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