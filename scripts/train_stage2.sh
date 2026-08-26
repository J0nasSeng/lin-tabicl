# This script is used to train TabICL for the second stage of the curriculum learning

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

# Stage 2 starts from the latest Stage 1 model and writes its checkpoints to a
# separate directory. Override these variables to use a different run layout.
STAGE1_CHECKPOINT_DIR=${STAGE1_CHECKPOINT_DIR:-/workspace/checkpoints_dyngraph_intraclass=0.25/stage1}
STAGE2_CHECKPOINT_DIR=${STAGE2_CHECKPOINT_DIR:-/workspace/checkpoints_dyngraph_intraclass=0.25/stage2}
STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-$(find "${STAGE1_CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'step-*.ckpt' -printf '%f\n' | sort -V | tail -n 1)}

if [[ -z "${STAGE1_CHECKPOINT}" || ! -f "${STAGE1_CHECKPOINT_DIR}/${STAGE1_CHECKPOINT}" && ! -f "${STAGE1_CHECKPOINT}" ]]; then
    echo "No Stage 1 checkpoint found in ${STAGE1_CHECKPOINT_DIR}" >&2
    exit 1
fi

if [[ -f "${STAGE1_CHECKPOINT_DIR}/${STAGE1_CHECKPOINT}" ]]; then
    STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT_DIR}/${STAGE1_CHECKPOINT}"
fi
mkdir -p "${STAGE2_CHECKPOINT_DIR}"

export CUDA_VISIBLE_DEVICES

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------

torchrun --standalone --nproc_per_node=${NUM_GPUS} /workspace/src/tabicl/train/_run.py \
            --wandb_log ${WAND_LOG} \
            --wandb_project TabICL \
            --wandb_name Stage2 \
            --wandb_dir /workspace/wandb/ \
            --wandb_mode ${WAND_MODE} \
            --device ${DEVICE} \
            --dtype bfloat16 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 2000 \
            --batch_size 8 \
            --micro_batch_size 1 \
            --log_conf_mat_every ${LOG_CONF_MAT_EVERY} \
            --lr 2e-5 \
            --supcon_weight 0.1 \
            --weight_decay 3e-5 \
            --scheduler polynomial_decay_warmup \
            --icl_decoder_type soft_kmeans \
            --warmup_proportion 0 \
            --poly_decay_lr_end 5e-6 \
            --poly_decay_power 2.0 \
            --gradient_clipping 1.0 \
            --prior_type nanotabicl \
            --prior_device cpu \
            --batch_size_per_gp 1 \
            --min_features 2 \
            --max_features 100 \
            --max_classes 10 \
            --min_seq_len 1000 \
            --max_seq_len 10000 \
            --log_seq_len True \
            --seq_len_per_gp True \
            --min_train_size 0.2 \
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
            --checkpoint_dir ${STAGE2_CHECKPOINT_DIR} \
            --checkpoint_path ${STAGE1_CHECKPOINT} \
            --graph_min_train_neighbors 4 \
            --graph_max_train_neighbors 4 \
            --graph_train_neighbors_per_test 2 \
            --graph_cross_label_fraction 0.25 \
            --graph_num_graphs 6 \
            --save_temp_every 5 \
            --save_perm_every 100 \
            --only_load_model True \
            --recompute True


# ------------------------------------------------------
# Save prior datasets to disk and load them for training
# ------------------------------------------------------

# Saving to disk
#python /workspace/src/tabicl/prior/_genload.py \
#    --save_dir /workspace/prior/stage2/ \
#    --np_seed 42 \
#    --torch_seed 42 \
#    --num_batches 2000 \
#    --resume_from 0 \
#    --batch_size 512 \
#    --batch_size_per_gp 2 \
#    --prior_type mix_scm \
#    --min_features 2 \
#    --max_features 100 \
#    --max_classes 10 \
#    --min_seq_len 1000 \
#    --max_seq_len 40000 \
#    --log_seq_len True \
#    --seq_len_per_gp True \
#    --min_train_size 0.5 \
#    --max_train_size 0.9 \
#    --n_jobs -1 \
#    --num_threads_per_generate 1 \
#    --device cpu

# Loading from disk and training
#torchrun --standalone --nproc_per_node=${NUM_GPUS} /workspace/src/tabicl/train/_run.py \
#            --wandb_log ${WAND_LOG} \
#            --wandb_project TabICL \
#            --wandb_name Stage2 \
#            --wandb_dir /workspace/wandb/ \
#            --wandb_mode ${WAND_MODE} \
#            --device ${DEVICE} \
#            --dtype float32 \
#            --np_seed 42 \
#            --torch_seed 42 \
#            --max_steps 2000 \
#            --batch_size 512 \
#            --micro_batch_size 1 \
#                --log_conf_mat_every ${LOG_CONF_MAT_EVERY} \
#            --lr 2e-5 \
#            --scheduler polynomial_decay_warmup \
#            --warmup_proportion 0 \
#            --poly_decay_lr_end 5e-6 \
#            --poly_decay_power 2.0 \
#            --gradient_clipping 1.0 \
#            --prior_dir /my/stage2/prior/dir \
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
#            --checkpoint_dir ${STAGE2_CHECKPOINT_DIR} \
#            --checkpoint_path ${STAGE1_CHECKPOINT} \
#            --save_temp_every 5 \
#            --save_perm_every 100 \
#            --only_load_model True