# This script is used to train TabICL for the third stage of the curriculum learning

# Choose ICL backbone: graph or encoder
ICL_BACKEND=${ICL_BACKEND:-graph-1d}
# Enable wandb logging by setting WAND_LOG=True (and optionally WAND_MODE=online)
WAND_LOG=${WAND_LOG:-True}
WAND_MODE=${WAND_MODE:-online}
# GPU selection controls
DEVICE=${DEVICE:-cuda}
NUM_GPUS=${NUM_GPUS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4}
# Confusion matrix logging interval
LOG_CONF_MAT_EVERY=${LOG_CONF_MAT_EVERY:-100}

# Stage 3 starts from the latest Stage 1 or 2 checkpoint and writes to a separate
# directory. Override these variables to use a different run layout.
STAGE2_CHECKPOINT_DIR=${STAGE2_CHECKPOINT_DIR:-/workspace/checkpoints_dyngraph_intraclass=0.25_no_supcon_4_graphs/stage1}
STAGE3_CHECKPOINT_DIR=${STAGE3_CHECKPOINT_DIR:-/workspace/checkpoints_dyngraph_intraclass=0.25_no_supcon_4_graphs/stage3}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-$(find "${STAGE2_CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'step-*.ckpt' -printf '%f\n' | sort -V | tail -n 1)}

if [[ -z "${STAGE2_CHECKPOINT}" || ! -f "${STAGE2_CHECKPOINT_DIR}/${STAGE2_CHECKPOINT}" && ! -f "${STAGE2_CHECKPOINT}" ]]; then
    echo "No Stage 2 checkpoint found in ${STAGE2_CHECKPOINT_DIR}" >&2
    exit 1
fi

if [[ -f "${STAGE2_CHECKPOINT_DIR}/${STAGE2_CHECKPOINT}" ]]; then
    STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT_DIR}/${STAGE2_CHECKPOINT}"
fi
mkdir -p "${STAGE3_CHECKPOINT_DIR}"

export CUDA_VISIBLE_DEVICES

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------

torchrun --standalone --nproc_per_node=${NUM_GPUS} /workspace/src/tabicl/train/_run.py \
            --wandb_log ${WAND_LOG} \
            --wandb_project TabICL \
            --wandb_name Stage3 \
            --wandb_dir /workspace/wandb/ \
            --wandb_mode ${WAND_MODE} \
            --device ${DEVICE} \
            --dtype bfloat16 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 50 \
            --batch_size 512 \
            --micro_batch_size 1 \
            --log_conf_mat_every ${LOG_CONF_MAT_EVERY} \
            --lr 2e-6 \
            --scheduler constant \
            --gradient_clipping 1.0 \
            --prior_type nanotabicl \
            --prior_device cpu \
            --batch_size_per_gp 1 \
            --min_features 2 \
            --max_features 256 \
            --max_classes 10 \
            --min_seq_len 40000 \
            --max_seq_len 60000 \
            --log_seq_len True \
            --seq_len_per_gp True \
            --replay_small True \
            --icl_decoder_type soft_kmeans \
            --normalization std \
            --min_train_size 0.5 \
            --max_train_size 0.9 \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 8 \
            --col_num_inds 128 \
            --freeze_col True \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --freeze_row True \
            --icl_num_blocks 4 \
            --icl_nhead 8 \
            --icl_backend ${ICL_BACKEND} \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_dir ${STAGE3_CHECKPOINT_DIR} \
            --checkpoint_path ${STAGE2_CHECKPOINT} \
            --save_temp_every 10 \
            --save_perm_every 50 \
            --only_load_model True \
            --supcon_weight 0.0 \
            --entropy_weight 0.0 \
            --weight_decay 1e-4 \
            --icl_soft_kmeans_temperature 0.5 \
            --label_smoothing 0.0 \
            --graph_min_train_neighbors 4 \
            --graph_max_train_neighbors 4 \
            --graph_train_neighbors_per_test 2 \
            --graph_cross_label_fraction 0.25 \
            --graph_num_graphs 4 \
            --recompute False


# ------------------------------------------------------
# Save prior datasets to disk and load them for training
# ------------------------------------------------------

# Saving to disk
#python /path/to/tabicl/prior/genload.py \
#    --save_dir /my/stage3/prior/dir \
#    --np_seed 42 \
#    --torch_seed 42 \
#    --num_batches 50 \
#    --resume_from 0 \
#    --batch_size 512 \
#    --batch_size_per_gp 1 \
#    --prior_type mix_scm \
#    --min_features 2 \
#    --max_features 100 \
#    --max_classes 10 \
#    --min_seq_len 40000 \
#    --max_seq_len 60000 \
#    --log_seq_len True \
#    --seq_len_per_gp True \
#    --replay_small True \
#    --min_train_size 0.5 \
#    --max_train_size 0.9 \
#    --n_jobs -1 \
#    --num_threads_per_generate 1 \
#    --device cpu
#
## Loading from disk and training
#torchrun --standalone --nproc_per_node=${NUM_GPUS} /path/to/tabicl/train/run.py \
#            --wandb_log ${WAND_LOG} \
#            --wandb_project TabICL \
#            --wandb_name Stage3 \
#            --wandb_dir /my/wandb/dir \
#            --wandb_mode ${WAND_MODE} \
#            --device ${DEVICE} \
#            --dtype float32 \
#            --np_seed 42 \
#            --torch_seed 42 \
#            --max_steps 50 \
#            --batch_size 512 \
#            --micro_batch_size 1 \
#                --log_conf_mat_every ${LOG_CONF_MAT_EVERY} \
#            --lr 2e-6 \
#            --scheduler constant \
#            --gradient_clipping 1.0 \
#            --prior_dir /my/stage3/prior/dir \
#            --load_prior_start 0 \
#            --delete_after_load False \
#            --prior_device cpu \
#            --embed_dim 128 \
#            --col_num_blocks 3 \
#            --col_nhead 8 \
#            --col_num_inds 128 \
#            --freeze_col True \
#            --row_num_blocks 3 \
#            --row_nhead 8 \
#            --row_num_cls 4 \
#            --row_rope_base 100000 \
#            --freeze_row True \
#            --icl_num_blocks 4 \
#            --icl_nhead 8 \
#            --icl_backend ${ICL_BACKEND} \
#            --ff_factor 2 \
#            --norm_first True \
#            --checkpoint_dir ${STAGE3_CHECKPOINT_DIR} \
#            --checkpoint_path ${STAGE2_CHECKPOINT} \
#            --save_temp_every 1000 \
#            --save_perm_every 5000 \
#            --only_load_model True \
#            --graph_min_train_neighbors 4 \
#            --graph_max_train_neighbors 4 \
#            --graph_train_neighbors_per_test 2 \
#            --graph_cross_label_fraction 0.25 \
#            --graph_num_graphs 4 \
#            --recompute False