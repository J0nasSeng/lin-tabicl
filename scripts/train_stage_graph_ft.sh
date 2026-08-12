# Fine-tune the Stage 2 checkpoint on a mixed tabular/graph prior.

# Choose ICL backbone: graph or encoder
ICL_BACKEND=${ICL_BACKEND:-graph}
# Enable wandb logging by setting WAND_LOG=True (and optionally WAND_MODE=online)
WAND_LOG=${WAND_LOG:-True}
WAND_MODE=${WAND_MODE:-online}
# GPU selection controls
DEVICE=${DEVICE:-cuda}
NUM_GPUS=${NUM_GPUS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5}

# Stage 2 is used as the initialization checkpoint. The source checkpoint is
# never written to: --only_load_model resets training state and all new
# checkpoints are written below STAGE_GRAPH_FT_CHECKPOINT_DIR.
STAGE2_CHECKPOINT_DIR=${STAGE2_CHECKPOINT_DIR:-/workspace/checkpoints_dyngraph_intraclass=0.25/stage2}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-$(find "${STAGE2_CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'step-*.ckpt' -printf '%f\n' | sort -V | tail -n 1)}
STAGE_GRAPH_FT_CHECKPOINT_DIR=${STAGE_GRAPH_FT_CHECKPOINT_DIR:-/workspace/checkpoints_dyngraph_intraclass=0.25/stage_graph_ft}

if [[ -z "${STAGE2_CHECKPOINT}" || ! -f "${STAGE2_CHECKPOINT_DIR}/${STAGE2_CHECKPOINT}" && ! -f "${STAGE2_CHECKPOINT}" ]]; then
    echo "No Stage 2 checkpoint found in ${STAGE2_CHECKPOINT_DIR}" >&2
    exit 1
fi

if [[ -f "${STAGE2_CHECKPOINT_DIR}/${STAGE2_CHECKPOINT}" ]]; then
    STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT_DIR}/${STAGE2_CHECKPOINT}"
fi
mkdir -p "${STAGE_GRAPH_FT_CHECKPOINT_DIR}"
# Confusion matrix logging interval
LOG_CONF_MAT_EVERY=${LOG_CONF_MAT_EVERY:-200000} # effectively disables confusion matrix logging if set to a large number

export CUDA_VISIBLE_DEVICES

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------

torchrun --standalone --nproc_per_node=${NUM_GPUS} /workspace/src/tabicl/train/_run.py \
            --wandb_log ${WAND_LOG} \
            --wandb_project TabICL \
            --wandb_name GraphFT \
            --wandb_dir /workspace/wandb/ \
            --wandb_mode ${WAND_MODE} \
            --device ${DEVICE} \
            --dtype bfloat16 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 3000 \
            --batch_size 512 \
            --micro_batch_size 4 \
            --log_conf_mat_every ${LOG_CONF_MAT_EVERY} \
            --lr 2e-5 \
            --weight_decay 3e-5 \
            --supcon_weight 0.1 \
            --entropy_weight 0.0 \
            --icl_decoder_type soft_kmeans \
            --scheduler polynomial_decay_warmup \
            --warmup_proportion 0.02 \
            --gradient_clipping 2.0 \
            --prior_type nanotabicl \
            --prior_device cpu \
            --normalization std \
            --batch_size_per_gp 8 \
            --min_features 2 \
            --max_features 100 \
            --max_classes 10 \
            --max_seq_len 1024 \
            --min_train_size 0.1 \
            --max_train_size 0.6 \
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
            --checkpoint_dir ${STAGE_GRAPH_FT_CHECKPOINT_DIR} \
            --checkpoint_path ${STAGE2_CHECKPOINT} \
            --only_load_model True \
            --save_temp_every 1000 \
            --save_perm_every 5000 \
            --icl_soft_kmeans_temperature 0.5 \
            --label_smoothing 0.1 \
            --graph_min_train_neighbors 4 \
            --graph_max_train_neighbors 4 \
            --graph_train_neighbors_per_test 2 \
            --graph_cross_label_fraction 0.25 \
            --graph_num_graphs 6 \
            --tab_graphs v2 \
            --mode_prob 0.3 \
            --recompute False \
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