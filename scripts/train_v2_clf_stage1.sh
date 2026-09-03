#!/bin/bash
# TabICLv2 pre-training -- CLASSIFIER, Stage 1 (of 3).
# (The regressor is a separate checkpoint; see train_v2_reg_stage{1,2,3}.sh.)
#
# Recipe from "TabICLv2: A better, faster, scalable, and open tabular foundation
# model" (https://arxiv.org/abs/2602.11139), §4.1 / Appendix B.1. Three-stage
# curriculum that progressively increases dataset size (batch size 64 throughout):
#
#   Stage 1: 500K steps, 1,024 samples/dataset,       30-90% train, max LR 8e-4, grad clip 10
#   Stage 2:  40K steps, 400-10,240 samples (log-u), 79-81% train, max LR 1e-4, grad clip 10
#   Stage 3:  10K steps, 400-60,000 samples (log-u), 79-81% train, max LR 2e-5, grad clip  1
#
# Common: Muon optimizer, cautious weight decay parameter 0.01 (see note below),
# cosine schedule, AMP, per-micro-batch train/test sizes (--seq_len_per_gp), up to
# 100 features, 8 attention heads everywhere, graph_scm prior. AMP is always used.
#
# The classifier uses LayerNorm WITH biases (the --norm_type default; paper Table A.1),
# unlike the regressor, which uses bias-free LayerNorm (--norm_type layernorm_nobias).
# v2 uses standard residual init (--zero_init False; v1 zero-initializes residual
# branches) and enables FlashAttention-3 only in stages 2 & 3 (--use_flash_attn3).
#
# Runtime configuration can be overridden from the environment, for example:
#   ICL_BACKEND=encoder WAND_LOG=True DEVICE=cuda:1 NUM_GPUS=1 ./train_v2_clf_stage1.sh

ICL_BACKEND=${ICL_BACKEND:-graph-1d}
WAND_LOG=${WAND_LOG:-True}
WAND_MODE=${WAND_MODE:-online}
WAND_DIR=${WAND_DIR:-/workspace/wandb/}
DEVICE=${DEVICE:-cuda}
NUM_GPUS=${NUM_GPUS:-2}                       # the paper used 4 GPUs for pre-training
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
LOG_CONF_MAT_EVERY=${LOG_CONF_MAT_EVERY:-100000000} # no confusion matrix logging in stage 1

export CUDA_VISIBLE_DEVICES

CKPT_DIR=/workspace/checkpoints/tabiclv2-clf-graph-1d/stage1

torchrun --standalone --nproc_per_node=$NUM_GPUS -m tabicl.train \
            --wandb_log ${WAND_LOG} \
            --wandb_project TabICLv2 \
            --wandb_name tabiclv2_clf_stage1 \
            --wandb_dir ${WAND_DIR} \
            --wandb_mode ${WAND_MODE} \
            --device ${DEVICE} \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 500000 \
            --batch_size 64 \
            --micro_batch_size 2 \
            --log_conf_mat_every ${LOG_CONF_MAT_EVERY} \
            --lr 8e-4 \
            --muon True \
            --beta1 0.9 \
            --weight_decay 0.01 \
            --use_cautious_wd False \
            --scheduler cosine_with_restarts \
            --warmup_proportion 0.01 \
            --cosine_num_cycles 1 \
            --cosine_amplitude_decay 1 \
            --cosine_lr_end 1e-7 \
            --gradient_clipping 10.0 \
            --prior_type graph_scm \
            --prior_device cpu \
            --icl_backend ${ICL_BACKEND} \
            --n_jobs 16 \
            --batch_size_per_gp 4 \
            --min_features 1 \
            --max_features 256 \
            --max_classes 10 \
            --max_seq_len 1024 \
            --min_train_size 0.3 \
            --max_train_size 0.9 \
            --seq_len_per_gp False \
            --graph_noise False \
            --filter_unpredictable_graphs True \
            --filter_unpredictable_datasets True \
            --allow_act_warping False \
            --min_n_nodes 2 \
            --max_n_nodes 32 \
            --cauchy_dag_offset 0.0 \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 8 \
            --col_num_inds 128 \
            --col_affine False \
            --col_feature_group same \
            --col_feature_group_size 3 \
            --col_target_aware True \
            --col_ssmax True \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --row_rope_interleaved False \
            --icl_num_blocks 12 \
            --icl_nhead 8 \
            --icl_ssmax True \
            --ssmax_type qassmax-mlp-elementwise \
            --ff_factor 2 \
            --norm_first True \
            --zero_init False \
            --use_flash_attn3 False \
            --checkpoint_dir $CKPT_DIR \
            --save_temp_every 1000 \
            --save_perm_every 5000 \
            --graph_num_graphs 6

# Note on cautious weight decay: the paper reports using cautious weight decay (parameter
# 0.01), but the reference pretraining runs that produced the released checkpoints
# did NOT actually pass it to Muon (it was left unwired), so --use_cautious_wd is set
# to False here to reproduce that behavior. Set it to True to enable it as described
# in the paper. --weight_decay 0.01 is the decay strength either way.
