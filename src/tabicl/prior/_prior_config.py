from ._activations import get_activations


DEFAULT_FIXED_HP = {
    # SCMPrior
    "mix_probs": (0.7, 0.3),
    # TreeSCM
    "tree_model": "xgboost",
    "tree_depth_lambda": 0.5,
    "tree_n_estimators_lambda": 0.5,
    # Reg2Cls
    "balanced": True,
    "multiclass_ordered_prob": 0.0,
    "multiclass_jitter_scale": 0.0,
    "cat_prob": 0.2,
    "max_categories": float("inf"),
    "scale_by_max_features": False,
    "permute_features": True,
    "permute_labels": True,
    # GraphPrior v2 distributions.
    "graph_homophily_prob": 0.5,
    "graph_transition_scope_prob": 0.5,
    "graph_remain_prob_range": (0.7, 0.99),
    "graph_heterophily_structure_probs": {"bipartite": 1 / 3, "cyclic": 1 / 3, "grouped": 1 / 3},
    "graph_group_count_range": (2, 4),
    "graph_v1_prob": 0.4,
    "graph_v2_prob": 0.4,
    "graph_prob": 0.2,
    #"multiclass_type": "quantile",
    #"is_causal": True,
}

DEFAULT_SAMPLED_HP = {
    # Reg2Cls
    "multiclass_type": {"distribution": "meta_choice", "choice_values": ["value", "rank", "quantile"]},
    # MLPSCM
    "mlp_activations": {
        "distribution": "meta_choice_mixed",
        "choice_values": get_activations(random=True, scale=True, diverse=True),
    },
    "block_wise_dropout": {"distribution": "meta_choice", "choice_values": [True, False]},
    #"mlp_dropout_prob": {"distribution": "meta_beta", "scale": 0.9, "min": 0.1, "max": 5.0},
    # MLPSCM and TreeSCM
    "is_causal": {"distribution": "meta_choice", "choice_values": [True, False]},
    "num_causes": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 12,
        "min_mean": 1,
        "round": True,
        "lower_bound": 1,
    },
    "y_is_effect": {"distribution": "meta_choice", "choice_values": [True, False]},
    "in_clique": {"distribution": "meta_choice", "choice_values": [True, False]},
    "sort_features": {"distribution": "meta_choice", "choice_values": [True, False]},
    "num_layers": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 4,
        "min_mean": 1,
        "round": True,
        "lower_bound": 2,
    },
    "hidden_dim": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 130,
        "min_mean": 5,
        "round": True,
        "lower_bound": 4,
    },
    "init_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10.0,
        "min_mean": 0.01,
        "round": False,
        "lower_bound": 0.0,
    },
    "noise_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 0.3,
        "min_mean": 0.0001,
        "round": False,
        "lower_bound": 0.0,
    },
    "sampling": {"distribution": "meta_choice", "choice_values": ["normal", "mixed", "uniform"]},
    "pre_sample_cause_stats": {"distribution": "meta_choice", "choice_values": [True, False]},
    "pre_sample_noise_std": {"distribution": "meta_choice", "choice_values": [True, False]},
}
