import argparse


def str2bool(value: str) -> bool:
    return value.lower() == "true"


def train_size_type(value: str):
    parsed = float(value)
    if 0 < parsed < 1:
        return parsed
    if parsed.is_integer():
        return int(parsed)
    raise argparse.ArgumentTypeError(
        "Train size must be either an integer (absolute position) or a float between 0 and 1"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a checkpointed TabICL model on prior datasets")

    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cpu", help="Evaluation device")
    parser.add_argument("--num_datasets", type=int, default=64, help="Number of datasets to evaluate")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Number of datasets sampled per prior draw")
    parser.add_argument("--np_seed", type=int, default=42, help="NumPy random seed")
    parser.add_argument("--torch_seed", type=int, default=42, help="Torch random seed")
    parser.add_argument("--output_figure_path", type=str, default="eval_overview.png", help="Output PNG path")

    parser.add_argument("--prior_type", type=str, default="nanotabicl", help="Prior type")
    parser.add_argument("--prior_device", type=str, default="cpu", help="Prior generation device")
    parser.add_argument("--batch_size_per_gp", type=int, default=4, help="Prior batch size per group")
    parser.add_argument("--min_features", type=int, default=2, help="Minimum sampled features")
    parser.add_argument("--max_features", type=int, default=10, help="Maximum sampled features")
    parser.add_argument("--max_classes", type=int, default=10, help="Maximum classes in prior")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum sequence length")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--log_seq_len", type=str2bool, default=False, help="Log-uniform sequence sampling")
    parser.add_argument("--seq_len_per_gp", type=str2bool, default=False, help="Sample sequence length per group")
    parser.add_argument(
        "--min_train_size",
        type=train_size_type,
        default=0.1,
        help="Minimum train split (int or ratio)",
    )
    parser.add_argument(
        "--max_train_size",
        type=train_size_type,
        default=0.9,
        help="Maximum train split (int or ratio)",
    )
    parser.add_argument("--replay_small", type=str2bool, default=False, help="Replay smaller sequence lengths")

    return parser
