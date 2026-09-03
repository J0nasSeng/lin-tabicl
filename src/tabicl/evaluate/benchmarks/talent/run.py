"""Run the local TabICL graph-1d adapter with the TALENT benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tqdm import tqdm
from TALENT.model.lib.data import get_dataset
from TALENT.model.lib.evaluation import evaluate
from TALENT.model.utils import get_deep_args, set_seeds, show_results

from .spec import get_method, get_method_spec


MODEL_TYPE = "tabicl_graph_1d"
DATASET_TYPES = {
	"binary_clf": 2,
	"multi_clf": "multi",
}


def _take_option(option_names):
	for option in option_names:
		if option in sys.argv:
			index = sys.argv.index(option)
			if index + 1 >= len(sys.argv):
				raise ValueError(f"{option} requires a value")
			value = sys.argv[index + 1]
			del sys.argv[index:index + 2]
			return value
	return None


def _find_datasets(dataset_path, dataset_type):
	"""Return dataset directories whose info.json matches the requested type."""
	target = DATASET_TYPES[dataset_type]
	datasets = []
	for directory in sorted(Path(dataset_path).iterdir()):
		info_path = directory / "info.json"
		if not directory.is_dir() or not info_path.is_file():
			continue
		try:
			info = json.loads(info_path.read_text())
		except (OSError, json.JSONDecodeError) as error:
			raise ValueError(f"Could not read dataset metadata: {info_path}") from error
		n_classes = info.get("n_classes")
		matches = n_classes == target if target == 2 else isinstance(n_classes, int) and n_classes > 2
		if matches:
			datasets.append(directory.name)
	if not datasets:
		raise ValueError(f"No datasets found for type {dataset_type!r} in {dataset_path}")
	return datasets


def _get_args():
	"""Reuse TALENT's CLI parser while allowing our local model name."""
	argv = sys.argv[:]
	dataset_was_specified = any(
		argument == "--dataset" or argument.startswith("--dataset=")
		for argument in sys.argv
	)
	checkpoint_path = _take_option(("--checkpoint_path", "--checkpoint-path"))
	device = _take_option(("--device",))
	dataset_type = _take_option(("--dataset-type", "--dataset_type"))
	if dataset_type is not None and dataset_type not in DATASET_TYPES:
		raise ValueError(
			f"Unknown dataset type {dataset_type!r}; choose from {tuple(DATASET_TYPES)}"
		)
	if dataset_type is not None and dataset_was_specified:
		raise ValueError("--dataset and --dataset-type are mutually exclusive")
	continue_on_error = "--continue-on-error" in sys.argv
	if continue_on_error:
		sys.argv.remove("--continue-on-error")
	try:
		if "--model_type" in sys.argv:
			index = sys.argv.index("--model_type") + 1
			if index < len(sys.argv) and sys.argv[index] == MODEL_TYPE:
				sys.argv[index] = "tabicl_v2"
		args, _, _ = get_deep_args()
	finally:
		sys.argv[:] = argv

	args.model_type = MODEL_TYPE
	args.cat_policy = "indices"
	args.normalization = "none"
	args.num_policy = "none"
	args.tabicl_device = device
	args.dataset_type = dataset_type
	args.continue_on_error = continue_on_error
	args.config = {
		"general": {
			"model_path": checkpoint_path or args.model_path,
			"allow_auto_download": False,
			"n_estimators": 8,
		}
	}
	return args


def _run_dataset(args, spec, dataset_name):
	args.dataset = dataset_name
	train_val_data, test_data, info = get_dataset(dataset_name, args.dataset_path)
	spec.validate_args(args)

	loss_list, results_list, time_list = [], [], []
	for seed in tqdm(range(args.seed_num), desc=dataset_name):
		args.seed = seed
		set_seeds(args.seed)
		method = get_method(MODEL_TYPE)(args, info["task_type"] == "regression")
		method.fit(train_val_data, info)
		eval_result = evaluate(
			method,
			train_val_data,
			test_data,
			info,
			model_name=args.evaluate_option,
			output_type=spec.output_type,
			tune_threshold=True,
		)
		loss_list.append(eval_result["loss"])
		results_list.append(eval_result["metrics"])
		metric_name = eval_result["metric_names"]
		time_list.append((method.fit_time, method.predict_time))

	show_results(args, info, metric_name, loss_list, results_list, time_list)


if __name__ == "__main__":
	args = _get_args()
	spec = get_method_spec(MODEL_TYPE)
	if args.dataset_type is not None:
		datasets = _find_datasets(args.dataset_path, args.dataset_type)
	else:
		datasets = [args.dataset]

	for dataset_name in datasets:
		try:
			_run_dataset(args, spec, dataset_name)
		except Exception:
			if not args.continue_on_error:
				raise
			print(f"Skipping failed dataset: {dataset_name}", file=sys.stderr)
