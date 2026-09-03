"""TALENT adapter for the TabICL graph-1d classifier."""

from __future__ import annotations

import inspect
import time

import numpy as np
import torch
import torch.nn.functional as F

from TALENT.model.methods.base import Method
from TALENT.model.lib.data import (
	Dataset,
	data_enc_process,
	data_label_process,
	data_nan_process,
)


class TabICLGraphMethod(Method):
	"""TALENT method wrapper for a graph-1d TabICL checkpoint."""

	def __init__(self, args, is_regression):
		if is_regression:
			raise ValueError("TabICLGraphMethod supports classification only")
		requested_device = getattr(args, "tabicl_device", None)
		super().__init__(args, is_regression)
		if requested_device is not None:
			self.args.device = requested_device
		if args.normalization != "none":
			raise ValueError("TabICLGraphMethod requires normalization='none'")
		if args.cat_policy != "indices":
			raise ValueError("TabICLGraphMethod requires cat_policy='indices'")
		if args.num_policy != "none":
			raise ValueError("TabICLGraphMethod requires num_policy='none'")
		if args.tune is True:
			raise ValueError("TabICLGraphMethod does not support tuning")

	def data_format(self, is_train=True, N=None, C=None, y=None):
		if is_train:
			self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(
				self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy
			)
			self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
			self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(
				self.N, self.C, self.args.cat_policy
			)
			self.criterion = F.cross_entropy
			return

		N_test, C_test, _, _, _ = data_nan_process(
			N,
			C,
			self.args.num_nan_policy,
			self.args.cat_nan_policy,
			self.num_new_value,
			self.imputer,
			self.cat_new_value,
		)
		N_test, C_test, _, _, _ = data_enc_process(
			N_test,
			C_test,
			self.args.cat_policy,
			None,
			self.ord_encoder,
			self.mode_values,
			self.cat_encoder,
		)
		y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)
		if N_test is not None and C_test is not None:
			self.N_test, self.C_test = N_test["test"], C_test["test"]
		elif N_test is None and C_test is not None:
			self.N_test, self.C_test = None, C_test["test"]
		else:
			self.N_test, self.C_test = N_test["test"], None
		self.y_test = y_test["test"]

	def construct_model(self, model_config=None, cat_indices=None):
		try:
			from tabicl import TabICLClassifier
		except ImportError as error:
			raise ImportError(
				"TabICLGraphMethod requires the local `tabicl` package."
			) from error

		general = self.args.config.get("general", {}) or {}
		common = {
			"device": self.args.device,
			"random_state": self.args.seed,
			"n_estimators": general.get("n_estimators", 8),
			"batch_size": general.get("batch_size", 1),
			"use_amp": general.get("use_amp", True),
			"allow_auto_download": general.get("allow_auto_download", False),
			"verbose": general.get("verbose", False),
			"gat_mode": "ensemble",
			"gat_num_iterations": general.get("gat_num_iterations", 1),
			"gat_entry_layer": general.get("gat_entry_layer"),
			"max_chunk_size": general.get("max_chunk_size"),
			"decoder_chunk_size": general.get("decoder_chunk_size", 5000),
		}
		for key in (
			"checkpoint_version",
			"model_path",
			"norm_methods",
			"feat_shuffle_method",
			"outlier_threshold",
			"inference_config",
			"offload_mode",
			"use_fa3",
		):
			if key in general:
				common[key] = general[key]
		accepted = set(inspect.signature(TabICLClassifier.__init__).parameters)
		self.model = TabICLClassifier(**{key: value for key, value in common.items() if key in accepted})

	def fit(self, data, info, train=True, config=None):
		N, C, y = data
		self.D = Dataset(N, C, y, info)
		self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
		self.is_binclass, self.is_multiclass, self.is_regression = (
			self.D.is_binclass,
			self.D.is_multiclass,
			self.D.is_regression,
		)
		self.data_format(is_train=True)

		sampled_Y = self.y["train"]
		if self.N is not None and self.C is not None:
			sampled_X = np.concatenate((self.N["train"], self.C["train"]), axis=1)
		elif self.N is None and self.C is not None:
			sampled_X = self.C["train"]
		else:
			sampled_X = self.N["train"]
		sampled_X, sampled_Y = self.subsample_train_rows(sampled_X, sampled_Y)
		self.sampled_X, self.sampled_Y = sampled_X, sampled_Y
		self.construct_model()

		started = time.time()
		self.model.fit(self.sampled_X, self.sampled_Y)
		self.fit_time = time.time() - started

	def predict(self, data, info, model_name):
		N, C, y = data
		self.data_format(False, N, C, y)
		if self.N_test is not None and self.C_test is not None:
			test_X = np.concatenate((self.N_test, self.C_test), axis=1)
		elif self.N_test is None and self.C_test is not None:
			test_X = self.C_test
		else:
			test_X = self.N_test

		started = time.time()
		test_proba = np.asarray(self.model.predict_proba(test_X), dtype=np.float32)
		self.predict_time = time.time() - started
		test_label = np.asarray(self.y_test)
		loss = self.criterion(torch.log(torch.from_numpy(test_proba).clamp_min(1e-12)), torch.from_numpy(test_label)).item()
		results, metric_names = self.metric(test_proba, test_label, self.y_info)

		print("Test: loss={:.4f}".format(loss))
		for name, result in zip(metric_names, results):
			print("[{}]={:.4f}".format(name, result))
		return loss, results, metric_names, test_proba


TabICLGraph1DMethod = TabICLGraphMethod
Model = TabICLGraphMethod

__all__ = ["Model", "TabICLGraph1DMethod", "TabICLGraphMethod"]
