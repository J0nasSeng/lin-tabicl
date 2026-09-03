"""Local TALENT method specification for the graph-1d TabICL adapter."""

from dataclasses import dataclass
from enum import Enum
from importlib import import_module


class Architecture(str, Enum):
	DEEP = "deep"


class Hardware(str, Enum):
	GPU = "gpu"


class OutputType(str, Enum):
	PROBABILITIES = "probabilities"


@dataclass(frozen=True)
class MethodSpec:
	name: str
	module: str
	class_name: str
	architecture: Architecture
	hardware: Hardware
	output_type: OutputType
	cat_policy: tuple[str, ...] | None = None
	normalization: str | None = None
	num_policy: str | None = None
	supports_hpo: bool = True
	supports_regression: bool = True
	supports_classification: bool = True
	train_row_limit: int | None = None
	notes: str = ""

	def get_class(self):
		return getattr(import_module(self.module), self.class_name)

	def validate_args(self, args):
		if self.cat_policy is not None and args.cat_policy not in self.cat_policy:
			raise ValueError(f"cat_policy must be one of {self.cat_policy}")
		if self.normalization is not None and args.normalization != self.normalization:
			raise ValueError(f"normalization must be {self.normalization!r}")
		if self.num_policy is not None and args.num_policy != self.num_policy:
			raise ValueError(f"num_policy must be {self.num_policy!r}")
		if not self.supports_hpo and args.tune:
			raise ValueError(f"{self.name} does not support HPO")
		if not self.supports_regression and getattr(args, "task_type", None) == "regression":
			raise ValueError(f"{self.name} does not support regression")


TABICL_GRAPH_1D_SPEC = MethodSpec(
	name="tabicl_graph_1d",
	module="tabicl.evaluate.benchmarks.talent.model",
	class_name="TabICLGraphMethod",
	architecture=Architecture.DEEP,
	hardware=Hardware.GPU,
	output_type=OutputType.PROBABILITIES,
	cat_policy=("indices",),
	normalization="none",
	num_policy="none",
	supports_hpo=False,
	supports_regression=False,
	train_row_limit=1_000_000,
	notes="TabICL graph-1d backend; classification only.",
)


def get_method_spec(model_type: str = "tabicl_graph_1d") -> MethodSpec:
	if model_type != TABICL_GRAPH_1D_SPEC.name:
		raise KeyError(f"Unknown local TALENT model: {model_type}")
	return TABICL_GRAPH_1D_SPEC


def get_method(model_type: str = "tabicl_graph_1d"):
	return get_method_spec(model_type).get_class()
