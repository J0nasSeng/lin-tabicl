"""Download and unpack the GraphLand dataset."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import kagglehub


DATASET = "bazhenovgleb/graphland"


def _unzip_recursively(root: Path) -> None:
	"""Extract every zip archive found below ``root``.

	Archives may contain further archives, so extraction is repeated until no
	zip files remain. Existing files are replaced to make reruns deterministic.
	"""
	while True:
		archives = sorted(root.rglob("*.zip"))
		if not archives:
			return

		for archive in archives:
			extract_dir = archive.with_suffix("")
			extract_dir.mkdir(parents=True, exist_ok=True)
			with zipfile.ZipFile(archive) as zipped:
				zipped.extractall(extract_dir)
			archive.unlink()


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"output_dir",
		type=Path,
		help="Directory in which to store the downloaded and unpacked dataset",
	)
	args = parser.parse_args()

	output_dir = args.output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	# KaggleHub manages its own cache. Copy the downloaded contents into the
	# caller-selected directory so the final dataset location is predictable.
	downloaded_dir = Path(kagglehub.dataset_download(DATASET))
	for source in downloaded_dir.iterdir():
		destination = output_dir / source.name
		if source.is_dir():
			shutil.copytree(source, destination, dirs_exist_ok=True)
		else:
			shutil.copy2(source, destination)

	_unzip_recursively(output_dir)
	print(f"Dataset stored in: {output_dir}")


if __name__ == "__main__":
	main()
