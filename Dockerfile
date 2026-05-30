FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
RUN apt update && apt install -y --no-install-recommends build-essential && apt install -y git

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Copy project files (optional) and set default command
COPY . /app

WORKDIR /app
RUN uv sync --all-extras

CMD ["bash"]

# Notes for use:
# - Build: docker build -t tabicl:gpu .
# - Run with NVIDIA container toolkit: docker run --gpus all -it --rm -v $(pwd):/workspace tabicl:gpu
# - Inside the container you can use `uv` (installed) or pip. Example: `pip install -r requirements.txt` or `uv install ...`
