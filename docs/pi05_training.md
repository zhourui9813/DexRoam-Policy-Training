# DexRoam π0.5 Training and Inference

This implementation is built on Physical Intelligence's official
[OpenPI π0.5](https://github.com/Physical-Intelligence/openpi), adapted to the DexRoam setting.
This document describes how to fine-tune π0.5 on the converted LeRobot dataset.
For data download and conversion, see the [top-level README](../README.md).

## 1. Download Pretrained Models

The π0.5 base model and tokenizer are hosted on Google Cloud Storage, so the Google Cloud CLI is required.

Install it on Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install ca-certificates gnupg curl -y
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update
sudo apt-get install google-cloud-cli -y
```

Then log in:

```bash
gcloud auth login
```

> [!TIP]
> For other operating systems and installation methods, see the
> [Google Cloud CLI installation guide](https://docs.cloud.google.com/sdk/docs/install-sdk).

From the repository root, download the base model and tokenizer to the paths expected by the provided scripts:

```bash
cd dexroam_pi05
mkdir -p checkpoints/pretrained/openpi
mkdir -p checkpoints/pretrained/paligemma/big_vision

gcloud storage rsync --recursive \
  gs://openpi-assets/checkpoints/pi05_base \
  checkpoints/pretrained/openpi/pi05_base

gcloud storage cp \
  gs://big_vision/paligemma_tokenizer.model \
  checkpoints/pretrained/paligemma/big_vision/paligemma_tokenizer.model
```

For other pretrained OpenPI models, see the
[official model checkpoint list](https://github.com/Physical-Intelligence/openpi#model-checkpoints).

## 2. Environment Setup

π0.5 **must use a separate environment from GR00T**:

```bash
cd dexroam_pi05
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Verify the installation:

```bash
uv run python -c "import openpi; print('OpenPI installed successfully')"
```

## 3. Fine-tuning

Fine-tuning consists of three steps: configure training, compute normalization statistics, and launch training.

### 3.1 Configure Training

DexRoam training configs are defined in `src/openpi/training/dexroam_train_config.py`. Set the fields according to your dataset and hardware:

| Field | Description |
| --- | --- |
| `name` | Unique config name; must match `CONFIG_NAME` in `run_compute_norm_stats.sh`, `TRAIN_CONFIG_NAME` in `run_train_pi.sh`, and `--policy.config` in `run_pi_server.sh` |
| `action_dim` | DexRoam action dimension; keep it at `56` for the provided dataset |
| `action_horizon` | Number of predicted action steps |
| `max_token_len` | Maximum language-token sequence length |
| `repo_id` | Dataset path relative to `HF_LEROBOT_HOME` |
| `prompt_from_task` | Reads the language prompt from the task description stored in the LeRobot dataset |
| `num_workers` | Number of data-loading workers; reduce it if CPU memory or worker resources are limited |
| `batch_size` | Global batch size; must be divisible by the number of JAX devices |
| `fsdp_devices` | Number of devices across which the model is sharded; use `1` for single-GPU training |
| `weight_loader` | Path to the pretrained weights loaded at the start of training |

> [!IMPORTANT]
> If you change `name`, update it consistently in all three scripts listed above.

### 3.2 Compute Normalization Statistics

Once the config is ready, compute its normalization statistics:

```bash
cd dexroam_pi05
source .venv/bin/activate
bash scripts/run_compute_norm_stats.sh
```

The generated `norm_stats.json` is written under `dexroam_pi05/assets/`.

### 3.3 Launch Training

After `norm_stats.json` has been generated, launch training:

```bash
bash scripts/run_train_pi.sh
```

The script selects the config via `TRAIN_CONFIG_NAME` and creates a timestamped experiment name.

## 4. Launch the Inference Server

Start the π0.5 inference server:

```bash
bash scripts/run_pi_server.sh
```

The `--port 8000` argument sets the WebSocket port. The server address is fixed to `0.0.0.0` in `scripts/run_pi_server.py`.

For more details, see the [official OpenPI documentation](https://github.com/Physical-Intelligence/openpi).