# DexRoam GR00T Training and Inference

This implementation is built on NVIDIA's official
[Isaac GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T), adapted to the DexRoam setting.
This document describes how to fine-tune GR00T N1.7 on the converted LeRobot dataset.
For data download and conversion, see the [top-level README](../README.md).

## 1. Download Pretrained Models

Download the following pretrained models:

| Model                                                        | Purpose                                        |
| ------------------------------------------------------------ | ---------------------------------------------- |
| [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) | Base model (`BASE_MODEL`)                      |
| [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | Vision-language backbone (`COSMOS_MODEL_PATH`) |

> [!NOTE]
> Cosmos-Reason2-2B is a gated model. Request access on its Hugging Face page before downloading.

From the repository root, download both models:

```bash
cd dexroam_gr00t
mkdir -p checkpoints/pretrained

hf download nvidia/GR00T-N1.7-3B \
  --local-dir checkpoints/pretrained/GR00T-N1.7-3B

hf download nvidia/Cosmos-Reason2-2B \
  --local-dir checkpoints/pretrained/Cosmos-Reason2-2B
```

## 2. Environment Setup

```bash
cd dexroam_gr00t
uv sync --python 3.12
```

Verify the installation:

```bash
uv run python -c "import gr00t; print('GR00T installed successfully')"
```

## 3. Fine-tuning

You can launch training directly with `scripts/run_train_gr00t.sh`:

```bash
cd dexroam_gr00t
source .venv/bin/activate
bash scripts/run_train_gr00t.sh
```

All training parameters are defined in the `Environment Configuration` and `Training Configuration` sections of the script. To use custom data, models, or training settings, edit the corresponding variables:

| Variable                 | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `COSMOS_MODEL_PATH`      | Path to the `Cosmos-Reason2-2B` model directory              |
| `BASE_MODEL`             | Path to the `GR00T-N1.7-3B` base model directory             |
| `MODALITY_CONFIG`        | Path to the modality config used for training                |
| `NAME`                   | Experiment name, also used in the output directory           |
| `DATASET`                | Root directory of the LeRobot v2.1 dataset                   |
| `OUTPUT_DIR`             | Checkpoint output directory; defaults to one generated from the experiment name and launch time |
| `MAX_STEPS`              | Total number of training steps                               |
| `GLOBAL_BATCH_SIZE`      | Global batch size across all GPUs                            |

## 4. Launch the Inference Server

Start the GR00T inference server with:

```bash
bash scripts/run_gr00t_server.sh
```

The listening address and port are set by the `--host` and `--port` arguments at the end of the Python command in the script.

For more details, see the [official Isaac-GR00T documentation](https://github.com/NVIDIA/Isaac-GR00T#installation).