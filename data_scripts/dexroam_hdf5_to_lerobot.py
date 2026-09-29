#!/usr/bin/env python3
"""
Convert Astribot HDF5 episodes into a standalone LeRobot EEF stereo dataset.

The output includes both LeRobot v2.1 ``episodes_stats.jsonl`` and GR00T
``meta/stats.json`` so the same dataset can be consumed by Dexroam-pi and GR00T.
All conversion helpers are embedded in this file; it has no repository-local imports.

Default dataset design (matches ``astribot-xhand-deploy/astri_deploy_match_config.py``
and ``astri_deploy_match_modality.json``):
- ``observation.state`` is **53-D** float32: current left/right/torso **9-D EEF poses** (xyz + rot6d;
  loaded from `/fk_poses_dict_from_state`, where HDF5 stores 7D xyz+quat) + left/right dex-hand
  **12-D** joint angles (radians; HDF5 degrees converted like the joint converter) + head joints
  **2-D**.
- ``action`` is **56-D** float32 in deploy modality order:
  left/right dex hand **12-D + 12-D** (next-timestep), left/right/torso Cartesian commands
  **9-D + 9-D + 9-D** built from ``/fk_poses_relative_action/{astribot_arm_left,
  astribot_arm_right, astribot_torso}`` (per-step local-frame relative pose stored as
  xyz+quat in HDF5, lifted to xyz+rot6d here), head joints **2-D** from
  ``/joints_dict/joints_position_command[:, -2:]`` (next-timestep), and mobile relative
  position **3-D** from ``/mobile_dict/relative_position``.
- Video: ``head_stereo_left`` from ``/images_dict/stereo_left/rgb`` and
  ``head_stereo_right`` from ``/images_dict/stereo_right/rgb``.
- Language column: ``annotation.human.task_description`` (int task index), per GR00T modality mapping.

Example::

    python dexroam_hdf5_to_lerobot.py \\
        --input-dir /home/ps/yyb/data/pick_burger_converted \\
        --output-root /path/to/lerobot_out \\
        --task \"pick burger\"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_VIDEO_BATCH_SIZE = 128

def ensure_dataset_exists(h5_file: h5py.File, key: str) -> h5py.Dataset:
    if key not in h5_file:
        raise KeyError(f"Missing dataset: {key}")
    obj = h5_file[key]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"Expected dataset at {key}, got {type(obj)}")
    return obj

def to_float32(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)

def degrees_to_radians(arr: np.ndarray) -> np.ndarray:
    return np.deg2rad(np.asarray(arr, dtype=np.float32)).astype(np.float32)

def pose7d_xyzw_to_xyz_rot6d(pose7d: np.ndarray) -> np.ndarray:
    """Convert Astribot 7D pose data to GR00T EEF 9D format.

    The raw Astribot pose datasets are stored as:

    ``[x, y, z, qx, qy, qz, qw]``

    Evidence from the raw HDF5:
    - all last-4 values have unit norm
    - the chassis identity pose is stored as ``[0, 0, 0, 0, 0, 0, 1]``,
      which matches quaternion identity in ``xyzw`` order

    This function converts that 7D pose into the official GR00T EEF format:

    ``[x, y, z, rot6d_0, rot6d_1, rot6d_2, rot6d_3, rot6d_4, rot6d_5]``

    where ``rot6d`` is formed by flattening the first two rows of the 3x3
    rotation matrix.
    """
    pose7d = np.asarray(pose7d, dtype=np.float32)
    if pose7d.shape[-1] != 7:
        raise ValueError(f"Expected last dimension 7 for pose7d, got shape {pose7d.shape}.")

    leading_shape = pose7d.shape[:-1]
    flat_pose7d = pose7d.reshape(-1, 7)

    xyz = flat_pose7d[:, :3]
    quat_xyzw = flat_pose7d[:, 3:7]
    quat_norm = np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    if np.any(quat_norm < 1e-8):
        raise ValueError("Encountered a near-zero quaternion while converting pose7d.")
    quat_xyzw = quat_xyzw / quat_norm

    rotation_matrices = quaternion_xyzw_to_matrix(quat_xyzw)
    rot6d = rotation_matrices[:, :2, :].reshape(-1, 6)

    xyz_rot6d = np.concatenate([xyz, rot6d.astype(np.float32)], axis=1)
    return xyz_rot6d.reshape(*leading_shape, 9).astype(np.float32)

def quaternion_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert normalized quaternions in xyzw order to rotation matrices."""
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    if quat_xyzw.ndim != 2 or quat_xyzw.shape[1] != 4:
        raise ValueError(
            f"Expected quaternion array with shape (N, 4), got {quat_xyzw.shape}."
        )

    x = quat_xyzw[:, 0]
    y = quat_xyzw[:, 1]
    z = quat_xyzw[:, 2]
    w = quat_xyzw[:, 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    rotation_matrices = np.empty((quat_xyzw.shape[0], 3, 3), dtype=np.float32)
    rotation_matrices[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    rotation_matrices[:, 0, 1] = 2.0 * (xy - zw)
    rotation_matrices[:, 0, 2] = 2.0 * (xz + yw)
    rotation_matrices[:, 1, 0] = 2.0 * (xy + zw)
    rotation_matrices[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    rotation_matrices[:, 1, 2] = 2.0 * (yz - xw)
    rotation_matrices[:, 2, 0] = 2.0 * (xz - yw)
    rotation_matrices[:, 2, 1] = 2.0 * (yz + xw)
    rotation_matrices[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return rotation_matrices

def extract_episode_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    if match:
        return (int(match.group(1)), path.name)
    return (-1, path.name)

def export_video_to_mp4(
    frames: h5py.Dataset,
    output_path: Path,
    fps: int,
    compression_crf: int,
    *,
    raw_input_pix_fmt: str = "rgb24",
    progress_desc: str | None = None,
    frame_batch_size: int = DEFAULT_VIDEO_BATCH_SIZE,
    show_progress: bool | None = None,
) -> None:
    """Stream raw frames to ffmpeg. ``raw_input_pix_fmt`` is the layout of bytes in ``frames``.

    Astribot HDF5 color images typically follow the SDK convention (**BGR** in memory) even when
    the dataset path ends in ``rgb``; use ``raw_input_pix_fmt="bgr24"`` so reds are not swapped
    with blues in the output MP4.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_frames, height, width, channels = frames.shape
    if channels != 3:
        raise ValueError(f"Expected RGB frames with 3 channels, got {frames.shape}")

    if raw_input_pix_fmt not in ("rgb24", "bgr24"):
        raise ValueError(f"raw_input_pix_fmt must be 'rgb24' or 'bgr24', got {raw_input_pix_fmt!r}")
    if frame_batch_size <= 0:
        raise ValueError(f"frame_batch_size must be positive, got {frame_batch_size}.")
    if show_progress is None:
        show_progress = progress_desc is not None and sys.stderr.isatty()

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-f",
        "rawvideo",
        "-pix_fmt",
        raw_input_pix_fmt,
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(compression_crf),
        str(output_path),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=1024 * 1024,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed or not available on PATH.") from exc

    assert proc.stdin is not None
    progress_bar = None
    if show_progress and progress_desc is not None:
        progress_bar = tqdm(total=num_frames, desc=progress_desc, unit="frame", leave=False)
    stderr = ""
    try:
        for start in range(0, num_frames, frame_batch_size):
            end = min(start + frame_batch_size, num_frames)
            batch = np.asarray(frames[start:end], dtype=np.uint8)
            if batch.ndim != 4 or batch.shape[-1] != 3:
                raise ValueError(
                    f"Expected frame batch with shape (N, H, W, 3), got {batch.shape} "
                    f"for frames[{start}:{end}]."
                )
            if not batch.flags["C_CONTIGUOUS"]:
                batch = np.ascontiguousarray(batch)
            proc.stdin.write(batch.tobytes())
            if progress_bar is not None:
                progress_bar.update(end - start)
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        returncode = proc.wait()
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()

    if returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {output_path}:\n{stderr}")

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

def validate_episode_shapes(
    reference_shapes: dict[str, tuple[int, int, int]] | None,
    episode_shapes: dict[str, tuple[int, int, int]],
    hdf5_path: Path,
) -> dict[str, tuple[int, int, int]]:
    if reference_shapes is None:
        return episode_shapes
    if set(reference_shapes) != set(episode_shapes):
        raise ValueError(
            f"Video keys mismatch for {hdf5_path}. "
            f"Expected {sorted(reference_shapes)}, got {sorted(episode_shapes)}"
        )
    for video_key, ref_shape in reference_shapes.items():
        if ref_shape != episode_shapes[video_key]:
            raise ValueError(
                f"Video shape mismatch for {hdf5_path} and key {video_key}: "
                f"expected {ref_shape}, got {episode_shapes[video_key]}"
            )
    return reference_shapes

def get_feature_stats(array: np.ndarray, axis=0, keepdims=False) -> dict[str, np.ndarray]:
    """Compute the numeric statistics used by LeRobot episode metadata."""
    values = np.asarray(array)
    count = np.array([len(values)], dtype=np.int64)
    return {
        "min": np.min(values, axis=axis, keepdims=keepdims),
        "max": np.max(values, axis=axis, keepdims=keepdims),
        "mean": np.mean(values, axis=axis, keepdims=keepdims),
        "std": np.std(values, axis=axis, keepdims=keepdims),
        "count": count,
    }


def _path_has_error_marker(path: Path, root: Path | None = None) -> bool:
    """Check error markers relative to the searched input root."""
    try:
        parts = path.resolve().relative_to(root.resolve()).parts if root is not None else (path.name,)
    except ValueError:
        parts = (path.name,)
    return any("error" in part.lower() for part in parts)


def _filter_error_paths(
    paths: list[Path], *, skip_error_paths: bool, roots: list[Path] | None = None
) -> tuple[list[Path], list[Path]]:
    if not skip_error_paths:
        return paths, []
    kept: list[Path] = []
    skipped: list[Path] = []
    for path in paths:
        root = next((candidate for candidate in (roots or []) if path.resolve().is_relative_to(candidate.resolve())), None)
        if _path_has_error_marker(path, root):
            skipped.append(path)
        else:
            kept.append(path)
    return kept, skipped


def _hdf5_glob_patterns(args: argparse.Namespace) -> list[str]:
    patterns = [args.glob]
    if getattr(args, "include_h5", False) and "*.h5" not in patterns:
        patterns.append("*.h5")
    return patterns


def _sort_hdf5_path(path: Path, input_dir: Path) -> tuple[str, Any, str]:
    try:
        rel_parent = path.parent.relative_to(input_dir)
    except ValueError:
        rel_parent = path.parent
    return str(rel_parent), extract_episode_sort_key(path), path.name


def _collect_hdf5_paths_from_dir(args: argparse.Namespace, input_dir: Path) -> list[Path]:
    recursive = bool(getattr(args, "recursive", False))
    raw_paths: dict[Path, None] = {}
    for pattern in _hdf5_glob_patterns(args):
        iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        for path in iterator:
            if path.is_file():
                raw_paths[path.resolve()] = None
    return sorted(raw_paths.keys(), key=lambda path: _sort_hdf5_path(path, input_dir))


def _report_input_path_summary(
    *,
    searched_dirs: list[Path],
    kept_paths: list[Path],
    skipped_paths: list[Path],
    always: bool = False,
) -> None:
    if not skipped_paths and not always:
        return
    if searched_dirs:
        print(
            "[input filter] skipped %d path(s) containing 'error'; using %d HDF5 file(s) from %d input dir(s)."
            % (len(skipped_paths), len(kept_paths), len(searched_dirs)),
            file=sys.stderr,
        )
    else:
        print(
            "[input filter] skipped %d path(s) containing 'error'; using %d HDF5 file(s)."
            % (len(skipped_paths), len(kept_paths)),
            file=sys.stderr,
        )
    preview = skipped_paths[:5]
    for path in preview:
        print(f"[input filter] skipped: {path}", file=sys.stderr)
    if len(skipped_paths) > len(preview):
        print(
            "[input filter] ... %d more skipped path(s)."
            % (len(skipped_paths) - len(preview)),
            file=sys.stderr,
        )


def resolve_input_paths_multi(args: argparse.Namespace) -> list[Path]:
    """Resolve HDF5 paths from ``--input``, ``--input-dir``, or ``--input-dirs``.

    For ``--input-dirs``, files inside each directory are sorted with the shared
    episode-sort heuristic, and directories are concatenated in the order given
    on the command line so a single LeRobot dataset is built from all of them.
    """
    searched_dirs: list[Path] = []
    if getattr(args, "input_dirs", None):
        paths: list[Path] = []
        for raw_dir in args.input_dirs:
            input_dir = Path(raw_dir).resolve()
            if not input_dir.is_dir():
                raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
            searched_dirs.append(input_dir)
            dir_paths = _collect_hdf5_paths_from_dir(args, input_dir)
            if not dir_paths:
                raise FileNotFoundError(
                    f"No HDF5 files matched {args.glob!r} under directory: {input_dir}"
                )
            paths.extend(dir_paths)
        paths, skipped_paths = _filter_error_paths(
            paths,
            skip_error_paths=getattr(args, "skip_error_paths", True),
            roots=searched_dirs,
        )
        _report_input_path_summary(
            searched_dirs=searched_dirs,
            kept_paths=paths,
            skipped_paths=skipped_paths,
            always=bool(getattr(args, "recursive", False) or getattr(args, "include_h5", False)),
        )
        if not paths:
            raise FileNotFoundError("All matched HDF5 files were skipped by the error-path filter.")
        return paths

    if getattr(args, "input_dir", None):
        input_dir = args.input_dir.resolve()
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
        searched_dirs.append(input_dir)
        paths = _collect_hdf5_paths_from_dir(args, input_dir)
        if not paths:
            raise FileNotFoundError(
                f"No HDF5 files matched {args.glob!r} under directory: {input_dir}"
            )
        paths, skipped_paths = _filter_error_paths(
            paths,
            skip_error_paths=getattr(args, "skip_error_paths", True),
            roots=searched_dirs,
        )
        _report_input_path_summary(
            searched_dirs=searched_dirs,
            kept_paths=paths,
            skipped_paths=skipped_paths,
            always=bool(getattr(args, "recursive", False) or getattr(args, "include_h5", False)),
        )
        if not paths:
            raise FileNotFoundError("All matched HDF5 files were skipped by the error-path filter.")
        return paths

    if args.input is None:
        raise ValueError("One of --input, --input-dir, or --input-dirs is required.")
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input HDF5 file does not exist: {input_path}")
    paths = [input_path]
    paths, skipped_paths = _filter_error_paths(
        paths,
        skip_error_paths=getattr(args, "skip_error_paths", True),
        roots=None,
    )
    _report_input_path_summary(
        searched_dirs=[],
        kept_paths=paths,
        skipped_paths=skipped_paths,
    )
    if not paths:
        raise FileNotFoundError("Input HDF5 file was skipped by the error-path filter.")
    return paths


HEAD_JOINT_DIM = 2

# --- 53-D state / 56-D action, 9-D EEF poses, stereo RGB only ---
STATE_DIM = 53
ACTION_DIM = 56

POSE_STATE_KEYS = (
    "/fk_poses_dict_from_state/astribot_arm_left",
    "/fk_poses_dict_from_state/astribot_arm_right",
    "/fk_poses_dict_from_state/astribot_torso",
)
RELATIVE_ACTION_KEYS = (
    "/fk_poses_relative_action/astribot_arm_left",
    "/fk_poses_relative_action/astribot_arm_right",
    "/fk_poses_relative_action/astribot_torso",
)
DEX_STATE_KEYS = (
    "/dex_hand_dict/left/state",
    "/dex_hand_dict/right/state",
)
HEAD_JOINT_KEY = "/joints_dict/joints_position_command"
MOBILE_RELATIVE_KEY = "/mobile_dict/relative_position"
TIME_KEY = "/time"
VIDEO_SPECS = {
    "head_stereo_left": {
        "dataset": "/images_dict/stereo_left/rgb",
    },
    "head_stereo_right": {
        "dataset": "/images_dict/stereo_right/rgb",
    },
}


def _eef9_feature_names(prefix: str) -> list[str]:
    return [
        f"{prefix}_x",
        f"{prefix}_y",
        f"{prefix}_z",
        f"{prefix}_rot6d_1",
        f"{prefix}_rot6d_2",
        f"{prefix}_rot6d_3",
        f"{prefix}_rot6d_4",
        f"{prefix}_rot6d_5",
        f"{prefix}_rot6d_6",
    ]


STATE_FEATURE_NAMES = [
    *_eef9_feature_names("poses_left"),
    *_eef9_feature_names("poses_right"),
    *_eef9_feature_names("poses_torso"),
    *[f"dex_left_joint_{i}" for i in range(12)],
    *[f"dex_right_joint_{i}" for i in range(12)],
    "head_joint_0",
    "head_joint_1",
]
ACTION_FEATURE_NAMES = [
    *[f"cmd_left_hand_{i}" for i in range(12)],
    *[f"cmd_right_hand_{i}" for i in range(12)],
    *_eef9_feature_names("cmd_left_arm"),
    *_eef9_feature_names("cmd_right_arm"),
    *_eef9_feature_names("cmd_torso"),
    "cmd_head_joint_0",
    "cmd_head_joint_1",
    *[f"cmd_mobile_relative_{i}" for i in range(3)],
]

MODALITY_JSON: dict[str, Any] = {
    # Short keys match ``astri_deploy_match_config.py``. Compatibility aliases share the same slices so
    # older checkpoints / modality_configs that still use ``poses_astribot_*`` / ``cmd_*`` keys do
    # not KeyError in ``LeRobotEpisodeLoader.get_dataset_statistics``.
    "state": {
        "left_arm": {"start": 0, "end": 9},
        "right_arm": {"start": 9, "end": 18},
        "torso": {"start": 18, "end": 27},
        "left_dex_hand": {"start": 27, "end": 39},
        "right_dex_hand": {"start": 39, "end": 51},
        "head_joint": {"start": 51, "end": 53},
        "poses_astribot_arm_left": {"start": 0, "end": 9},
        "poses_astribot_arm_right": {"start": 9, "end": 18},
        "poses_astribot_torso": {"start": 18, "end": 27},
        "dex_hand_left_state": {"start": 27, "end": 39},
        "dex_hand_right_state": {"start": 39, "end": 51},
        "head_joint_state": {"start": 51, "end": 53},
    },
    "action": {
        "left_dex_hand": {"start": 0, "end": 12},
        "right_dex_hand": {"start": 12, "end": 24},
        "left_arm": {"start": 24, "end": 33},
        "right_arm": {"start": 33, "end": 42},
        "torso": {"start": 42, "end": 51},
        "head_joint": {"start": 51, "end": 53},
        "mobile_relative": {"start": 53, "end": 56},
        "cmd_left_hand": {"start": 0, "end": 12},
        "cmd_right_hand": {"start": 12, "end": 24},
        "cmd_left_arm": {"start": 24, "end": 33},
        "cmd_right_arm": {"start": 33, "end": 42},
        "cmd_torso": {"start": 42, "end": 51},
        "cmd_head_joint": {"start": 51, "end": 53},
        "cmd_mobile_relative": {"start": 53, "end": 56},
    },
    "video": {
        "head_stereo_left": {"original_key": "observation.images.head_stereo_left"},
        "head_stereo_right": {"original_key": "observation.images.head_stereo_right"},
    },
    "annotation": {
        "human.task_description": {"original_key": "annotation.human.task_description"},
    },
}

def _sample_indices(data_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75):
    if data_len <= 0:
        return np.array([], dtype=np.int64)
    if data_len < min_num_samples:
        min_num_samples = data_len
    num_samples = max(min_num_samples, min(int(data_len**power), max_num_samples))
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(np.int64)


def _auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
    _, height, width = img.shape
    if max(width, height) < max_size_threshold:
        return img
    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def _serialize_stats(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, list[float] | int | float]]:
    serialized = {}
    for feature_key, feature_stats in stats.items():
        serialized[feature_key] = {}
        for stat_key, value in feature_stats.items():
            array = np.asarray(value)
            if isinstance(value, np.generic):
                serialized[feature_key][stat_key] = value.item()
            elif array.ndim == 0:
                serialized[feature_key][stat_key] = array.item()
            else:
                serialized[feature_key][stat_key] = array.tolist()
    return serialized


def _compute_numeric_feature_stats(array: np.ndarray) -> dict[str, np.ndarray]:
    axes_to_reduce = 0
    keepdims = array.ndim == 1
    return get_feature_stats(array, axis=axes_to_reduce, keepdims=keepdims)


def _compute_video_feature_stats(video_ds: h5py.Dataset) -> dict[str, np.ndarray]:
    sampled_indices = _sample_indices(len(video_ds))
    images = None
    for i, idx in enumerate(sampled_indices):
        img = np.asarray(video_ds[idx], dtype=np.uint8)
        img = np.transpose(img, (2, 0, 1))
        img = _auto_downsample_height_width(img)
        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)
        images[i] = img

    assert images is not None
    stats = get_feature_stats(images, axis=(0, 2, 3), keepdims=True)
    return {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in stats.items()}


def build_frame_timestamps(num_steps: int, fps: int) -> np.ndarray:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    return (np.arange(num_steps, dtype=np.float32) / float(fps)).astype(np.float32)


def load_pose7d_raw(h5_file: h5py.File, dataset_key: str) -> np.ndarray:
    """Load (T,7) xyz+quat float32 without rot6d conversion."""
    pose7d = to_float32(ensure_dataset_exists(h5_file, dataset_key)[:])
    if pose7d.ndim != 2 or pose7d.shape[1] != 7:
        raise ValueError(
            f"Expected a pose dataset with shape (T, 7) at {dataset_key}, got {pose7d.shape}."
        )
    quat_norm = np.linalg.norm(pose7d[:, 3:7], axis=1)
    zero_quat_mask = quat_norm < 1e-8
    if np.any(zero_quat_mask):
        pose7d = pose7d.copy()
        pose7d[zero_quat_mask, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return pose7d


def load_pose7d_as_xyz_rot6d(h5_file: h5py.File, dataset_key: str) -> np.ndarray:
    """Load ``(T, 7)`` xyz+quat from HDF5 and return ``(T, 9)`` xyz+rot6d for GR00T EEF."""
    pose7d = load_pose7d_raw(h5_file, dataset_key)
    return pose7d_xyzw_to_xyz_rot6d(pose7d)


def build_state_and_action(
    h5_file: h5py.File,
    zero_mobile_relative: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 53-D state and 56-D action used by the deploy dataset."""
    timestamps = np.asarray(ensure_dataset_exists(h5_file, TIME_KEY)[:], dtype=np.float64)
    tlen = len(timestamps)

    state_parts: list[np.ndarray] = []
    for key in POSE_STATE_KEYS:
        pose = load_pose7d_as_xyz_rot6d(h5_file, key)
        if len(pose) != tlen:
            raise ValueError(f"Length mismatch for {key}: expected {tlen}, got {len(pose)}")
        state_parts.append(pose)

    for key in DEX_STATE_KEYS:
        dex_deg = ensure_dataset_exists(h5_file, key)[:]
        dex_rad = degrees_to_radians(dex_deg)
        if len(dex_rad) != tlen:
            raise ValueError(f"Length mismatch for {key}: expected {tlen}, got {len(dex_rad)}")
        if dex_rad.shape[1] != 12:
            raise ValueError(f"{key} must be (T,12), got {dex_rad.shape}")
        state_parts.append(dex_rad)

    head_joint_cmd = to_float32(ensure_dataset_exists(h5_file, HEAD_JOINT_KEY)[:])
    if head_joint_cmd.ndim != 2 or head_joint_cmd.shape[1] < HEAD_JOINT_DIM:
        raise ValueError(
            f"{HEAD_JOINT_KEY} must be (T, >=2), got {head_joint_cmd.shape}."
        )
    if len(head_joint_cmd) != tlen:
        raise ValueError(
            f"Length mismatch for {HEAD_JOINT_KEY}: expected {tlen}, got {len(head_joint_cmd)}"
        )
    head_joint = head_joint_cmd[:, -HEAD_JOINT_DIM:]
    state_parts.append(head_joint)
    state = np.concatenate(state_parts, axis=1).astype(np.float32)
    if state.shape[1] != STATE_DIM:
        raise ValueError(f"Internal error: state dim {state.shape[1]} != {STATE_DIM}")

    cmd_left, cmd_right, cmd_torso = (
        load_pose7d_as_xyz_rot6d(h5_file, key) for key in RELATIVE_ACTION_KEYS
    )
    for key, rel in zip(
        RELATIVE_ACTION_KEYS, (cmd_left, cmd_right, cmd_torso)
    ):
        if len(rel) != tlen:
            raise ValueError(f"Length mismatch for {key}: expected {tlen}, got {len(rel)}")

    dex_left_rad = degrees_to_radians(ensure_dataset_exists(h5_file, DEX_STATE_KEYS[0])[:])
    dex_right_rad = degrees_to_radians(ensure_dataset_exists(h5_file, DEX_STATE_KEYS[1])[:])
    hand_left_action = np.concatenate([dex_left_rad[1:], dex_left_rad[-1:]], axis=0)
    hand_right_action = np.concatenate([dex_right_rad[1:], dex_right_rad[-1:]], axis=0)

    head_action = np.concatenate([head_joint[1:], head_joint[-1:]], axis=0)

    mobile_relative = to_float32(ensure_dataset_exists(h5_file, MOBILE_RELATIVE_KEY)[:])
    if mobile_relative.ndim != 2 or mobile_relative.shape[1] != 3:
        raise ValueError(
            f"{MOBILE_RELATIVE_KEY} must be (T, 3), got {mobile_relative.shape}."
        )
    if len(mobile_relative) != tlen:
        raise ValueError(
            f"Length mismatch for {MOBILE_RELATIVE_KEY}: expected {tlen}, got {len(mobile_relative)}"
        )
    if zero_mobile_relative:
        mobile_relative = np.zeros_like(mobile_relative, dtype=np.float32)

    action = np.concatenate(
        [
            hand_left_action,
            hand_right_action,
            cmd_left,
            cmd_right,
            cmd_torso,
            head_action,
            mobile_relative,
        ],
        axis=1,
    ).astype(np.float32)
    if action.shape[1] != ACTION_DIM:
        raise ValueError(f"Internal error: action dim {action.shape[1]} != {ACTION_DIM}")

    return state, action, timestamps


def build_dataframe(
    state: np.ndarray,
    action: np.ndarray,
    timestamps: np.ndarray,
    episode_index: int,
    global_index_offset: int,
    fps: int,
    task_index: int = 0,
    timestamp_source: str = "fps",
) -> pd.DataFrame:
    if not (len(state) == len(action) == len(timestamps)):
        raise ValueError(
            f"State/action/timestamp length mismatch: {len(state)}, {len(action)}, {len(timestamps)}"
        )
    num_steps = len(state)
    if timestamp_source == "fps":
        normalized_timestamps = build_frame_timestamps(num_steps, fps)
    elif timestamp_source == "hdf5":
        normalized_timestamps = (timestamps - timestamps[0]).astype(np.float32)
    else:
        raise ValueError(f"Unsupported timestamp source: {timestamp_source!r}")
    rows: list[dict[str, Any]] = []
    for frame_index in range(num_steps):
        rows.append(
            {
                "observation.state": state[frame_index],
                "action": action[frame_index],
                "timestamp": float(normalized_timestamps[frame_index]),
                "frame_index": frame_index,
                "episode_index": episode_index,
                "index": global_index_offset + frame_index,
                "task_index": task_index,
                "annotation.human.task_description": task_index,
                "next.reward": 0.0,
                "next.done": bool(frame_index == num_steps - 1),
            }
        )
    return pd.DataFrame(rows)


def build_info_json(
    video_shapes: dict[str, tuple[int, int, int]],
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_videos: int,
    total_chunks: int,
    fps: int,
    chunk_size: int,
) -> dict[str, Any]:
    if len(STATE_FEATURE_NAMES) != STATE_DIM:
        raise ValueError("STATE_FEATURE_NAMES length mismatch")
    if len(ACTION_FEATURE_NAMES) != ACTION_DIM:
        raise ValueError("ACTION_FEATURE_NAMES length mismatch")

    features: dict[str, Any] = {
        "action": {
            "dtype": "float32",
            "names": ACTION_FEATURE_NAMES,
            "shape": [ACTION_DIM],
        },
        "observation.state": {
            "dtype": "float32",
            "names": STATE_FEATURE_NAMES,
            "shape": [STATE_DIM],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "annotation.human.task_description": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
    }
    for video_key, (height, width, channels) in video_shapes.items():
        features[f"observation.images.{video_key}"] = {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": channels,
                "has_audio": False,
            },
        }

    return {
        "codebase_version": "v2.1",
        "robot_type": "astribot_deploy_match",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": chunk_size,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "total_chunks": total_chunks,
        "total_videos": total_videos,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path, help="Path to a source HDF5 file.")
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing one HDF5 file per episode.",
    )
    input_group.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        help=(
            "Multiple directories whose HDF5 files are concatenated (in the given order) "
            "into a single LeRobot dataset. Files within each directory are sorted by "
            "episode index."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where the LeRobot dataset will be written.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="default task",
        help="Task description stored in tasks.jsonl.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video FPS recorded in info.json and used for MP4 export.",
    )
    parser.add_argument(
        "--timestamp-source",
        choices=("fps", "hdf5"),
        default="fps",
        help="Timestamp convention: frame_index/fps or normalized HDF5 /time.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of episodes per chunk directory.",
    )
    parser.add_argument(
        "--compression-crf",
        type=int,
        default=18,
        help="ffmpeg CRF for libx264 export. Lower means higher quality.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.hdf5",
        help="Glob pattern used with --input-dir.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Recursively search --input-dir/--input-dirs for HDF5 files. "
            "Without this flag, only files directly under each input directory are used."
        ),
    )
    parser.add_argument(
        "--include-h5",
        action="store_true",
        help="Also include files matching *.h5 in addition to --glob.",
    )
    parser.add_argument(
        "--no-skip-error-paths",
        dest="skip_error_paths",
        action="store_false",
        help="Disable skipping paths that contain 'error'.",
    )
    parser.add_argument(
        "--video-batch-size",
        type=int,
        default=DEFAULT_VIDEO_BATCH_SIZE,
        help="How many frames to read from HDF5 and stream to ffmpeg per batch.",
    )
    parser.add_argument(
        "--no-video-progress",
        action="store_true",
        help="Disable per-video frame progress bars during MP4 export.",
    )
    parser.add_argument(
        "--zero-mobile-relative",
        action="store_true",
        help=(
            "Replace the final 3-D mobile_relative action slice with zeros in the "
            "exported LeRobot dataset."
        ),
    )
    return parser.parse_args()


def build_episode_stats(
    state: np.ndarray,
    action: np.ndarray,
    timestamps: np.ndarray,
    episode_index: int,
    task_index: int,
    video_feature_datasets: dict[str, h5py.Dataset],
    fps: int | None = None,
) -> dict[str, Any]:
    num_steps = len(state)
    normalized_timestamps = (
        build_frame_timestamps(num_steps, fps)
        if fps is not None
        else (timestamps - timestamps[0]).astype(np.float32)
    )
    frame_index = np.arange(num_steps, dtype=np.int64)
    episode_index_arr = np.full((num_steps,), episode_index, dtype=np.int64)
    task_index_arr = np.full((num_steps,), task_index, dtype=np.int64)
    annotation_task_arr = np.full((num_steps,), task_index, dtype=np.int64)

    stats = {
        "action": _compute_numeric_feature_stats(action.astype(np.float32)),
        "observation.state": _compute_numeric_feature_stats(state.astype(np.float32)),
        "timestamp": _compute_numeric_feature_stats(normalized_timestamps),
        "frame_index": _compute_numeric_feature_stats(frame_index),
        "episode_index": _compute_numeric_feature_stats(episode_index_arr),
        "index": _compute_numeric_feature_stats(frame_index),
        "task_index": _compute_numeric_feature_stats(task_index_arr),
        "annotation.human.task_description": _compute_numeric_feature_stats(annotation_task_arr),
    }

    for feature_key, video_ds in video_feature_datasets.items():
        stats[feature_key] = _compute_video_feature_stats(video_ds)

    return {
        "episode_index": episode_index,
        "stats": _serialize_stats(stats),
    }


def convert_single_episode(
    hdf5_path: Path,
    output_root: Path,
    episode_index: int,
    global_index_offset: int,
    args: argparse.Namespace,
) -> tuple[int, int, int, dict[str, tuple[int, int, int]], dict[str, Any]]:
    episode_chunk = episode_index // args.chunk_size
    data_dir = output_root / "data" / f"chunk-{episode_chunk:03d}"
    videos_dir = output_root / "videos" / f"chunk-{episode_chunk:03d}"
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as h5_file:
        state, action, timestamps = build_state_and_action(
            h5_file, zero_mobile_relative=args.zero_mobile_relative
        )
        df = build_dataframe(
            state=state,
            action=action,
            timestamps=timestamps,
            episode_index=episode_index,
            global_index_offset=global_index_offset,
            fps=args.fps,
            timestamp_source=args.timestamp_source,
        )
        video_specs = VIDEO_SPECS

        df.to_parquet(data_dir / f"episode_{episode_index:06d}.parquet", index=False)

        episode_video_shapes: dict[str, tuple[int, int, int]] = {}
        video_feature_datasets: dict[str, h5py.Dataset] = {}
        show_video_progress = (not args.no_video_progress) and sys.stderr.isatty()
        for video_key, spec in video_specs.items():
            dataset_key = spec["dataset"]
            if dataset_key not in h5_file:
                raise KeyError(
                    f"Required stereo video dataset {dataset_key!r} is missing in {hdf5_path}."
                )
            video_ds = ensure_dataset_exists(h5_file, dataset_key)
            _, height, width, channels = video_ds.shape
            episode_video_shapes[video_key] = (height, width, channels)
            video_feature_datasets[f"observation.images.{video_key}"] = video_ds

            output_path = (
                videos_dir
                / f"observation.images.{video_key}"
                / f"episode_{episode_index:06d}.mp4"
            )
            export_video_to_mp4(
                video_ds,
                output_path,
                fps=args.fps,
                compression_crf=args.compression_crf,
                raw_input_pix_fmt="bgr24",
                progress_desc=(
                    f"episode {episode_index + 1}: export {video_key} ({len(video_ds)} frames)"
                    if show_video_progress
                    else None
                ),
                frame_batch_size=args.video_batch_size,
                show_progress=show_video_progress,
            )

        episode_stats = build_episode_stats(
            state=state,
            action=action,
            timestamps=timestamps,
            episode_index=episode_index,
            task_index=0,
            video_feature_datasets=video_feature_datasets,
            fps=args.fps if args.timestamp_source == "fps" else None,
        )

    return len(df), state.shape[1], action.shape[1], episode_video_shapes, episode_stats


def _stack_parquet_feature_values(parquet_paths: list[Path], feature_key: str) -> np.ndarray:
    values: list[np.ndarray] = []
    for parquet_path in parquet_paths:
        frame_values = pd.read_parquet(parquet_path, columns=[feature_key])[feature_key]
        values.extend(np.asarray(value, dtype=np.float32).reshape(-1) for value in frame_values)
    if not values:
        raise ValueError(f"No values found for feature {feature_key!r}.")
    return np.stack(values, axis=0)


def write_global_gr00t_stats(output_root: Path, info: dict[str, Any]) -> None:
    """Write GR00T's global stats.json from the exported parquet files."""
    parquet_paths = sorted((output_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No episode parquet files found under {output_root / 'data'}.")
    stats: dict[str, Any] = {}
    for feature_key, feature_info in info["features"].items():
        if "float" not in feature_info.get("dtype", ""):
            continue
        values = _stack_parquet_feature_values(parquet_paths, feature_key)
        stats[feature_key] = {
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    write_json(output_root / "meta" / "stats.json", stats)


def convert_hdf5_to_lerobot(args: argparse.Namespace) -> Path:
    input_paths = resolve_input_paths_multi(args)
    output_root = args.output_root.resolve()

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_videos = 0
    episodes_rows: list[dict[str, Any]] = []
    episodes_stats_rows: list[dict[str, Any]] = []
    video_shapes: dict[str, tuple[int, int, int]] | None = None
    state_dim: int | None = None
    action_dim: int | None = None

    for episode_index, input_path in enumerate(
        tqdm(input_paths, desc="Converting episodes", unit="episode", disable=not sys.stderr.isatty())
    ):
        (
            frame_count,
            current_state_dim,
            current_action_dim,
            episode_video_shapes,
            episode_stats,
        ) = convert_single_episode(
            hdf5_path=input_path,
            output_root=output_root,
            episode_index=episode_index,
            global_index_offset=total_frames,
            args=args,
        )

        if state_dim is None:
            state_dim = current_state_dim
            action_dim = current_action_dim
        elif state_dim != current_state_dim or action_dim != current_action_dim:
            raise ValueError(
                f"State/action dimensions must match across episodes. "
                f"Expected state={state_dim}, action={action_dim}, "
                f"got state={current_state_dim}, action={current_action_dim} for {input_path}"
            )

        video_shapes = validate_episode_shapes(video_shapes, episode_video_shapes, input_path)

        episodes_rows.append(
            {
                "episode_index": episode_index,
                "tasks": [args.task],
                "length": frame_count,
            }
        )
        episodes_stats_rows.append(episode_stats)
        total_frames += frame_count
        total_videos += len(episode_video_shapes)

    assert state_dim is not None
    assert action_dim is not None
    assert video_shapes is not None
    total_episodes = len(input_paths)
    total_chunks = ((total_episodes - 1) // args.chunk_size) + 1

    if state_dim != STATE_DIM or action_dim != ACTION_DIM:
        raise ValueError(
            f"Expected state_dim={STATE_DIM}, action_dim={ACTION_DIM}; "
            f"got {state_dim=}, {action_dim=} ."
        )
    write_json(
        meta_dir / "info.json",
        build_info_json(
            video_shapes=video_shapes,
            total_episodes=total_episodes,
            total_frames=total_frames,
            total_tasks=1,
            total_videos=total_videos,
            total_chunks=total_chunks,
            fps=args.fps,
            chunk_size=args.chunk_size,
        ),
    )
    write_json(meta_dir / "modality.json", MODALITY_JSON)
    write_jsonl(meta_dir / "episodes.jsonl", episodes_rows)
    write_jsonl(
        meta_dir / "tasks.jsonl",
        [
            {
                "task_index": 0,
                "task": args.task,
            }
        ],
    )
    write_jsonl(meta_dir / "episodes_stats.jsonl", episodes_stats_rows)
    with open(meta_dir / "info.json", "r", encoding="utf-8") as info_file:
        info_payload = json.load(info_file)
    write_global_gr00t_stats(output_root, info_payload)

    return output_root


def main() -> None:
    args = parse_args()
    output_root = convert_hdf5_to_lerobot(args)
    print(f"Converted 53-D/56-D stereo dataset written to: {output_root}")


if __name__ == "__main__":
    main()
