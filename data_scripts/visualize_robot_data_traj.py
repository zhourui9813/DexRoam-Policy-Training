#!/usr/bin/env python3
"""Animate one DexRoam robot HDF5 episode with Open3D."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import h5py
    import numpy as np
    import open3d as o3d
    from scipy.spatial.transform import Rotation
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install numpy, h5py, scipy, and open3d."
    ) from exc


# Display constants. They are intentionally not CLI options.
WINDOW_SIZE = (1600, 900)
FRAME_SIZE = 0.10
GRID_STEP = 0.20
MARKER_RADIUS = 0.015
BODY_LINE_WIDTH = 2.0
LINE_WIDTH_TO_RADIUS = 0.002
FALLBACK_FPS = 30.0
CAMERA_HEIGHT = 0.06
CAMERA_DEPTH = 0.12
CAMERA_VERTICAL_FOV_DEG = 60.0
IMAGE_STEP = 4
MOBILE_POSITION_SOURCES = ("relative_position", "absolute_position")
POSE_GROUPS = ("fk_poses_dict_from_state", "poses_dict")
POSE_FRAMES = ("base", "world")

BACKGROUND_COLOR = np.ones(3)
BODY_POINT_COLOR = np.array([0x52, 0x46, 0x46]) / 255.0
BODY_LINE_COLOR = np.array([0xA8, 0xA4, 0x92]) / 255.0
HAND_COLOR = np.array([0xF6, 0x72, 0x80]) / 255.0
FRUSTUM_COLOR = np.array([0xA8, 0xCD, 0x89]) / 255.0
TRAJECTORY_COLOR = np.array([0xFF, 0x91, 0x37]) / 255.0
GRID_COLOR = (np.array([0x9B, 0xB4, 0xC0]) / 255.0) * 0.5 + BACKGROUND_COLOR * 0.5

POSE_NAMES = ("astribot_torso", "astribot_arm_left", "astribot_arm_right")
BODY_NAMES = (
    "astribot_chassis",
    "astribot_torso",
    "astribot_head",
    "astribot_arm_left",
    "astribot_arm_right",
)
BODY_EDGES = np.array([[1, 2]], dtype=np.int32)
HAND_COLORS = {"left": HAND_COLOR, "right": HAND_COLOR}
HAND_EEF = {"left": "astribot_arm_left", "right": "astribot_arm_right"}
HAND_EDGES = np.array(
    [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8], [8, 9],
        [0, 10], [10, 11], [11, 12], [12, 13], [13, 14],
        [0, 15], [15, 16], [16, 17], [17, 18], [18, 19],
        [0, 20], [20, 21], [21, 22], [22, 23], [23, 24],
    ],
    dtype=np.int32,
)
HAND_ROTATIONS = {
    "left": Rotation.from_matrix(
        np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])
    ),
    "right": Rotation.from_matrix(
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    ),
}

HEAD_OFFSET = np.array([0.0, 0.0, 0.1229])
HEAD_JOINT_2_ORIGIN = Rotation.from_euler("xyz", [-1.5708, 0.0, 0.0])
HEAD_TO_CAMERA = Rotation.from_matrix(
    np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
)
IMAGE_PATH = "/images_dict/stereo_left/rgb"


@dataclass
class Sequence:
    path: Path
    times: np.ndarray
    poses: dict[str, np.ndarray]
    transforms: dict[str, np.ndarray]
    hands: dict[str, np.ndarray]
    image_shape: tuple[int, int, int] | None

    @property
    def frame_count(self) -> int:
        return len(self.times)


def _read(root: h5py.File, path: str) -> np.ndarray:
    if path not in root:
        raise ValueError(f"Missing dataset: {path}")
    value = np.asarray(root[path][()])
    if value.shape[0] == 0:
        raise ValueError(f"Empty dataset: {path}")
    return value


def _finite(value: np.ndarray, path: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Dataset contains non-finite values: {path}")
    return value


def _pose7(root: h5py.File, path: str, frame_count: int) -> np.ndarray:
    pose = _finite(_read(root, path), path)
    if pose.shape != (frame_count, 7):
        raise ValueError(f"Invalid shape for {path}: expected ({frame_count}, 7), got {pose.shape}")
    norms = np.linalg.norm(pose[:, 3:7], axis=1)
    if np.any(norms < 1e-12):
        raise ValueError(f"Quaternion has near-zero norm: {path}")
    pose[:, 3:7] /= norms[:, None]
    return pose


def _pose_to_transform(pose: np.ndarray) -> np.ndarray:
    transforms = np.repeat(np.eye(4)[None], len(pose), axis=0)
    transforms[:, :3, :3] = Rotation.from_quat(pose[:, 3:7]).as_matrix()
    transforms[:, :3, 3] = pose[:, :3]
    return transforms


def _base_to_world(chassis: np.ndarray, local_pose: np.ndarray) -> np.ndarray:
    base_position = np.column_stack((chassis[:, :2], np.zeros(len(chassis))))
    base_rotation = Rotation.from_euler("z", chassis[:, 2])
    local_rotation = Rotation.from_quat(local_pose[:, 3:7])
    world = np.empty_like(local_pose)
    world[:, :3] = base_rotation.apply(local_pose[:, :3]) + base_position
    world[:, 3:7] = (base_rotation * local_rotation).as_quat()
    return world


def _integrate_relative_chassis(relative: np.ndarray) -> np.ndarray:
    """Integrate frame-to-frame local [dx, dy, dyaw] into world [x, y, yaw]."""
    chassis = np.zeros_like(relative)
    chassis[0] = relative[0]
    for index in range(1, len(relative)):
        dx, dy, dyaw = relative[index]
        x, y, yaw = chassis[index - 1]
        chassis[index] = [
            x + np.cos(yaw) * dx - np.sin(yaw) * dy,
            y + np.sin(yaw) * dx + np.cos(yaw) * dy,
            yaw + dyaw,
        ]
    return chassis


def _head_pose(torso: np.ndarray, joints: np.ndarray) -> np.ndarray:
    torso_rotation = Rotation.from_quat(torso[:, 3:7])
    rotation = (
        torso_rotation
        * Rotation.from_euler("z", joints[:, 0])
        * HEAD_JOINT_2_ORIGIN
        * Rotation.from_euler("z", joints[:, 1])
    )
    pose = np.empty_like(torso)
    pose[:, :3] = torso_rotation.apply(np.broadcast_to(HEAD_OFFSET, (len(torso), 3))) + torso[:, :3]
    pose[:, 3:7] = rotation.as_quat()
    return pose


def _camera_pose(head: np.ndarray) -> np.ndarray:
    head_rotation = Rotation.from_quat(head[:, 3:7])
    offset = np.broadcast_to([0.0, -CAMERA_HEIGHT, 0.0], (len(head), 3))
    pose = np.empty_like(head)
    pose[:, :3] = head_rotation.apply(offset) + head[:, :3]
    pose[:, 3:7] = (head_rotation * HEAD_TO_CAMERA).as_quat()
    return pose


def _hand_points(raw: np.ndarray, eef_pose: np.ndarray, side: str) -> np.ndarray:
    local = raw[:, :, :3] - raw[:, :1, :3]
    local = HAND_ROTATIONS[side].apply(local.reshape(-1, 3)).reshape(local.shape)
    repeated_rotation = Rotation.from_quat(np.repeat(eef_pose[:, 3:7], 25, axis=0))
    world = repeated_rotation.apply(local.reshape(-1, 3)).reshape(local.shape)
    return world + eef_pose[:, None, :3]


def _timestamps(root: h5py.File, frame_count: int, fps: float) -> np.ndarray:
    if "/time" in root:
        times = _finite(root["/time"][()], "/time")
        if times.shape == (frame_count,) and np.all(np.diff(times) >= 0.0):
            return times - times[0]
        print("[WARN] Invalid /time; using --fps instead.")
    return np.arange(frame_count, dtype=np.float64) / fps


def load_sequence(
    path: str | Path,
    fps: float = FALLBACK_FPS,
    mobile_position_source: str = "relative_position",
    pose_group: str = "fk_poses_dict_from_state",
    pose_frame: str = "base",
) -> Sequence:
    path = Path(path).expanduser().resolve()
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")

    with h5py.File(path, "r") as root:
        if "/mobile_dict" not in root:
            raise ValueError(
                "Unsupported raw/asynchronous HDF5: missing /mobile_dict. "
                "Please use a converted or retargeted/resampled episode."
            )

        mobile_path = f"/mobile_dict/{mobile_position_source}"
        chassis = _finite(_read(root, mobile_path), mobile_path)
        if chassis.ndim != 2 or chassis.shape[1] != 3:
            raise ValueError(
                f"Invalid shape for {mobile_path}: "
                f"expected (T, 3), got {chassis.shape}"
            )
        if mobile_position_source == "relative_position":
            chassis = _integrate_relative_chassis(chassis)
        frame_count = len(chassis)

        chassis_pose = np.zeros((frame_count, 7), dtype=np.float64)
        chassis_pose[:, :2] = chassis[:, :2]
        chassis_pose[:, 3:7] = Rotation.from_euler("z", chassis[:, 2]).as_quat()

        poses = {"astribot_chassis": chassis_pose}
        for name in POSE_NAMES:
            pose = _pose7(root, f"/{pose_group}/{name}", frame_count)
            poses[name] = _base_to_world(chassis, pose) if pose_frame == "base" else pose

        joints = _finite(
            _read(root, "/joints_dict/joints_position_command"),
            "/joints_dict/joints_position_command",
        )
        if joints.ndim != 2 or joints.shape[0] != frame_count or joints.shape[1] < 2:
            raise ValueError(
                "Invalid shape for /joints_dict/joints_position_command: "
                f"expected ({frame_count}, >=2), got {joints.shape}"
            )
        poses["astribot_head"] = _head_pose(poses["astribot_torso"], joints[:, -2:])
        poses["head_rgbd"] = _camera_pose(poses["astribot_head"])

        hands: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            hand_path = f"/human_hand_dict/{side}/raw_data"
            if hand_path not in root:
                continue
            raw = _finite(root[hand_path][()], hand_path)
            if raw.shape != (frame_count, 25 * 7):
                raise ValueError(
                    f"Invalid shape for {hand_path}: expected ({frame_count}, 175), got {raw.shape}"
                )
            raw = raw.reshape(frame_count, 25, 7)
            hands[side] = _hand_points(raw, poses[HAND_EEF[side]], side)

        image_shape = None
        if IMAGE_PATH in root:
            image = root[IMAGE_PATH]
            if (
                image.ndim != 4
                or image.shape[0] != frame_count
                or image.shape[1] < 2
                or image.shape[2] < 2
                or image.shape[-1] not in (3, 4)
            ):
                raise ValueError(
                    f"Invalid shape for {IMAGE_PATH}: expected "
                    f"({frame_count}, H>=2, W>=2, 3/4), "
                    f"got {image.shape}"
                )
            image_shape = (int(image.shape[1]), int(image.shape[2]), int(image.shape[3]))

        times = _timestamps(root, frame_count, fps)

    transforms = {name: _pose_to_transform(pose) for name, pose in poses.items()}
    return Sequence(path, times, poses, transforms, hands, image_shape)


def _line_set(points: np.ndarray, edges: np.ndarray, colors: np.ndarray) -> o3d.geometry.LineSet:
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(edges)
    if colors.ndim == 1:
        colors = np.repeat(colors[None], len(edges), axis=0)
    lines.colors = o3d.utility.Vector3dVector(colors)
    return lines


def _sphere(center: np.ndarray, color: np.ndarray) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_sphere(MARKER_RADIUS, resolution=16)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    mesh.translate(center)
    return mesh


def _grid(points: np.ndarray) -> o3d.geometry.LineSet:
    low = np.min(points[:, :2], axis=0) - 0.3
    high = np.max(points[:, :2], axis=0) + 0.3
    low = np.minimum(np.floor(low / GRID_STEP) * GRID_STEP, 0.0)
    high = np.maximum(np.ceil(high / GRID_STEP) * GRID_STEP, 0.0)
    xs = np.arange(low[0], high[0] + GRID_STEP / 2, GRID_STEP)
    ys = np.arange(low[1], high[1] + GRID_STEP / 2, GRID_STEP)
    vertices = []
    for x in xs:
        vertices.extend(([x, low[1], -0.001], [x, high[1], -0.001]))
    for y in ys:
        vertices.extend(([low[0], y, -0.001], [high[0], y, -0.001]))
    edges = np.arange(len(vertices), dtype=np.int32).reshape(-1, 2)
    return _line_set(np.asarray(vertices), edges, GRID_COLOR)


def static_scene(sequence: Sequence) -> list:
    all_points = np.vstack(
        [sequence.poses[name][:, :3] for name in (*BODY_NAMES, "head_rgbd")]
    )
    geometries: list = [
        _grid(all_points),
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=FRAME_SIZE * 1.3),
        _sphere(sequence.poses["astribot_chassis"][0, :3], BODY_POINT_COLOR),
    ]
    return geometries


class FrameActor:
    def __init__(self, size: float = FRAME_SIZE):
        self.mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self.transform = np.eye(4)

    def set(self, transform: np.ndarray) -> None:
        self.mesh.transform(transform @ np.linalg.inv(self.transform))
        self.transform = transform.copy()


class ImageReader:
    def __init__(self, path: Path):
        self.file = h5py.File(path, "r")
        self.data = self.file[IMAGE_PATH]

    def read(self, index: int) -> np.ndarray:
        image = np.asarray(self.data[index])
        image = np.clip(image[..., :3], 0, 255).astype(np.uint8)
        return image[..., ::-1].copy()  # Stored OpenCV frames are BGR.

    def close(self) -> None:
        self.file.close()


class CameraActor:
    def __init__(self, sequence: Sequence):
        self.sequence = sequence
        self.frame = FrameActor(FRAME_SIZE * 0.85)
        self.frustum = _line_set(
            np.zeros((5, 3)),
            np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]),
            FRUSTUM_COLOR,
        )
        self.reader = None
        self.plane = None
        self.local_plane = None
        self.local_corners = None
        if sequence.image_shape is not None:
            self._create_image_plane(*sequence.image_shape[:2])
            self.reader = ImageReader(sequence.path)
        else:
            self._create_frustum(16 / 9)

    def _create_frustum(self, aspect: float) -> None:
        half_h = CAMERA_DEPTH * np.tan(np.deg2rad(CAMERA_VERTICAL_FOV_DEG) / 2)
        half_w = half_h * aspect
        self.local_corners = np.array(
            [
                [-half_w, -half_h, CAMERA_DEPTH],
                [half_w, -half_h, CAMERA_DEPTH],
                [half_w, half_h, CAMERA_DEPTH],
                [-half_w, half_h, CAMERA_DEPTH],
            ]
        )

    def _create_image_plane(self, height: int, width: int) -> None:
        self.rows = np.unique(np.append(np.arange(0, height, IMAGE_STEP), height - 1))
        self.cols = np.unique(np.append(np.arange(0, width, IMAGE_STEP), width - 1))
        self._create_frustum(width / height)
        xs = np.interp(
            self.cols,
            [0, width - 1],
            [self.local_corners[0, 0], self.local_corners[1, 0]],
        )
        ys = np.interp(
            self.rows,
            [0, height - 1],
            [self.local_corners[0, 1], self.local_corners[2, 1]],
        )
        grid_x, grid_y = np.meshgrid(xs, ys)
        self.local_plane = np.column_stack(
            (grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, CAMERA_DEPTH))
        )
        triangles = []
        columns = len(self.cols)
        for row in range(len(self.rows) - 1):
            for col in range(columns - 1):
                a = row * columns + col
                triangles.extend(([a, a + columns, a + 1], [a + 1, a + columns, a + columns + 1]))
        self.plane = o3d.geometry.TriangleMesh()
        self.plane.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))

    def geometries(self) -> list:
        items = [self.frame.mesh, self.frustum]
        if self.plane is not None:
            items.append(self.plane)
        return items

    def set(self, index: int) -> None:
        transform = self.sequence.transforms["head_rgbd"][index]
        self.frame.set(transform)
        rotation, translation = transform[:3, :3], transform[:3, 3]
        local = np.vstack((np.zeros((1, 3)), self.local_corners))
        self.frustum.points = o3d.utility.Vector3dVector(local @ rotation.T + translation)
        if self.plane is not None and self.reader is not None:
            self.plane.vertices = o3d.utility.Vector3dVector(
                self.local_plane @ rotation.T + translation
            )
            image = self.reader.read(index)[self.rows][:, self.cols]
            self.plane.vertex_colors = o3d.utility.Vector3dVector(
                image.reshape(-1, 3).astype(np.float64) / 255.0
            )
            self.plane.compute_triangle_normals()
            self.plane.compute_vertex_normals()

    def close(self) -> None:
        if self.reader is not None:
            self.reader.close()


class HandActor:
    def __init__(self, color: np.ndarray):
        self.points = o3d.geometry.PointCloud()
        self.lines = _line_set(np.zeros((25, 3)), HAND_EDGES, color)
        self.color = color

    def set(self, points: np.ndarray) -> None:
        colors = np.repeat(self.color[None], 25, axis=0)
        self.points.points = o3d.utility.Vector3dVector(points)
        self.points.colors = o3d.utility.Vector3dVector(colors)
        self.lines.points = o3d.utility.Vector3dVector(points)


class BodyActor:
    def __init__(self):
        self.mesh = o3d.geometry.TriangleMesh()

    def set(self, points: np.ndarray) -> None:
        combined = o3d.geometry.TriangleMesh()
        radius = BODY_LINE_WIDTH * LINE_WIDTH_TO_RADIUS
        for start_index, end_index in BODY_EDGES:
            start, end = points[start_index], points[end_index]
            vector = end - start
            length = np.linalg.norm(vector)
            if length < 1e-8:
                continue
            segment = o3d.geometry.TriangleMesh.create_cylinder(
                radius=radius, height=length, resolution=12
            )
            rotation = Rotation.align_vectors(
                [vector / length], [[0.0, 0.0, 1.0]]
            )[0].as_matrix()
            segment.rotate(rotation, center=np.zeros(3))
            segment.translate((start + end) * 0.5)
            segment.paint_uniform_color(BODY_LINE_COLOR)
            segment.compute_vertex_normals()
            combined += segment

        self.mesh.vertices = o3d.utility.Vector3dVector(np.asarray(combined.vertices))
        self.mesh.triangles = o3d.utility.Vector3iVector(np.asarray(combined.triangles))
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(combined.vertex_colors)
        )
        self.mesh.vertex_normals = o3d.utility.Vector3dVector(
            np.asarray(combined.vertex_normals)
        )


class TrailActor:
    def __init__(self, points: np.ndarray, color: np.ndarray):
        self.color = color
        self.edges = np.column_stack(
            (np.arange(len(points) - 1), np.arange(1, len(points)))
        ).astype(np.int32)
        self.lines = _line_set(points, np.empty((0, 2), dtype=np.int32), color)

    def set(self, index: int) -> None:
        visible = self.edges[: min(max(index, 0), len(self.edges))]
        self.lines.lines = o3d.utility.Vector2iVector(visible)
        self.lines.colors = o3d.utility.Vector3dVector(
            np.repeat(self.color[None], len(visible), axis=0)
        )


class Viewer:
    def __init__(self, sequence: Sequence):
        self.sequence = sequence
        self.index = 0
        self.playing = True
        self.clock = float(sequence.times[0])
        self.last_tick = time.perf_counter()
        self.frames = {
            name: FrameActor(FRAME_SIZE * (1.1 if name == "astribot_chassis" else 1.0))
            for name in BODY_NAMES
        }
        self.body = BodyActor()
        self.hands = {side: HandActor(HAND_COLORS[side]) for side in sequence.hands}
        self.trails = {
            name: TrailActor(sequence.poses[name][:, :3], TRAJECTORY_COLOR)
            for name in ("astribot_chassis", "astribot_torso")
        }
        self.camera = CameraActor(sequence)
        self._apply(0)

    def geometries(self) -> list:
        items = [self.body.mesh, *(actor.mesh for actor in self.frames.values())]
        for hand in self.hands.values():
            items.extend((hand.points, hand.lines))
        items.extend(actor.lines for actor in self.trails.values())
        items.extend(self.camera.geometries())
        return items

    def _apply(self, index: int) -> None:
        self.index = int(np.clip(index, 0, self.sequence.frame_count - 1))
        for name, actor in self.frames.items():
            actor.set(self.sequence.transforms[name][self.index])
        body_points = np.vstack([self.sequence.poses[name][self.index, :3] for name in BODY_NAMES])
        self.body.set(body_points)
        for side, actor in self.hands.items():
            actor.set(self.sequence.hands[side][self.index])
        for actor in self.trails.values():
            actor.set(self.index)
        self.camera.set(self.index)

    def set_frame(self, index: int, vis, announce: bool = False) -> None:
        self._apply(index)
        for actor in self.frames.values():
            vis.update_geometry(actor.mesh)
        vis.update_geometry(self.body.mesh)
        for hand in self.hands.values():
            vis.update_geometry(hand.points)
            vis.update_geometry(hand.lines)
        for trail in self.trails.values():
            vis.update_geometry(trail.lines)
        for geometry in self.camera.geometries():
            vis.update_geometry(geometry)
        self.clock = float(self.sequence.times[self.index])
        self.last_tick = time.perf_counter()
        if announce:
            print(f"[INFO] frame {self.index + 1}/{self.sequence.frame_count}")

    def toggle(self) -> None:
        self.playing = not self.playing
        self.last_tick = time.perf_counter()
        print(f"[INFO] {'playing' if self.playing else 'paused'}")

    def step(self, amount: int, vis) -> None:
        self.playing = False
        self.set_frame(self.index + amount, vis, announce=True)

    def restart(self, vis) -> None:
        self.playing = False
        self.set_frame(0, vis, announce=True)

    def tick(self, vis) -> None:
        now = time.perf_counter()
        if not self.playing:
            self.last_tick = now
            return
        self.clock += now - self.last_tick
        self.last_tick = now
        next_index = int(np.searchsorted(self.sequence.times, self.clock, side="right") - 1)
        next_index = min(max(next_index, self.index), self.sequence.frame_count - 1)
        if next_index != self.index:
            self.set_frame(next_index, vis)
        if self.index == self.sequence.frame_count - 1:
            self.playing = False
            print("[INFO] playback finished")

    def close(self) -> None:
        self.camera.close()


def run(sequence: Sequence) -> None:
    viewer = Viewer(sequence)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        f"DexRoam robot trajectory - {sequence.path.name}",
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
    )
    options = vis.get_render_option()
    options.background_color = BACKGROUND_COLOR.copy()
    options.point_size = 5.0
    options.mesh_show_back_face = True

    first = True
    for geometry in static_scene(sequence) + viewer.geometries():
        vis.add_geometry(geometry, reset_bounding_box=first)
        first = False

    vis.register_key_callback(32, lambda _vis: (viewer.toggle(), False)[1])
    vis.register_key_callback(ord("N"), lambda _vis: (viewer.step(1, _vis), False)[1])
    vis.register_key_callback(ord("P"), lambda _vis: (viewer.step(-1, _vis), False)[1])
    vis.register_key_callback(ord("R"), lambda _vis: (viewer.restart(_vis), False)[1])

    print(f"[INFO] {sequence.frame_count} frames, duration={sequence.times[-1]:.2f}s")
    print("Controls: Space play/pause | N next | P previous | R restart")
    viewer.last_tick = time.perf_counter()
    try:
        while vis.poll_events():
            viewer.tick(vis)
            vis.update_renderer()
            time.sleep(0.001)
    finally:
        viewer.close()
        vis.destroy_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5_path", required=True, help="Robot episode HDF5 file.")
    parser.add_argument(
        "--fps",
        type=float,
        default=FALLBACK_FPS,
        help="Playback FPS when /time is unavailable (default: 30).",
    )
    parser.add_argument(
        "--mobile-position-source",
        choices=MOBILE_POSITION_SOURCES,
        default="relative_position",
        help="Chassis data in mobile_dict (default: relative_position).",
    )
    parser.add_argument(
        "--pose-group",
        choices=POSE_GROUPS,
        default="fk_poses_dict_from_state",
        help="Robot pose group (default: fk_poses_dict_from_state).",
    )
    parser.add_argument(
        "--pose-frame",
        choices=POSE_FRAMES,
        default="base",
        help="Coordinate frame of the selected robot poses (default: base).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = load_sequence(
        args.hdf5_path,
        args.fps,
        args.mobile_position_source,
        args.pose_group,
        args.pose_frame,
    )
    run(sequence)


if __name__ == "__main__":
    main()
