import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


STATE_DIM = 53
ACTION_DIM = 56
LEFT_STEREO_KEY = "observation/images/head_stereo_left"
RIGHT_STEREO_KEY = "observation/images/head_stereo_right"
STATE_KEY = "observation/state"


def make_dexroam_example() -> dict:
    """Creates a random deploy-match stereo EEF example for the DexRoam policy."""
    return {
        LEFT_STEREO_KEY: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        RIGHT_STEREO_KEY: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        STATE_KEY: np.random.rand(STATE_DIM).astype(np.float32),
        "actions": np.random.rand(10, ACTION_DIM).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _as_vector(data: dict, key: str, dim: int) -> np.ndarray:
    value = np.asarray(data[key], dtype=np.float32)
    if value.shape[-1] != dim:
        raise ValueError(f"Expected {key} to have last dimension {dim}, got shape {value.shape}.")
    return value


@dataclasses.dataclass(frozen=True)
class DexRoamInputs(transforms.DataTransformFn):
    """Converts deploy-match stereo EEF inputs to the standard pi model format."""

    model_type: _model.ModelType
    action_dim: int

    def __call__(self, data: dict) -> dict:
        if self.action_dim != ACTION_DIM:
            raise ValueError(f"DexRoam deploy-match expects action_dim={ACTION_DIM}, got {self.action_dim}.")

        state = _as_vector(data, STATE_KEY, STATE_DIM)
        left_image = _parse_image(data[LEFT_STEREO_KEY])
        right_image = _parse_image(data[RIGHT_STEREO_KEY])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (left_image, right_image, np.zeros_like(left_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        action_key = "actions" if "actions" in data else "action" if "action" in data else None
        if action_key is not None:
            actions = np.asarray(data[action_key], dtype=np.float32)
            if actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Expected DexRoam actions to have last dimension {self.action_dim}, got {actions.shape}."
                )
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class DexRoamOutputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
