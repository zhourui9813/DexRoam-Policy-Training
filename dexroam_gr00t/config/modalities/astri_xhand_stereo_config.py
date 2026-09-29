"""GR00T modality config aligned with ``astribot_xhand`` deploy I/O.

Matches:
  - ``AstriRealEnvironment.get_observation()`` keys / semantics (robot still exposes 7-D poses;
    ``GrootBridgePolicy`` lifts them to 9-D xyz+rot6d for the policy server).
  - This LeRobot dataset stores **53-D state / 56-D action** rot6d EEF vectors. State order follows
    the converter output (EEF poses first, dex hands next); action order follows deploy command order
    (dex hands first, EEF commands next).

Register with ``NEW_EMBODIMENT`` (same pattern as ``astri_config.py``). Use as::

    --embodiment-tag NEW_EMBODIMENT --modality-config-path astri_deploy_match_config.py

**Do not import** another file that also calls ``register_modality_config(..., NEW_EMBODIMENT)``
in the same process.

State / action **key order** must match this dataset's LeRobot ``meta/modality.json`` so flattened
``observation.state`` / ``action`` vectors concatenate to **53** / **56** without slice drift.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# LeRobot ``modality.json`` maps ``head_rgb`` -> observation.images.head_rgb.
astri_deploy_match_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head_stereo_left", "head_stereo_right"],
    ),
    # Proprio 53-D total: 9 + 9 + 9 EEF poses, 12 + 12 dex hands, 2 head joints.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "right_arm",
            "torso",
            "left_dex_hand",
            "right_dex_hand",
            "head_joint",
        ],
    ),
    # Action 56-D per step (12 + 12 dex hands, 9 + 9 + 9 EEF commands, 2 head, 3 mobile delta).
    "action": ModalityConfig(
        delta_indices=list(range(32)),
        modality_keys=[
            "left_dex_hand",
            "right_dex_hand",
            "left_arm",
            "right_arm",
            "torso",
            "head_joint",
            "mobile_relative",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(astri_deploy_match_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
