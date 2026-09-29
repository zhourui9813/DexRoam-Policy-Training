"""DexRoam policy training configs with 32-step action horizon."""

from pathlib import Path

from flax import nnx
import openpi.models.pi0_config as pi0_config
import openpi.policies.dexroam_policy as dexroam_policy
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.weight_loaders as weight_loaders


DEXROAM_PI05_ROOT = Path(__file__).resolve().parents[3]
PI05_BASE_PARAMS = DEXROAM_PI05_ROOT / "checkpoints" / "pretrained" / "openpi" / "pi05_base" / "params"


def get_dexroam_train_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import DataConfig
    from openpi.training.config import LeRobotDexRoamDataConfig
    from openpi.training.config import TrainConfig

    return [
        TrainConfig(
            name="DEXROAM_EXAMPLE",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=dexroam_policy.ACTION_DIM,
                action_horizon=32,
                max_token_len=256,
            ),
            data=LeRobotDexRoamDataConfig(
                repo_id="Astribot-Xhand-data/robot_push_chair_and_close_laptop",
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
            num_workers=16,
            batch_size=4,
            fsdp_devices=1,
            weight_loader=weight_loaders.HighDimActionWeightLoader(str(PI05_BASE_PARAMS)),
        ),
    ]
