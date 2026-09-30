"""Stair-climbing task config for the 29-DoF G1 with Dex1-1 grippers.

The source URDF has 33 movable joints: 29 G1 body joints and four gripper
joints. The locomotion policy controls and observes the 29 body joints; the
Dex1-1 joints remain at their default positions under their own PD actuator.
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_dsc_lab.assets.robots.unitree_29dof_dex1_1 import (
    G1_29DOF_BODY_JOINT_NAMES,
    G1_29DOF_DEX1_1_CFG as ROBOT_CFG,
)
from unitree_dsc_lab.tasks.stair_climb import mdp
from unitree_dsc_lab.tasks.stair_climb.robots.g1_23dof.stair_env_cfg import (
    ActionsCfg as BaseActionsCfg,
)
from unitree_dsc_lab.tasks.stair_climb.robots.g1_23dof.stair_env_cfg import (
    G1StairClimbEnvCfg as BaseG1StairClimbEnvCfg,
)
from unitree_dsc_lab.tasks.stair_climb.robots.g1_23dof.stair_env_cfg import (
    ObservationsCfg as BaseObservationsCfg,
)
from unitree_dsc_lab.tasks.stair_climb.robots.g1_23dof.stair_env_cfg import (
    RewardsCfg as BaseRewardsCfg,
)
from unitree_dsc_lab.tasks.stair_climb.robots.g1_23dof.stair_env_cfg import (
    StairSceneCfg as BaseStairSceneCfg,
)


def _body_joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot",
        joint_names=G1_29DOF_BODY_JOINT_NAMES,
        preserve_order=True,
    )


@configclass
class StairSceneCfg(BaseStairSceneCfg):
    """Base stair scene with the local 29-DoF Dex1-1 robot asset."""

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ActionsCfg(BaseActionsCfg):
    """Position targets for the 29 G1 body joints, in SDK order."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_29DOF_BODY_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg(BaseObservationsCfg):
    """Policy and critic observations restricted to the 29 body joints."""

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _body_joint_cfg()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _body_joint_cfg()},
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

    @configclass
    class CriticCfg(BaseObservationsCfg.CriticCfg):
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _body_joint_cfg()},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _body_joint_cfg()},
            scale=0.05,
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg(BaseRewardsCfg):
    """Rewards using the shoulder, elbow, and wrist joints of the 29-DoF G1."""

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            ),
        },
    )

    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )


@configclass
class G129DofDex11StairClimbEnvCfg(BaseG1StairClimbEnvCfg):
    """Training config with a 29-dimensional locomotion action space."""

    scene: StairSceneCfg = StairSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Command visualization pulls a remote USD arrow asset; keep the
        # default task usable in headless/offline training environments.
        self.commands.base_velocity.debug_vis = False


@configclass
class G129DofDex11StairClimbPlayEnvCfg(G129DofDex11StairClimbEnvCfg):
    """Play config with fewer parallel environments and no random pushes."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 4
        self.scene.terrain.terrain_generator.num_cols = 8
        self.events.push_robot = None
