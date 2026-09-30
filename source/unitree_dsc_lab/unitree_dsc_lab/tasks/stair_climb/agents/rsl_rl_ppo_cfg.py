"""rsl_rl PPO runner config for the stair-climbing task.

Uses rsl-rl >= 4.0.0's split actor/critic configuration. Both the actor and
the critic are instantiated as :class:`StairClimbActorCritic` (a thin
``MLPModel`` subclass) by qualified-name resolution
(``module.path:ClassName``); the only difference between the two roles is
which observation set they read (``actor`` vs. ``critic``) and whether they
emit a stochastic Gaussian distribution.

Hyperparameters follow paper §III-D + Table II. The three-stage trainer
([`tasks/stair_climb/policy/ppo_runner.py`](../policy/ppo_runner.py)) wraps
this cfg and rebinds the learning rate / freeze schedule per stage.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


_MODEL_CFG_KEYS = {
    "class_name",
    "hidden_dims",
    "activation",
    "obs_normalization",
    "distribution_cfg",
}


def to_rsl_rl_dict(cfg: RslRlOnPolicyRunnerCfg) -> dict:
    """Serialize a runner config with only supported model arguments."""
    train_cfg = cfg.to_dict()
    for model_name in ("actor", "critic"):
        model_cfg = train_cfg.get(model_name)
        if isinstance(model_cfg, dict):
            train_cfg[model_name] = {key: value for key, value in model_cfg.items() if key in _MODEL_CFG_KEYS}
    return train_cfg


_ACTOR_CRITIC_QUALNAME = (
    "unitree_dsc_lab.tasks.stair_climb.policy.actor_critic:StairClimbActorCritic"
)


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage-1 PPO runner cfg (paper §III-D)."""

    num_steps_per_env = 24
    # Action clipping applied by RslRlVecEnvWrapper.
    clip_actions = 1.0
    max_iterations = 6000
    save_interval = 100
    experiment_name = "stair_climb_g1_23dof"
    # Deprecated in rsl-rl >= 4.0.0 (per-model `obs_normalization` is used
    # instead). Keep the field assignment to satisfy `RslRlBaseRunnerCfg`'s
    # MISSING sentinel — the value is unused.
    empirical_normalization = True

    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }

    actor = RslRlMLPModelCfg(
        class_name=_ACTOR_CRITIC_QUALNAME,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            class_name="GaussianDistribution",
            init_std=1.0,
            std_type="log",
        ),
    )

    critic = RslRlMLPModelCfg(
        class_name=_ACTOR_CRITIC_QUALNAME,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
    )

    # Deprecated legacy field — required because `RslRlOnPolicyRunnerCfg.policy`
    # is annotated `MISSING`. Set to ``None`` so the new `actor`/`critic` path
    # is the only one in effect.
    policy = None

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G129DofDex11PPORunnerCfg(BasePPORunnerCfg):
    """PPO config with a separate log namespace for the 29-DoF task."""

    experiment_name = "stair_climb_g1_29dof_dex1_1"
