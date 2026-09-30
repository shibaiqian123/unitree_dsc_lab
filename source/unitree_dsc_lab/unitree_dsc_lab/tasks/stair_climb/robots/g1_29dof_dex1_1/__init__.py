import gymnasium as gym


gym.register(
    id="Unitree-G1-29dof-Dex1-1-StairClimb-v0",
    entry_point=(
        "unitree_dsc_lab.tasks.stair_climb.robots.g1_29dof_dex1_1.stair_env:"
        "G129DofDex11StairClimbEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.stair_env_cfg:G129DofDex11StairClimbEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.stair_env_cfg:G129DofDex11StairClimbPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_dsc_lab.tasks.stair_climb.agents.rsl_rl_ppo_cfg:"
            "G129DofDex11PPORunnerCfg"
        ),
    },
)
