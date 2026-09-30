"""Play a trained stair-climbing policy in Isaac Sim."""

from __future__ import annotations

import argparse
import os
import time

# PYTHON_ARGCOMPLETE_OK
try:
    import argcomplete
except ModuleNotFoundError:
    argcomplete = None

from cli_args import add_rsl_rl_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a trained stair-climbing checkpoint.")
    add_rsl_rl_args(parser)
    parser.add_argument("--num_play_envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--disable_fabric", action="store_true")
    parser.add_argument("--real_time", action="store_true", help="Limit playback to the environment control rate.")
    parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many steps; zero runs indefinitely.")
    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if not args.checkpoint:
        parser.error("--checkpoint is required for playback")
    checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint))
    if not os.path.isfile(checkpoint):
        parser.error(f"checkpoint does not exist: {checkpoint}")
    if args.num_play_envs < 1:
        parser.error("--num_play_envs must be at least 1")
    if args.max_steps < 0:
        parser.error("--max_steps must be non-negative")

    # Isaac Sim must be launched before importing Isaac Lab task modules.
    from isaaclab.app import AppLauncher  # type: ignore

    app_launcher = AppLauncher(headless=args.headless, device=args.device)
    simulation_app = app_launcher.app
    env = None

    try:
        import gymnasium as gym
        import torch

        import unitree_dsc_lab  # noqa: F401  registers gym tasks

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry

        from unitree_dsc_lab.tasks.stair_climb.agents.rsl_rl_ppo_cfg import to_rsl_rl_dict
        from unitree_dsc_lab.tasks.stair_climb.perception.encoder import BEVStudentEncoder
        from unitree_dsc_lab.tasks.stair_climb.policy.ppo_runner import ThreeStagePPORunner

        env_cfg = load_cfg_from_registry(args.task, "play_env_cfg_entry_point")
        env_cfg.scene.num_envs = args.num_play_envs
        env_cfg.seed = args.seed
        env_cfg.sim.device = args.device
        env_cfg.sim.use_fabric = not args.disable_fabric
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.eye = (4.5, 4.5, 2.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.8)

        runner_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        train_cfg = to_rsl_rl_dict(runner_cfg)

        env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)

        runner = ThreeStagePPORunner(
            env,
            train_cfg,
            BEVStudentEncoder(),
            log_dir=None,
            device=args.device,
        )
        runner.load(
            checkpoint,
            load_cfg={"actor": True, "critic": True, "optimizer": False},
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=env.device)
        obs = env.get_observations().to(env.device)
        step_dt = env.unwrapped.step_dt
        steps = 0

        print(f"[INFO] Playing checkpoint: {checkpoint}")
        print(f"[INFO] Task: {args.task} | environments: {args.num_play_envs} | device: {args.device}")

        while simulation_app.is_running() and (args.max_steps == 0 or steps < args.max_steps):
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy.reset(dones)
            steps += 1

            if args.real_time:
                sleep_time = step_dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
