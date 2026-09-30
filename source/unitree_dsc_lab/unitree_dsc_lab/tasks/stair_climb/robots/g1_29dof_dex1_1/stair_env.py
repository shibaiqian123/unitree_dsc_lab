"""Stair-climbing environment for the 29-DoF G1 with Dex1-1 grippers."""

from unitree_dsc_lab.tasks.stair_climb.robots.g1_23dof.stair_env import (
    G1StairClimbEnv,
)


class G129DofDex11StairClimbEnv(G1StairClimbEnv):
    """G1 stair environment using the local 29-DoF Dex1-1 asset."""

