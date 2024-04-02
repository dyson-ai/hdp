from typing import Dict

import torch
import torch.nn as nn

from agents.peract_diffusion.diff_traj import utils
from agents.peract_diffusion.diff_traj.diffusion import GaussianInvDynDiffusion
from agents.peract_diffusion.diff_traj.robot import DiffRobot


class MultiLevelDiffusion(nn.Module):
    def __init__(
        self,
        models: Dict[str, GaussianInvDynDiffusion],
        diff_optim: bool = True,
        diff_optim_steps: int = 50,
        diff_lr: float = 0.1,
    ) -> None:
        super().__init__()
        self._models = nn.ModuleDict(models)
        self._diff_optim = diff_optim
        self._diff_optim_steps = diff_optim_steps
        self._diff_lr = diff_lr

    @property
    def robot_offset(self):
        return self._models["joint_positions"].robot_offset[None, None]

    @property
    def robot_rot(self):
        return utils.matrix_to_quaternion(
            self._models["joint_positions"].robot_rot[None, None]
        )

    def conditional_sample(
        self, cond, horizon=None, diff_gripper=True, *args, **kwargs
    ):
        samples = {"multi": {}}
        for k, v in self._models.items():
            cond, kwargs = self._form_diffusion_batch(k, diff_gripper, **kwargs)
            del kwargs["x"]
            sample = v.conditional_sample(cond, horizon, *args, **kwargs)
            samples.update(sample)

        if self._diff_optim:
            robot = kwargs["robot"]
            joints = samples["joint_positions"]["joint_positions"].clone().detach()
            poses = samples["gripper_poses"]["traj"].clone().view(-1, 7).detach()

            batch_size, seq_len = joints.size(0), joints.size(1)
            joints = joints.view(-1, 7)

            for _ in range(self._diff_optim_steps):
                joints, losses = robot.inverse_kinematics_autodiff_single_step_batch_pt(
                    joints, poses, self._diff_lr
                )

            last_poses = poses.view(batch_size, seq_len, 7)[:, -1]
            last_joints = joints.view(batch_size, seq_len, 7)[:, -1]

            for _ in range(self._diff_optim_steps):
                (
                    last_joints,
                    losses,
                ) = robot.inverse_kinematics_autodiff_single_step_batch_pt(
                    last_joints, last_poses, self._diff_lr / 10
                )

            # last_joints = kwargs["joint_positions"][:, -1]
            joints = joints.view(batch_size, seq_len, 7)
            if diff_gripper:
                joints[:, -1] = joints[:, -2]
            else:
                joints[:, -1] = last_joints
            joints = joints.view(-1, 7)

            samples["multi"]["joint_positions"] = joints.view(batch_size, seq_len, 7)
            diff = (
                samples["multi"]["joint_positions"][:, -1]
                - kwargs["joint_positions"][:, -1]
            )
            diff = diff.mean(dim=0)

            trajs = robot.forward_kinematics_batch(joints).view(batch_size, seq_len, 7)

            samples["multi"]["traj"] = trajs
        else:
            samples["multi"] = samples["joint_positions"]

        return samples

    def loss(self, x, cond, **kwargs):
        losses = 0
        infos = {}
        for k, v in self._models.items():
            cond, kwargs = self._form_diffusion_batch(k, **kwargs)
            loss, info = v.loss(cond=cond, **kwargs)
            losses += loss
            infos.update(info)

        return losses, infos

    def _form_diffusion_batch(self, diffusion_var, diff_gripper=True, **kwargs):
        joint_positions = kwargs["joint_positions"]
        robot = kwargs["robot"]
        batch_size, seq_len = joint_positions.size(0), joint_positions.size(1)

        if diff_gripper:
            trajs = robot.forward_kinematics_batch(
                joint_positions.view(-1, 7),
            ).view(batch_size, seq_len, 7)

            kwargs["gripper_poses"] = trajs

        if diffusion_var == "gripper_poses":
            x = kwargs["gripper_poses"]
            new_cond = {
                0: x[:, 0],
                -1: x[:, -1],
            }
        elif diffusion_var == "joint_positions":
            x = kwargs["joint_positions"]
            new_cond = {0: x[:, 0]}
        else:
            raise NotImplementedError

        ret_kwargs = {k: v for k, v in kwargs.items()}
        ret_kwargs["x"] = x

        return new_cond, ret_kwargs
