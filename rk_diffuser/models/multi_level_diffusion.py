from typing import Dict

import torch
import torch.nn as nn

from rk_diffuser import utils
from rk_diffuser.models.diffusion import GaussianDynDiffusion


class MultiLevelDiffusion(nn.Module):
    """The multi-level diffusion policy.

    We train both the joint and the gripper pose diffusion models, and use
    differentiable kinematics to distill the gripper pose information into
    the joint positions.
    """

    def __init__(
        self,
        models: Dict[str, GaussianDynDiffusion],
        diff_optim: bool = True,
        diff_optim_steps: int = 50,
        diff_lr: float = 0.1,
        pose_augment: bool = False,
        sim: bool = True,
    ) -> None:
        """
        Initializes an instance of the class.

        Parameters:
            models (Dict[str, GaussianDynDiffusion]): A dictionary of models.
            diff_optim (bool, optional): Flag indicating whether differentiable optimization should be used. Defaults to True.
            diff_optim_steps (int, optional): The number of differentiable optimization steps. Defaults to 50.
            diff_lr (float, optional): The learning rate for differentiable optimization. Defaults to 0.1.
            pose_augment (bool, optional): Flag indicating whether pose augmentation should be used. Defaults to False.
            sim (bool): simulation or real_world.

        Returns:
            None
        """
        super().__init__()
        self._models = nn.ModuleDict(models)
        self._diff_optim = diff_optim
        self._diff_optim_steps = diff_optim_steps
        self._diff_lr = diff_lr
        self._pose_augment = pose_augment
        self._sim = sim

    @property
    def robot_offset(self):
        """Get the offset value of the robot."""
        return self._models["joint_positions"].robot_offset[None, None]

    @property
    def robot_rot(self):
        """Get the rotation matrix of the robot."""
        return utils.matrix_to_quaternion(
            self._models["joint_positions"].robot_rot[None, None]
        )

    def conditional_sample(
        self, cond: list, horizon: int = None, *args, **kwargs
    ) -> dict:
        """This function wrap the conditional sample function of each diffusion models."""
        samples = {"multi": {}}
        for k, v in self._models.items():
            cond, kwargs = self._form_diffusion_batch(k, False, train=False, **kwargs)
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
            if "diffusion_hist" in samples["gripper_poses"]:
                diffusion_hist = samples["gripper_poses"]["diffusion_hist"]
                diffusion_hist[:, -1] = trajs
                samples["multi"]["diffusion_hist"] = diffusion_hist
        else:
            samples["multi"] = samples["joint_positions"]

        return samples

    def loss(self, x: torch.tensor, cond: dict, **kwargs) -> tuple:
        """
        Calculates the loss and collects information for each model in the DiffusionModel.

        Args:
            x (torch.tensor): The input tensor.
            cond (dict): The conditioning information.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the total loss and a dictionary of information for each model.
        """
        losses = 0
        infos = {}
        for k, v in self._models.items():
            cond, kwargs = self._form_diffusion_batch(
                k, self._pose_augment, train=True, **kwargs
            )
            loss, info = v.loss(cond=cond, **kwargs)
            losses += loss
            infos.update(info)

        return losses, infos

    def _form_diffusion_batch(
        self, diffusion_var: str, pose_augment: bool, train: bool = False, **kwargs
    ) -> dict:
        """Form the diffusion model input data.

        Args:
            diffusion_var: diffusion variable, e.g., gripper_poses or joint_positions.
            pose_augment: whether to use pose augmentation.
            **kwargs: Additional keyword arguments.

        Returns:
            The condition and input data for the diffusion model.
        """
        joint_positions = kwargs["joint_positions"]
        robot = kwargs["robot"]
        batch_size, seq_len = joint_positions.size(0), joint_positions.size(1)

        trajs = robot.forward_kinematics_batch(
            joint_positions.view(-1, 7),
        ).view(batch_size, seq_len, 7)

        gt_poses = kwargs["gripper_poses"]

        if self._sim and train:
            # In RLBench, the gripper poses are not fully aligned with the FK output because
            # of the imperfect dynamics. We replace it with the FK results instead. However,
            # during evaluation or in real-world, this is not needed.
            kwargs["gripper_poses"] = trajs

        def _get_rand_offset(ts):
            return (
                torch.randn_like(ts, dtype=ts.dtype, device=ts.device).clamp(-1, 1)
                * 0.05
            )

        start, end = gt_poses[:, 0], gt_poses[:, -1]
        if pose_augment:
            start += _get_rand_offset(start)
            end += _get_rand_offset(end)

        if diffusion_var == "gripper_poses":
            x = kwargs["gripper_poses"]
            new_cond = {
                0: start,
                -1: end,
            }

        elif diffusion_var == "joint_positions":
            x = kwargs["joint_positions"]
            new_cond = {0: x[:, 0]}
        else:
            raise NotImplementedError

        ret_kwargs = {k: v for k, v in kwargs.items()}
        ret_kwargs["x"] = x
        ret_kwargs["start"] = start
        ret_kwargs["end"] = end

        return new_cond, ret_kwargs
