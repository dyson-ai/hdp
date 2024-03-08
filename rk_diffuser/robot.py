from typing import Optional

import pytorch_kinematics as pk
import torch
import torch.nn as nn
import torch.nn.functional as F

import rk_diffuser.utils as utils


class DiffRobot(nn.Module):
    """A robot object with differentiable kinematics."""

    def __init__(
        self,
        urdf_path: str,
        joint_name: str = "Pandatip",
        offset_mat: Optional[torch.tensor] = None,
        sim: bool = True,
    ) -> None:
        """
        Initialize the robot object.

        Args:
            urdf_path (str): The path to the URDF file.
            joint_name (str, optional): The name of the joint. Defaults to "Pandatip".
            offset_mat (Optional[torch.tensor], optional): The offset matrix. Defaults to None.

        Returns:
            None
        """
        super().__init__()

        self._robot = pk.build_serial_chain_from_urdf(
            open(urdf_path).read(), joint_name
        )
        if offset_mat is None:
            if sim:
                offset_mat = torch.tensor(
                    [
                        [
                            [0.9979, -0.0121, -0.0564, 0.0783],
                            [0.0122, 0.9994, 0.0036, -0.0029],
                            [0.0561, -0.0039, 0.9980, 0.0130],
                            [0.0000, 0.0000, 0.0000, 0.0000],
                        ]
                    ],
                    dtype=torch.float32,
                )
            else:  # for real robot
                offset_mat = torch.tensor(
                    [
                        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    ],
                    dtype=torch.float32,
                )
        self.register_buffer("_offset_mat", offset_mat)

    def to(self, device: torch.device) -> None:
        """
        Move the model and the robot to the specified device.

        Args:
            device (torch.device): The device to move the model and robot to.

        Returns:
            None
        """
        super().to(device=device)
        self._robot.to(device=device)

    def forward_kinematics_batch(self, joints: torch.tensor) -> torch.tensor:
        """
        Calculate the forward kinematics for a batch of joint configurations.

        Args:
            joints (torch.tensor): joint positions of the robot of shape [N, num_joints]

        Returns:
            torch.tensor: A tensor containing the predicted EE poses of the robot.
        """
        assert len(joints.shape) == 2
        matrix = self._robot.forward_kinematics(joints).get_matrix()
        matrix = torch.bmm(self._offset_mat.repeat(matrix.size(0), 1, 1), matrix)
        pred_poses = utils.matrix_to_rlb_pose(matrix)
        pred_poses = utils.proc_quaternion(pred_poses)

        return pred_poses

    def inverse_kinematics_autodiff_single_step_batch_pt(
        self,
        joints: torch.tensor,
        target_poses: torch.tensor,
        alpha: float = 500,
    ) -> tuple:
        """
        Perform a single step of inverse kinematics using automatic differentiation in a batched manner.

        Args:
            joints (torch.tensor): The input joint angles. Shape: (batch_size, num_joints)
            target_poses (torch.tensor): The target poses. Shape: (batch_size, 7)
            alpha (float, optional): The learning rate for the update step. Defaults to 500.

        Returns:
            tuple: A tuple containing the updated joint angles and a dictionary containing the translation
                and rotation errors.
        """
        joints = joints.detach()
        joints.requires_grad = True

        pred_poses = self.forward_kinematics_batch(joints)
        pred_trans = pred_poses[:, :3]
        pred_quat = pred_poses[:, 3:]

        t_err = F.mse_loss(target_poses[:, 0:3], pred_trans, reduction="none")
        r_err = F.mse_loss(target_poses[:, 3:], pred_quat, reduction="none")
        t_err = t_err.sum(dim=-1).mean()
        r_err = r_err.sum(dim=-1).mean()

        # Note that we only use translation error now as we observe rotation error is unstable
        # This is made possible given the predicted joint positions have provided a nice initial guess
        loss = t_err
        loss.backward()
        joints_updated = joints - joints.grad * alpha
        return joints_updated.detach(), {"trans": t_err, "rot": r_err}
