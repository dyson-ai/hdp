from typing import Optional

import pytorch_kinematics as pk
import torch
import torch.nn as nn
import torch.nn.functional as F

import agents.peract_diffusion.diff_traj.utils as utils


class DiffRobot(nn.Module):
    def __init__(
        self,
        urdf_path: str,
        joint_name: str = "Pandatip",
        offset_mat: Optional[torch.tensor] = None,
        sim: bool = True,
    ) -> None:
        super().__init__()

        self._robot = pk.build_serial_chain_from_urdf(
            open(urdf_path).read(), joint_name
        )
        self._sim = sim

        if sim:
            if offset_mat is None:
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
            self.register_buffer("_offset_mat", offset_mat)

    def to(self, device: torch.device):
        super().to(device=device)
        self._robot.to(device=device)

    def forward_kinematics_batch(self, joints: torch.tensor):
        assert len(joints.shape) == 2
        matrix = self._robot.forward_kinematics(joints).get_matrix()
        if self._sim:
            matrix = torch.bmm(self._offset_mat.repeat(matrix.size(0), 1, 1), matrix)
        pred_poses = utils.matrix_to_rlb_pose(matrix)
        pred_poses = utils.proc_quaternion(pred_poses)

        return pred_poses

    def inverse_kinematics_autodiff_single_step_batch_pt(
        self,
        joints: torch.tensor,
        target_poses: torch.tensor,
        alpha: float = 500,
        use_quat: bool = False,
    ):
        joints = joints.detach()
        joints.requires_grad = True

        pred_poses = self.forward_kinematics_batch(joints)
        pred_trans = pred_poses[:, :3]
        pred_quat = pred_poses[:, 3:]

        t_err = F.mse_loss(target_poses[:, 0:3], pred_trans, reduction="none")
        r_err = F.mse_loss(target_poses[:, 3:], pred_quat, reduction="none")
        t_err = t_err.sum(dim=-1).mean()
        r_err = r_err.sum(dim=-1).mean()

        loss = t_err
        if use_quat:
            loss += r_err

        loss.backward()
        joints_updated = joints - joints.grad * alpha
        return joints_updated.detach(), {"trans": t_err, "rot": r_err}
