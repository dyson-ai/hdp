import getpass
import logging
import os
from typing import List, Union

import numpy as np
import plotly.graph_objects as go
import torch
import torchvision.transforms.functional as tvf
from yarr.agents.agent import ActResult, Agent, Summary

from rk_diffuser.models.diffusion import GaussianDynDiffusion
from rk_diffuser.models.multi_level_diffusion import MultiLevelDiffusion
from rk_diffuser.robot import DiffRobot
from peract.agents.peract_diffusion.qattention_peract_bc_agent import (
    QAttentionPerActBCAgent,
)
from helpers import utils

NAME = "QAttentionStackAgent"


class QAttentionStackAgent(Agent):
    def __init__(
        self,
        qattention_agents: List[QAttentionPerActBCAgent],
        diffusion_agent: Union[MultiLevelDiffusion, GaussianDynDiffusion],
        rotation_resolution: float,
        robot: DiffRobot,
        camera_names: List[str],
        scene_bounds: List[float],
        rotation_prediction_depth: int = 0,
        diff_model_path: str = "",
        diff_gripper: bool = True,
    ):
        super(QAttentionStackAgent, self).__init__()
        self._qattention_agents = qattention_agents
        self._diffusion_agent = diffusion_agent
        self._robot = robot
        self._rotation_resolution = rotation_resolution
        self._camera_names = camera_names
        self._scene_bounds = scene_bounds
        self._rotation_prediction_depth = rotation_prediction_depth
        self._diff_model_path = diff_model_path
        self._keyframes = []
        self._diff_gripper = diff_gripper
        # assert os.path.exists(self._diff_model_path)

    def build(self, training: bool, device=None) -> None:
        self._device = device
        if self._device is None:
            self._device = torch.device("cpu")
        for qa in self._qattention_agents:
            qa.build(training, device)

    def update(self, step: int, replay_sample: dict) -> dict:
        priorities = 0
        total_losses = 0.0
        for qa in self._qattention_agents:
            update_dict = qa.update(step, replay_sample)
            replay_sample.update(update_dict)
            total_losses += update_dict["total_loss"]
        return {
            "total_losses": total_losses,
        }

    def act(self, step: int, observation: dict, deterministic=False) -> ActResult:
        observation_elements = {}
        translation_results, rot_grip_results, ignore_collisions_results = [], [], []
        infos = {}
        for depth, qagent in enumerate(self._qattention_agents):
            act_results = qagent.act(step, observation, deterministic)
            attention_coordinate = (
                act_results.observation_elements["attention_coordinate"].cpu().numpy()
            )
            observation_elements["attention_coordinate_layer_%d" % depth] = (
                attention_coordinate[0]
            )

            translation_idxs, rot_grip_idxs, ignore_collisions_idxs = act_results.action
            translation_results.append(translation_idxs)
            if rot_grip_idxs is not None:
                rot_grip_results.append(rot_grip_idxs)
            if ignore_collisions_idxs is not None:
                ignore_collisions_results.append(ignore_collisions_idxs)

            observation["attention_coordinate"] = act_results.observation_elements[
                "attention_coordinate"
            ]
            observation["prev_layer_voxel_grid"] = act_results.observation_elements[
                "prev_layer_voxel_grid"
            ]
            observation["prev_layer_bounds"] = act_results.observation_elements[
                "prev_layer_bounds"
            ]

            for n in self._camera_names:
                px, py = utils.point_to_pixel_index(
                    attention_coordinate[0],
                    observation["%s_camera_extrinsics" % n][0, 0].cpu().numpy(),
                    observation["%s_camera_intrinsics" % n][0, 0].cpu().numpy(),
                )
                pc_t = torch.tensor(
                    [[[py, px]]], dtype=torch.float32, device=self._device
                )
                observation["%s_pixel_coord" % n] = pc_t
                observation_elements["%s_pixel_coord" % n] = [py, px]

            infos.update(act_results.info)

        rgai = torch.cat(rot_grip_results, 1)[0].cpu().numpy()
        ignore_collisions = float(
            torch.cat(ignore_collisions_results, 1)[0].cpu().numpy()
        )
        observation_elements["trans_action_indicies"] = (
            torch.cat(translation_results, 1)[0].cpu().numpy()
        )
        observation_elements["rot_grip_action_indicies"] = rgai
        quat = utils.discrete_euler_to_quaternion(
            rgai[-4:-1], self._rotation_resolution
        )
        trans = (
            act_results.observation_elements["attention_coordinate"].cpu().numpy()[0]
        )

        _cond, _continuous_data_dict = self._prepare_diffusion_data_dict(
            observation,
            trans,
            quat,
        )
        diff_output = self._diffusion_agent.conditional_sample(
            cond=_cond, diff_gripper=self._diff_gripper, **_continuous_data_dict
        )
        traj = diff_output["multi"]["traj"]
        joint_positions = diff_output["multi"]["joint_positions"].cpu().detach().numpy()
        joint_positions = joint_positions.reshape(-1)
        continuous_action = np.concatenate([joint_positions, rgai[-1:]])
        gt_traj = torch.stack([_cond[0], _cond[-1]], dim=1)

        # Plotting function is for my debugging use

        # self._plot(
        #     traj.cpu().detach().numpy(),
        #     _continuous_data_dict["pcds"].cpu().detach().numpy(),
        #     _continuous_data_dict["rgbs"].cpu().detach().numpy(),
        #     gt_trajs=gt_traj.cpu().detach().numpy(),
        # )
        return ActResult(
            continuous_action, observation_elements=observation_elements, info=infos
        )

    def _plot(self, predicted_trajs, pcds, rgbs, gt_trajs=None):
        sampled_trajs = predicted_trajs[0]
        tx, ty, tz = (
            sampled_trajs[:, 0],
            sampled_trajs[:, 1],
            sampled_trajs[:, 2],
        )

        if gt_trajs is not None:
            gt_traj = gt_trajs[0]
            gx, gy, gz = gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2]

        pcds = pcds.reshape(-1, 3)

        bound_min, bound_max = self._scene_bounds

        rgbs = rgbs.reshape(-1, 3).astype(np.uint8)

        bound = np.array([bound_min, bound_max], dtype=np.float32)

        pcd_mask = (pcds > bound[0:1]) * (pcds < bound[1:2])
        pcd_mask = np.all(pcd_mask, axis=1)
        indices = np.where(pcd_mask)[0]

        pcds = pcds[indices]
        rgbs = rgbs[indices]

        rgb_strings = [
            f"rgb{rgbs[i][0],rgbs[i][1],rgbs[i][2]}" for i in range(len(rgbs))
        ]

        pcd_plots = [
            go.Scatter3d(
                x=pcds[:, 0],
                y=pcds[:, 1],
                z=pcds[:, 2],
                mode="markers",
                marker=dict(
                    size=8,
                    color=rgb_strings,
                ),
            )
        ]

        plot_data = [
            go.Scatter3d(
                x=tx,
                y=ty,
                z=tz,
                mode="markers",
                marker=dict(size=6, color="blue"),
            ),
        ] + pcd_plots

        if gt_trajs is not None:
            gt_plot = [
                go.Scatter3d(
                    x=gx,
                    y=gy,
                    z=gz,
                    mode="markers",
                    marker=dict(size=10, color="red"),
                )
            ]
            plot_data += gt_plot

        fig = go.Figure(plot_data)
        path = f"/home/{getpass.getuser()}/hdp_plots"
        os.makedirs(path, exist_ok=True)
        existings = os.listdir(path)
        fig.write_html(os.path.join(path, f"vis_{len(existings)}.html"))

    def _prepare_diffusion_data_dict(
        self, observation: dict, trans: np.array, quat: np.array
    ) -> dict:
        start_pose = observation["gripper_pose"][0]
        joints = observation["joint_position"][0]
        if quat[-1] < 0:
            quat = -quat

        end_pose = torch.tensor(
            np.concatenate([trans, quat], axis=0),
            dtype=start_pose.dtype,
            device=start_pose.device,
        )[None]

        rgbs = tvf.resize(observation["front_rgb"][0], (64, 64))
        pcds = tvf.resize(observation["front_point_cloud"][0], (64, 64))
        proprios = observation["proprios"][0]
        cond = {0: start_pose, -1: end_pose}
        rank = torch.zeros([1, 10], dtype=torch.float32, device="cuda")
        rank[0, -1] = 1

        return cond, dict(
            pcds=pcds.permute(0, 2, 3, 1),
            rgbs=rgbs.permute(0, 2, 3, 1),
            proprios=proprios,
            start=start_pose,
            end=end_pose,
            rank=rank,
            gripper_poses=torch.stack([start_pose, end_pose], dim=1),
            joint_positions=torch.stack([joints, joints], dim=1),
            robot=self._robot,
        )

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for qa in self._qattention_agents:
            summaries.extend(qa.update_summaries())
        return summaries

    def act_summaries(self) -> List[Summary]:
        s = []
        for qa in self._qattention_agents:
            s.extend(qa.act_summaries())
        return s

    def load_weights(self, savedir: str, backbone: str = "unet"):
        for qa in self._qattention_agents:
            qa.load_weights(savedir)

        load_model_path = self._diff_model_path
        try:
            state_dict = torch.load(load_model_path)
            del_keys = [k for k in state_dict if "loss_fn.weights" in k]
            for k in del_keys:
                del state_dict[k]

            self._diffusion_agent.load_state_dict(state_dict, strict=False)
            print("diffusion model loaded")
        except Exception as e:
            print("diffusion model load failed")
            print(str(e))

    def save_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.save_weights(savedir)
