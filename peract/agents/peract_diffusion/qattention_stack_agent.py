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
        use_demo: bool = False,
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
        self._demo = None
        self._keyframes = []
        self._use_demo = use_demo
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

    def _set_demo(self, observation):
        keys = [
            "traj_joint_vels",
            "traj_joint_positions",
            "traj_gripper_opens",
            "traj_gripper_poses",
        ]
        self._demo = {k: observation[k][0, 0] for k in keys}

    def _get_keyframes(self):
        keypoints = []
        prev_gripper_open = self._demo["traj_gripper_opens"][0]
        stopped_buffer = 0
        demo_len = len(self._demo["traj_gripper_opens"])

        def is_stopped(i, stopped_buffer, delta=0.01):
            next_is_not_final = i == (demo_len - 2)
            gripper_opens = self._demo["traj_gripper_opens"]
            joint_vels = self._demo["traj_joint_vels"]
            gripper_state_no_change = i < (demo_len - 2) and (
                gripper_opens[i] == gripper_opens[i + 1]
                and gripper_opens[i] == gripper_opens[i - 1]
                and gripper_opens[i - 2] == gripper_opens[i - 1]
            )
            small_delta = torch.allclose(
                joint_vels[i],
                torch.tensor(0.0, dtype=joint_vels.dtype, device=joint_vels.device),
                atol=delta,
            )
            stopped = (
                stopped_buffer <= 0
                and small_delta
                and (not next_is_not_final)
                and gripper_state_no_change
            )
            return stopped

        gripper_opens = self._demo["traj_gripper_opens"]
        for i in range(demo_len):
            stopped = is_stopped(i, stopped_buffer, 0.01)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            last = i == (demo_len - 1)

            if i != 0 and (gripper_opens[i] != prev_gripper_open or last or stopped):
                keypoints.append(i)
            prev_gripper_open = gripper_opens[i]

        if len(keypoints) > 1 and keypoints[-1] - 1 == keypoints[-2]:
            keypoints.pop(-2)
        logging.debug("Found %d keypoints." % len(keypoints), keypoints)

        return [0] + keypoints

    def act(self, step: int, observation: dict, deterministic=False) -> ActResult:
        if "traj_joint_positions" in observation:
            try:
                self._set_demo(observation)
                self._keyframes = self._get_keyframes()
            except:
                pass

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
        continuous_action = np.concatenate(
            [
                trans,
                quat,
                rgai[-1:],
                [ignore_collisions],
            ]
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
        self._plot(
            traj.cpu().detach().numpy(),
            _continuous_data_dict["pcds"].cpu().detach().numpy(),
            _continuous_data_dict["rgbs"].cpu().detach().numpy(),
            gt_trajs=gt_traj.cpu().detach().numpy(),
        )
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
        path = f"/home/{getpass.getuser()}/repos/peract/plots"
        os.makedirs(path, exist_ok=True)
        existings = os.listdir(path)
        fig.write_html(os.path.join(path, f"vis_{len(existings)}.html"))

    def _prepare_diffusion_data_dict(
        self, observation: dict, trans: np.array, quat: np.array
    ) -> dict:
        if self._use_demo and len(self._keyframes) > 1:
            idx = self._keyframes.pop(0)
            joints = self._demo["traj_joint_positions"]
            gripper_poses = self._robot.forward_kinematics_batch(joints)

            start_pose = gripper_poses[idx : idx + 1]
            joints = joints[idx : idx + 1]
            end_pose = gripper_poses[self._keyframes[0] : self._keyframes[0] + 1]

        else:
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

            if backbone == "unet":
                model_keys = self._diffusion_agent.state_dict()
                for k in del_keys:
                    del model_keys[k]
                mis_matched_keys = [k for k in model_keys if k not in state_dict]

                new_dict = {}
                if len(mis_matched_keys) > 0:
                    print("trying to load legacy models")

                    for k in model_keys:
                        if k not in mis_matched_keys:
                            new_dict[k] = state_dict[k]
                        else:
                            legacy_key = k.replace("_backbone.", "")
                            new_dict[k] = state_dict[legacy_key]

                    state_dict = new_dict

            self._diffusion_agent.load_state_dict(state_dict, strict=False)
            print("diffusion model loaded")
        except Exception as e:
            print("diffusion model load failed")
            print(str(e))

    def save_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.save_weights(savedir)
