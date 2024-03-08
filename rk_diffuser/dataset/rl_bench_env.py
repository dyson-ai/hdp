import logging
from typing import List, Type

import numpy as np
from pyrep.const import RenderMode
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.objects import Dummy, VisionSensor
from rlbench import ActionMode, ObservationConfig
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from yarr.agents.agent import VideoSummary
from yarr.envs.rlbench_env import RLBenchEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition

RECORD_EVERY = 20


class CustomRLBenchEnv(RLBenchEnv):
    def __init__(
        self,
        task_class: Type[Task],
        observation_config: ObservationConfig,
        action_mode: ActionMode,
        episode_length: int,
        dataset_root: str = "",
        channels_last: bool = False,
        reward_scale=100.0,
        headless: bool = True,
        time_in_state: bool = False,
    ):
        super(CustomRLBenchEnv, self).__init__(
            task_class,
            observation_config,
            action_mode,
            dataset_root,
            channels_last,
            headless=headless,
        )
        self._reward_scale = reward_scale
        self._episode_index = 0
        self._record_current_episode = False
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._episode_length = episode_length
        self._time_in_state = time_in_state
        self._i = 0

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomRLBenchEnv, self).observation_elements
        for oe in obs_elems:
            if oe.name == "low_dim_state":
                oe.shape = (
                    oe.shape[0] - 7 * 2 + int(self._time_in_state),
                )  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
        obs_elems.append(ObservationElement("gripper_pose", (7,), np.float32))
        return obs_elems

    def extract_obs(self, obs: Observation, t=None, prev_action=None):
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        joint_pos = obs.joint_positions
        # obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.wrist_camera_matrix = None
        obs.joint_positions = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0.0, 0.04
            )

        obs_dict = super(CustomRLBenchEnv, self).extract_obs(obs)

        if self._time_in_state:
            time = (
                1.0 - ((self._i if t is None else t) / float(self._episode_length - 1))
            ) * 2.0 - 1.0
            obs_dict["low_dim_state"] = np.concatenate(
                [obs_dict["low_dim_state"], [time]]
            ).astype(np.float32)

        obs.gripper_matrix = grip_mat
        # obs.gripper_pose = grip_pose
        obs.joint_positions = joint_pos

        obs_dict["gripper_pose"] = grip_pose
        return obs_dict

    def launch(self):
        super(CustomRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        cam_placeholder = Dummy("cam_cinematic_placeholder")
        cam_base = Dummy("cam_cinematic_base")
        cam_base.rotate([0, 0, np.pi * 0.75])
        self._record_cam = VisionSensor.create([320, 180])
        self._record_cam.set_explicit_handling(True)
        self._record_cam.set_pose(cam_placeholder.get_pose())
        self._record_cam.set_render_mode(RenderMode.OPENGL)

    def reset(self) -> dict:
        self._i = 0
        self._previous_obs_dict = super(CustomRLBenchEnv, self).reset()
        self._record_current_episode = self._episode_index % RECORD_EVERY == 0
        self._episode_index += 1
        self._recorded_images.clear()
        return self._previous_obs_dict

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _my_callback(self):
        self._record_cam.handle_explicitly()
        cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(cap)

    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10,) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

    def step(self, action: np.ndarray, record: bool = False) -> Transition:
        success = False
        obs = self._previous_obs_dict  # in case action fails.

        ik_error = False
        try:
            obs, reward, terminal = self._task.step(action)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0
            ik_error = True

        summaries = []
        self._i += 1
        if (
            (terminal or self._i == self._episode_length)
            and (self._record_current_episode)
            or record
        ):
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            summaries.append(
                VideoSummary(
                    "episode_rollout_" + ("success" if success else "fail"), vid, fps=30
                )
            )
            summaries.append(ik_error)
        return Transition(obs, reward, terminal, summaries=summaries)

    def reset_to_demo(self, demo):
        self._task.reset_to_demo(demo)
        self._i = 0
        self._episode_index += 1
        self._recorded_images.clear()

    def eval_planner(self, amount, callable_each_step):
        ctr_loop = self._task._robot.arm.joints[0].is_control_loop_enabled()
        self._task._robot.arm.set_control_loop_enabled(True)
        demos = self._get_live_demos(amount, callable_each_step)
        self._task._robot.arm.set_control_loop_enabled(ctr_loop)

        return len(demos) / amount

    def _get_live_demos(
        self,
        amount: int,
        callable_each_step,
    ):
        demos = []
        for i in range(amount):
            random_seed = np.random.get_state()
            self.reset()
            try:
                demo = self._task._scene.get_demo(callable_each_step=callable_each_step)
                demo.random_seed = random_seed
                demos.append(demo)
            except Exception as e:
                logging.info("Bad demo. " + str(e))
        return demos
