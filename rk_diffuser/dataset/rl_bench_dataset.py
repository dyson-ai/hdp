import os
import pickle
from collections import namedtuple
from multiprocessing import Manager, Process
from typing import Any, List

import numpy as np
import tqdm
from pyrep.const import RenderMode
from pyrep.robots.configuration_paths.arm_configuration_path import ArmConfigurationPath
from rlbench import CameraConfig, ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    ArmActionMode,
    EndEffectorPoseViaIK,
    EndEffectorPoseViaPlanning,
    assert_action_shape,
)
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import task as rlbench_tasks
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.scene import Scene
from rlbench.backend.utils import task_file_to_task_class
from torch.utils.data import Dataset

import rk_diffuser.utils as utils
from rk_diffuser.dataset.rl_bench_env import CustomRLBenchEnv

Batch = namedtuple("Batch", "trajectories proprios pcds conditions")
VisualObs = namedtuple("VisualObs", "rgbs pcds")

JOINT_THRESH = 0.05


class NormEEPosePlanning(EndEffectorPoseViaPlanning):
    def action(self, scene: Scene, action: np.ndarray):
        xyz = action[:3]
        quat = action[3:]
        quat = quat / np.linalg.norm(quat)

        action = np.concatenate([xyz, quat])

        super().action(scene, action)


class TrajectoryActionMode(EndEffectorPoseViaIK):
    """A sequence of end-effector poses representing a trajectory."""

    def __init__(
        self,
        points: int,
        absolute_mode: bool = True,
        frame: str = "world",
        collision_checking: bool = False,
    ):
        super().__init__(absolute_mode, frame, collision_checking)
        self._points = points

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7 * self._points,))
        if np.all(action == 0):
            raise InvalidActionError("No valid trajectory given.")

        action = action.reshape(self._points, 7)
        for a in action:
            xyz = a[:3]
            quat = a[3:]
            quat = quat / np.linalg.norm(quat)
            super().action(scene, np.concatenate([xyz, quat]))

    def action_shape(self, scene: Scene) -> tuple:
        return (7 * self._points,)


class JointsTrajectoryActionMode(ArmActionMode):
    """A sequence of joint configurations representing a trajectory."""

    def __init__(self, points: int):
        self._points = points

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7 * self._points,))
        if np.all(action == 0):
            raise InvalidActionError("No valid trajectory given.")

        action = self._pre_proc_traj(action)
        path = ArmConfigurationPath(scene.robot.arm, action)
        done = False
        while not done:
            done = path.step()
            scene.step()
            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success:
                break

    def _pre_proc_traj(self, action):
        action = action.reshape(-1, 7)
        new_actions = [action[0]]

        for idx in range(1, len(action) - 2):
            diff = new_actions[-1] - action[idx]
            if np.abs(diff).max() > JOINT_THRESH:
                new_actions.append(action[idx])

        new_actions.append(action[-1])
        return np.stack(new_actions, axis=0).reshape(-1)

    def action_shape(self, scene: Scene) -> tuple:
        return (7 * self._points,)


def _create_obs_config(camera_names: List[str], camera_resolution: List[int]):
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=False,
        depth=False,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL,
    )

    cam_obs = []
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams
        cam_obs.append("%s_rgb" % n)
        cam_obs.append("%s_pointcloud" % n)

    # Some of these obs are only used for keypoint detection.
    obs_config = ObservationConfig(
        front_camera=kwargs.get("front", unused_cams),
        left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
        right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
        wrist_camera=kwargs.get("wrist", unused_cams),
        overhead_camera=kwargs.get("overhead", unused_cams),
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    return obs_config


def _get_action_mode(num_points, use_traj=False):
    if use_traj:
        arm_action_mode = JointsTrajectoryActionMode(num_points)
    else:
        arm_action_mode = TrajectoryActionMode(num_points)
    gripper_action_mode = Discrete()
    return MoveArmThenGripper(arm_action_mode, gripper_action_mode)


class RLBenchDataset(Dataset):
    def __init__(
        self,
        tasks: List,
        tasks_ratio: List,
        camera_names: List,
        num_episodes: int,
        data_raw_path: str = "",
        traj_len: int = 100,
        output_img_size: int = 64,
        observation_dim: int = 7,
        frame_skips: int = 1,
        rank_bins: int = 10,
        robot: Any = None,
        diffusion_var: str = "gripper_poses",
        training: bool = True,
        demo_aug_ratio: float = 0.0,
        demo_aug_min_len: int = 20,
        use_cached: bool = True,
        ds_img_size: int = 128,
    ) -> None:
        super().__init__()

        self._tasks = tasks
        self._tasks_ratio = tasks_ratio
        self._camera_names = camera_names
        self._num_episodes = num_episodes
        self._data_raw_path = data_raw_path
        self._traj_len = traj_len
        self._output_img_size = output_img_size
        self._observation_dim = observation_dim
        self._frame_skips = frame_skips
        self._rank_bins = rank_bins
        self._robot = robot
        self._diffusion_var = diffusion_var
        self._training = training
        self._demo_aug_ratio = demo_aug_ratio
        self._demo_aug_min_len = demo_aug_min_len
        self._use_cached = use_cached

        # Note that this image size is the default size of the dataset
        # which will be used to load in the dataset.
        # We will need to resize it to the desired size for the diffusion
        # policy experiment.
        self._ds_img_size = ds_img_size

        os.makedirs(data_raw_path, exist_ok=True)
        if len(tasks_ratio) != len(tasks):
            tasks_ratio = [1.0] * len(tasks)

        datasets = {
            "pcds": [],
            "rgbs": [],
            "gripper_poses": [],
            "proprios": [],
            "joint_positions": [],
        }

        self._demos = {}

        manager = Manager()
        demo_dict_mp = manager.dict()
        ds_list = manager.list()
        procs = []

        for task in tasks:
            procs.append(Process(target=self._get_demos, args=(task, demo_dict_mp)))

        [p.start() for p in procs]
        [p.join() for p in procs]

        procs = []
        for task in tasks:
            procs.append(
                Process(
                    target=self._load_demos_to_dataset,
                    args=(
                        demo_dict_mp[task],
                        num_episodes,
                        ds_list,
                    ),
                )
            )

        [p.start() for p in procs]
        [p.join() for p in procs]

        for ds in ds_list:
            for k, v in ds.items():
                datasets[k].extend(v)

        self._dataset = datasets

        self._total_length = len(self._dataset["gripper_poses"])
        rand_order = np.random.permutation(self._total_length)

        permuted_ds = {}
        for k, v in self._dataset.items():
            permuted_ds[k] = [v[idx] for idx in rand_order]

        self._dataset = permuted_ds

    def _get_env(self, task, headless=True, use_traj=False):
        observation_config = _create_obs_config(
            self._camera_names,
            [self._ds_img_size, self._ds_img_size],
        )
        action_mode = _get_action_mode(self._traj_len, use_traj=use_traj)

        task_files = [
            t.replace(".py", "")
            for t in os.listdir(rlbench_tasks.TASKS_PATH)
            if t != "__init__.py" and t.endswith(".py")
        ]
        if task not in task_files:
            raise ValueError("Task %s not recognised!." % task)
        task_class = task_file_to_task_class(task)

        env = CustomRLBenchEnv(
            task_class=task_class,
            observation_config=observation_config,
            action_mode=action_mode,
            episode_length=200,
            dataset_root=self._data_raw_path,
            headless=headless,
            time_in_state=True,
        )

        return env

    def _get_demos(self, task, ret_dict):
        def _get_demos_fn():
            env = self._get_env(task)
            env.launch()
            print(f"Getting demos for task {task}")
            demos = env._task.get_demos(amount=self._num_episodes, live_demos=False)
            env.shutdown()
            return demos

        demos = None
        cache_path = os.path.join(self._data_raw_path, task, "cache.pkl")
        if self._use_cached and os.path.isfile(cache_path):
            with open(cache_path, "rb") as fin:
                demos = pickle.load(fin)

        if demos is None:
            demos = _get_demos_fn()
            with open(
                os.path.join(self._data_raw_path, task, "cache.pkl"), "wb"
            ) as fout:
                pickle.dump(demos, fout, pickle.HIGHEST_PROTOCOL)

        ret_dict[task] = demos
        train = "training" if self._training else "evaluation"
        print(f"Finished retrieving {self._num_episodes} {train} demos for task {task}")

    def _load_demos_to_dataset(self, demos, num_of_episodes, ds_list):
        dataset = {
            "pcds": [],
            "rgbs": [],
            "gripper_poses": [],
            "proprios": [],
            "joint_positions": [],
        }
        import cv2

        replace = num_of_episodes > len(demos)
        indices = np.random.choice(len(demos), num_of_episodes, replace=replace)
        print("preprocessing dataset")
        for n in tqdm.tqdm(indices):
            demo = demos[n]

            key_frames = utils.keypoint_discovery(demo=demo, stopping_delta=0.01)
            key_frames = [0] + key_frames

            for i in range(len(key_frames) - 1):
                start, end = key_frames[i : i + 2]
                observations = demo._observations[start:end]

                pcds = []
                poses = []
                rgbs = []
                proprio = []
                joints = []

                for obs in observations:
                    pcds.append(
                        np.concatenate(
                            [
                                cv2.resize(
                                    getattr(obs, f"{c}_point_cloud"),
                                    (64, 64),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                                for c in self._camera_names
                            ],
                            axis=0,
                        )
                    )

                    rgbs.append(
                        np.concatenate(
                            [
                                cv2.resize(
                                    getattr(obs, f"{c}_rgb"),
                                    (64, 64),
                                ).astype(int)
                                for c in self._camera_names
                            ],
                            axis=0,
                        )
                    )

                    poses.append(obs.gripper_pose)
                    proprio.append(obs.get_low_dim_data())

                    if "executed_demo_joint_position_action" in obs.misc:
                        joint_action = obs.misc["executed_demo_joint_position_action"]
                    else:
                        joint_action = obs.joint_positions

                    joints.append(joint_action)

                pcds = np.stack(pcds, axis=0).astype(np.float32)
                poses = np.stack(poses, axis=0).astype(np.float32)
                poses = utils.proc_quaternion(poses)
                proprio = np.stack(proprio, axis=0).astype(np.float32)
                rgbs = np.stack(rgbs, axis=0).astype(np.float32)
                joints = np.stack(joints, axis=0).astype(np.float32)

                dataset["pcds"].append(pcds)
                dataset["gripper_poses"].append(poses)
                dataset["rgbs"].append(rgbs)
                dataset["proprios"].append(proprio)
                dataset["joint_positions"].append(joints)

        ds_list.append(dataset)

    def __len__(self):
        return len(self._dataset["gripper_poses"])

    def min_traj_len(self):
        traj_lens = [len(traj) for traj in self._dataset["pcds"]]
        return min(traj_lens)

    def get_conditions(self, observations):
        ret_dict = {
            0: observations[0],
        }

        if self._diffusion_var == "gripper_poses":
            ret_dict[-1] = observations[-1]

        return ret_dict

    def _calc_rank(self, poses):
        traj = poses[:, :3]
        euc_dist = np.linalg.norm(traj[-1] - traj[0])
        dist = traj[1:] - traj[:-1]
        dist = np.linalg.norm(dist, axis=1).sum()

        rank = euc_dist / (dist + 1e-5)
        rank_id = rank / (1.0 / self._rank_bins)
        return np.eye(self._rank_bins, dtype=np.float32)[
            int(np.clip(rank_id, 0, self._rank_bins - 1))
        ]

    def __getitem__(self, index) -> Any:
        gripper_poses = self._dataset["gripper_poses"][index]
        joints = self._dataset["joint_positions"][index]
        traj_len = self._traj_len * self._frame_skips
        cur_traj_len = len(gripper_poses)

        start, end = 0, cur_traj_len - 1

        if self._demo_aug_ratio > 0:
            start_left = np.random.rand() < 0.5
            demo_aug_len = min(cur_traj_len - 1, self._demo_aug_min_len)
            if start_left:
                start = np.random.randint(0, demo_aug_len)
                end = np.random.randint(
                    max(cur_traj_len - demo_aug_len, start), cur_traj_len
                )
            else:
                start = np.random.randint(cur_traj_len - demo_aug_len, cur_traj_len - 1)

            gripper_poses = gripper_poses[start:end]
            joints = joints[start:end]

        pcds = self._dataset["pcds"][index][start]
        rgbs = self._dataset["rgbs"][index][start]
        proprios = self._dataset["proprios"][index][start]
        if len(gripper_poses) < traj_len:
            repeats = traj_len // len(gripper_poses) + 1
            indices = np.arange(len(gripper_poses)).repeat(repeats=repeats, axis=0)
            gripper_poses = gripper_poses[indices]
            joints = joints[indices]

        indices = np.random.choice(
            len(gripper_poses) - 1, self._traj_len - 2, replace=False
        )
        indices = np.sort(indices) + 1
        indices = np.concatenate(
            [np.array([0]), indices, np.array([len(gripper_poses) - 1])]
        )

        gripper_poses = gripper_poses[indices]
        joints = joints[indices]

        rank = self._calc_rank(gripper_poses)

        if self._diffusion_var == "gripper_poses":
            x = gripper_poses
        else:
            x = joints

        batch = dict(
            cond=self.get_conditions(x),
            proprios=proprios,
            pcds=pcds,
            rank=rank,
            joint_positions=joints,
            gripper_poses=gripper_poses,
            start=gripper_poses[0],
            end=gripper_poses[-1],
            rgbs=rgbs,
        )

        if self._training:
            batch["x"] = x

        return batch
