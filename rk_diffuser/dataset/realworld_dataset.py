import logging
import os
from multiprocessing import Manager, Process
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
import torch
from natsort import natsorted
from safetensors.numpy import load_file
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from rk_diffuser import utils
import cv2
from multiprocessing import Process, Manager
import pickle


class RealWorldDataset:
    """Class to load real world dataset"""

    def __init__(
        self,
        tasks: List,
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
        robot_offset: List | None = None,
        camera_extrinsics: List | None = None,
        save_processed_data: bool = False,
        load_processed_data: bool = False,
    ) -> None:
        self._tasks = tasks
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
        self._demos = {}
        if robot_offset is None:
            self._robot_to_world = np.eye(4)
        else:
            self._robot_to_world = self._pose_to_matrix(robot_offset)
        if camera_extrinsics is None:
            self._camera_to_robot = [np.eye(4) for _ in range(len(self._camera_names))]
        else:
            self._camera_to_robot = {
                cam: self._pose_to_matrix(extrinsics)
                for cam, extrinsics in camera_extrinsics.items()
            }
        self._camera_to_world = {
            cam: np.matmul(self._robot_to_world, cam_to_robot)
            for cam, cam_to_robot in self._camera_to_robot.items()
        }
        self._demo_aug_ratio = demo_aug_ratio
        self._demo_aug_min_len = demo_aug_min_len
        self._camera_intrinsics = {}
        dataset = {
            "pcds": [],
            "rgbs": [],
            "gripper_poses": [],
            "proprios": [],
            "joint_positions": [],
        }

        # check if train and eval.pkl exist then load data from those files
        # else load raw data
        if load_processed_data:
            if self._training:
                assert os.path.exists(
                    data_raw_path + "/train.pkl"
                ), "Train .pkl not found"
                with open(data_raw_path + "/train.pkl", "rb") as f:
                    self._demos = pickle.load(f)
            else:
                assert os.path.exists(
                    data_raw_path + "/eval.pkl"
                ), "Eval .pkl not found"
                with open(data_raw_path + "/eval.pkl", "rb") as f:
                    self._demos = pickle.load(f)

            # check if task if present, trim as per num episodes
            for task in self._tasks:
                assert task in self._demos.keys(), "Task not present in .pkl file"
                self._demos[task] = self._demos[task]

            # delete unwanted task
            for task in list(self._demos.keys()):
                if task not in self._tasks:
                    del self._demos[task]

            for task, task_dataset in self._demos.items():
                for k, v in task_dataset.items():
                    dataset[k].extend(v)
        else:
            manager = Manager()
            demo_dict = manager.dict()
            procs = []

            for task in tasks:
                procs.append(
                    Process(
                        target=self._load_dataset, args=(task, data_raw_path, demo_dict)
                    )
                )

            [p.start() for p in procs]
            [p.join() for p in procs]

            for task, task_dataset in demo_dict.items():
                self._demos[task] = task_dataset
                for k, v in task_dataset.items():
                    dataset[k].extend(v)

        # save dataset
        if save_processed_data:
            if self._training:
                # save self._demos in picke file
                with open(data_raw_path + "/train.pkl", "wb") as f:
                    pickle.dump(self._demos, f)
            else:
                # save self._demos in picke file
                with open(data_raw_path + "/eval.pkl", "wb") as f:
                    pickle.dump(self._demos, f)
        self._dataset = dataset
        self._total_length = len(self._dataset["gripper_poses"])
        rand_order = np.random.permutation(self._total_length)

        permuted_ds = {}
        for k, v in self._dataset.items():
            permuted_ds[k] = [v[idx] for idx in rand_order]

        self._dataset = permuted_ds

    def _load_dataset(self, task, data_raw_path, demo_dict):
        task_dataset = {
            "pcds": [],
            "rgbs": [],
            "gripper_poses": [],
            "proprios": [],
            "joint_positions": [],
        }
        if self._training:
            data_root = Path(os.path.join(data_raw_path, task, "train"))
            print("Training set")
        else:
            data_root = Path(os.path.join(data_raw_path, task, "eval"))
            print("Eval set")

        if not os.path.exists(data_root):
            raise Exception(f"{data_root} does not exist")
        for demo_path in tqdm(
            natsorted(data_root.glob("*.safetensors"))[: self._num_episodes],
            desc=f"{task}: ",
        ):
            demo = load_file(demo_path)
            # save camera intrinsics from the demo
            if not self._camera_intrinsics:
                for camera in self._camera_names:
                    self._camera_intrinsics[camera] = demo[
                        f"misc_{camera}_depth_intrinsics"
                    ]
            demo = self._downsample_images(demo)  # reduce image size
            obs = self._extract_obs_from_demo(demo)
            for k, v in obs.items():
                task_dataset[k].extend(v)

        demo_dict[task] = task_dataset

    def _downsample_images(self, demo):
        for cam in self._camera_names:
            rgb_images = demo[f"obs_{cam}_rgb"]
            depth_images = demo[f"obs_{cam}_depth"]
            resized_rgb_images = []
            resized_pcds = []
            for i in range(len(rgb_images)):
                resized_rgb_images.append(
                    cv2.resize(
                        rgb_images[i], (self._output_img_size, self._output_img_size)
                    )
                )
                pcds = self._pointcloud_from_depth_and_camera_params(
                    depth_images[i],
                    extrinsics=np.eye(4),
                    intrinsics=self._camera_intrinsics[cam],
                    offset=self._camera_to_world[cam],
                )
                resized_pcds.append(pcds)
            demo[f"obs_{cam}_rgb"] = np.array(resized_rgb_images, dtype=np.uint8)
            demo[f"obs_{cam}_depth"] = np.array(resized_pcds, dtype=np.float16)
        return demo

    def _extract_obs_from_demo(self, demo: dict):
        """
        Generates a dataset from a list of demos.

        Args:
            demos (list): A list of demos.

        Returns:
            dict: A dictionary containing the generated dataset with the following keys:
                - "pcds" (list): A list of point clouds.
                - "rgbs" (list): A list of RGB images.
                - "gripper_poses" (list): A list of gripper poses.
                - "proprios" (list): A list of proprioceptive data.
                - "joint_positions" (list): A list of joint positions.
        """
        dataset = {
            "pcds": [],
            "rgbs": [],
            "gripper_poses": [],
            "proprios": [],
            "joint_positions": [],
        }

        key_frames = self._keypoint_discovery(
            obs={
                "obs_vecEEPose": demo["obs_vecEEPose"],
                "obs_vecJointPositions": demo["obs_vecJointPositions"],
            },
            stopping_delta=0.01,
        )
        # print(f"Discovered keyframes: {key_frames}")
        for i in range(len(key_frames) - 1):
            start, end = key_frames[i : i + 2]
            # observations = demo[start:end]
            observations = {
                key: value[start:end] for key, value in demo.items() if "obs" in key
            }

            pcds = []
            poses = []
            rgbs = []
            proprio = []
            joints = []

            for i in range(end - start):
                # pcds.append(np.concatenate([
                #     self._pointcloud_from_depth_and_camera_params(observations[f"obs_{camera}_depth"][i], extrinsics=self.camera_to_robot, intrinsics=self._camera_intrinsics[camera]) for camera in self._camera_names], axis=0))
                pcds.append(
                    np.concatenate(
                        [
                            observations[f"obs_{camera}_depth"][i]
                            for camera in self._camera_names
                        ],
                        axis=0,
                    )
                )
                rgbs.append(
                    np.concatenate(
                        [
                            observations[f"obs_{camera}_rgb"][i]
                            for camera in self._camera_names
                        ],
                        axis=0,
                    )
                )
                poses.append(
                    self._matrix_to_pose(
                        observations["obs_vecEEPose"][i], offset=self._robot_to_world
                    )
                )
                proprio.append(
                    self._get_low_dim_data(
                        {key: value[i] for key, value in observations.items()}
                    )
                )
                joints.append(observations["obs_vecJointPositions"][i][:-1])

            pcds = np.stack(pcds, axis=0).astype(np.float32)
            poses = np.stack(poses, axis=0).astype(np.float32)
            start, end = poses[0, :3], poses[-1, :3]
            if np.linalg.norm(end - start) <= 0.01:
                continue
            poses = utils.proc_quaternion(poses)
            proprio = np.stack(proprio, axis=0).astype(np.float32)
            rgbs = np.stack(rgbs, axis=0).astype(np.float32)
            joints = np.stack(joints, axis=0).astype(np.float32)
            dataset["pcds"].append(pcds)
            dataset["gripper_poses"].append(poses)
            dataset["rgbs"].append(rgbs)
            dataset["proprios"].append(proprio)
            dataset["joint_positions"].append(joints)

        return dataset

    def _keypoint_discovery(self, obs: dict, stopping_delta=0.1) -> List[int]:
        """
        Discover keypoints in a given observation.

        Args:
            obs (dict): The observation dictionary containing the joint positions and end effector poses.
            stopping_delta (float, optional): The stopping threshold for detecting if the robot has stopped. Defaults to 0.1.

        Returns:
            List[int]: A list of indices representing the keypoints in the observation.

        """
        demo_joint_positions = obs["obs_vecJointPositions"]
        demo_ee_poses = obs["obs_vecEEPose"]
        episode_keypoints = []
        prev_gripper_open = self._is_gripper_open(demo_joint_positions[0])
        stopped_buffer = 0
        for i, joint_pose in enumerate(demo_joint_positions):
            stopped = self._is_stopped(
                demo_joint_positions, i, stopped_buffer, stopping_delta
            )
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo_joint_positions) - 1)
            if i != 0 and (
                self._is_gripper_open(joint_pose) != prev_gripper_open
                or last
                or stopped
            ):
                episode_keypoints.append(i)
            prev_gripper_open = self._is_gripper_open(joint_pose)
        if (
            len(episode_keypoints) > 1
            and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
        ):
            episode_keypoints.pop(-2)
        # remove the last frame if the robot stopped and only gripper is toggled
        if np.linalg.norm(
            demo_ee_poses[episode_keypoints[-2]] - demo_ee_poses[episode_keypoints[-1]]
        ) < 0.1 and self._is_gripper_open(
            demo_joint_positions[episode_keypoints[-2]]
        ) == self._is_gripper_open(
            demo_joint_positions[episode_keypoints[-1]]
        ):
            episode_keypoints = episode_keypoints[:-1]
        episode_keypoints = [
            0
        ] + episode_keypoints  # adding the first keyframe as keypoint
        logging.debug("Found %d keypoints." % len(episode_keypoints), episode_keypoints)
        return episode_keypoints

    def _is_stopped(self, joint_poses, i, stopped_buffer, delta=0.1):
        """
        Check if the robot arm is stopped at a particular joint position.

        Parameters:
            joint_poses (list): A list of joint positions.
            i (int): The index of the current joint position.
            stopped_buffer (int): The buffer for determining if the robot arm is stopped.
            delta (float, optional): The tolerance for determining if the joint position is close to zero.

        Returns:
            bool: True if the robot arm is stopped, False otherwise.
        """
        next_is_not_final = i == (len(joint_poses) - 2)
        gripper_state_no_change = i < (len(joint_poses) - 2) and (
            self._is_gripper_open(joint_poses[i])
            == self._is_gripper_open(joint_poses[i + 1])
            and self._is_gripper_open(joint_poses[i])
            == self._is_gripper_open(joint_poses[i - 1])
            and self._is_gripper_open(joint_poses[i - 2])
            == self._is_gripper_open(joint_poses[i - 1])
        )
        small_delta = np.allclose(joint_poses[i][:-1], 0, atol=delta)
        stopped = (
            stopped_buffer <= 0
            and small_delta
            and (not next_is_not_final)
            and gripper_state_no_change
        )
        return stopped

    def _is_gripper_open(self, pose):
        """
        Check if the gripper is open based on the joint position observation.

        Parameters:
            obs (list): The observation dictionary containing the joint positions.

        Returns:
            float: 1.0 if the gripper is open, 0 otherwise.
        """
        return 1.0 if pose[-1] > 0.9 else 0

    def _matrix_to_pose(self, matrix, offset=np.eye(4)):
        """Convert transformation matrix to pose

        Args:
            matrix (np.ndarray): gripper pose matrix
            offset (np.ndarray, optional): Gripper pose offset as [position,quaternion(xyzw)]. Defaults to np.ndarray | None=None.

        Returns:
           pose (np.ndarray) : pose as [position,quaternion]
        """
        tran_matrix = np.matmul(offset, matrix)

        rotation = Rotation.from_matrix(tran_matrix[:3, :3])
        position = tran_matrix[:3, 3]
        quat = rotation.as_quat()
        gripper_pose = np.concatenate([position, quat])
        return gripper_pose

    def _pose_to_matrix(self, pose):
        """Convert pose(position, quaternion) to matrix

        Args:
            pose (numpy.ndarray): pose as position,quaternion

        Returns:
            tran_mat (numpy.ndarray): transformation matrix
        """
        rot_mat = Rotation.from_quat(pose[3:]).as_matrix()
        tran_mat = np.eye(4)
        tran_mat[:3, :3] = rot_mat
        tran_mat[:3, 3] = pose[:3]
        return tran_mat

    def _get_low_dim_data(self, obs):
        low_dim_data = (
            []
            if self._is_gripper_open(obs["obs_vecJointPositions"]) is None
            else [[self._is_gripper_open(obs["obs_vecJointPositions"])]]
        )
        for data in [
            obs["obs_vecJointVelocities"][:-1],
            obs["obs_vecJointPositions"][:-1],
            self._matrix_to_pose(obs["obs_vecEEPose"], offset=self._robot_to_world),
            [obs["obs_vecJointPositions"][-1]] * 2,  # for two finger joints
        ]:
            if data is not None:
                low_dim_data = np.append(low_dim_data, data)
        return low_dim_data

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
        rank_id = rank / (1.0 / self._rank_bins + 1e-5)
        rank_id = np.clip(rank_id, 0, self._rank_bins - 1)
        return np.eye(self._rank_bins, dtype=np.float32)[int(rank_id)]

    def _filter_traj(self, joints_positions: np.ndarray) -> list:
        """
        Filters a numpy array of poses based on a distance threshold.

        Args:
            joint_positions (np.ndarray): The numpy array of poses, where each row represents a pose.

        Returns:
            list: The list of indices that satisfy the distance threshold.
        """
        indices = [0]
        assert (
            joints_positions.shape[-1] == 7
        ), "Joint positions array doesn't have a valid shape"
        for i, joints in enumerate(joints_positions):
            prev_joints = joints_positions[indices[-1]]
            if abs(joints - prev_joints).max() >= 0.05:
                indices.append(i)

        return indices

    def __getitem__(self, index) -> Any:
        gripper_poses = self._dataset["gripper_poses"][index]
        joints = self._dataset["joint_positions"][index]
        traj_len = self._traj_len * self._frame_skips

        valid_indices = self._filter_traj(joints)
        gripper_poses = gripper_poses[valid_indices]
        joints = joints[valid_indices]
        cur_traj_len = len(gripper_poses)
        start, end = 0, cur_traj_len
        if self._demo_aug_ratio > 0:
            start_left = np.random.rand() < 0.5
            demo_aug_len = min(cur_traj_len, self._demo_aug_min_len)
            if start_left:
                start = np.random.randint(0, demo_aug_len - 1)
                end = np.random.randint(
                    max(cur_traj_len - demo_aug_len, start + 2), cur_traj_len + 1
                )
            else:
                start = np.random.randint(cur_traj_len - demo_aug_len, cur_traj_len - 1)
            gripper_poses = gripper_poses[start:end]
            joints = joints[start:end]

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
        pcds = self._dataset["pcds"][index][start]
        rgbs = self._dataset["rgbs"][index][start]
        proprios = self._dataset["proprios"][index][start]

        rank = self._calc_rank(gripper_poses)

        if self._diffusion_var == "gripper_poses":
            x = gripper_poses
        else:
            x = joints

        batch = dict(
            cond=self.get_conditions(x),
            proprios=proprios,
            rgbs=rgbs,
            pcds=pcds,
            rank=rank,
            joint_positions=joints,
            gripper_poses=gripper_poses,
            start=gripper_poses[0],
            end=gripper_poses[-1],
        )

        if self._training:
            batch["x"] = x
        else:
            batch["rgbs"] = rgbs

        return batch

    def _prepare_input(self, task: str, use_demos: bool = False):
        diff_input = []  # input for diffusion
        gt_traj = []  # ground truth trajectories
        if use_demos:
            demos = self._demos[task]
            no_of_traj = len(demos["gripper_poses"])
            print(f"No. of trajectories found: {no_of_traj}")
            for i in range(no_of_traj):
                start_pose = torch.tensor(
                    demos["gripper_poses"][i][0], dtype=torch.float32, device="cuda"
                ).unsqueeze(0)
                end_pose = torch.tensor(
                    demos["gripper_poses"][i][-1], dtype=torch.float32, device="cuda"
                ).unsqueeze(0)
                start_joints = torch.tensor(
                    demos["joint_positions"][i][0], dtype=torch.float32, device="cuda"
                ).unsqueeze(0)
                end_joints = torch.tensor(
                    demos["joint_positions"][i][-1], dtype=torch.float32, device="cuda"
                ).unsqueeze(0)
                cond = {0: start_pose, -1: end_pose}
                pcds = torch.tensor(
                    demos["pcds"][i][0], dtype=torch.float32, device="cuda"
                ).unsqueeze(0)
                rgbs = torch.tensor(
                    demos["rgbs"][i][0], dtype=torch.float32, device="cuda"
                ).unsqueeze(0)
                rank = torch.zeros(
                    [1, self._rank_bins], dtype=torch.float32, device="cuda"
                )
                rank[0, -1] = 1
                proprios = torch.tensor(
                    demos["proprios"][i][0], dtype=torch.float32, device="cuda"
                ).unsqueeze(0)

                diff_input.append(
                    dict(
                        cond=cond,
                        pcds=pcds,
                        rgbs=rgbs,
                        proprios=proprios,
                        start=start_pose,
                        end=end_pose,
                        rank=rank,
                        gripper_poses=torch.stack([start_pose, end_pose], dim=1),
                        joint_positions=torch.stack([start_joints, end_joints], dim=1),
                        robot=self._robot,
                    )
                )

                gt_traj.append(
                    torch.tensor(
                        demos["gripper_poses"][i], dtype=torch.float32, device="cuda"
                    ).unsqueeze(0)
                )

            return diff_input, gt_traj

    def _create_uniform_pixel_coords_image(self, resolution: np.ndarray):
        """
        Create a uniform pixel coordinates image.

        Args:
            resolution (np.ndarray): A numpy array specifying the resolution of the image.

        Returns:
            uniform_pixel_coords (np.ndarray): A numpy array representing the uniform pixel coordinates image.
        """
        pixel_x_coords = np.reshape(
            np.tile(np.arange(resolution[1]), [resolution[0]]),
            (resolution[0], resolution[1], 1),
        ).astype(np.float32)
        pixel_y_coords = np.reshape(
            np.tile(np.arange(resolution[0]), [resolution[1]]),
            (resolution[1], resolution[0], 1),
        ).astype(np.float32)
        pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
        uniform_pixel_coords = np.concatenate(
            (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1
        )
        return uniform_pixel_coords

    def _transform(self, coords, trans):
        """
        Transforms the given coordinates using the specified transformation matrix.

        Args:
            coords (numpy.ndarray): The input coordinates to be transformed.
            trans (numpy.ndarray): The transformation matrix to be applied.

        Returns:
            numpy.ndarray: The transformed coordinates.

        Raises:
            None
        """
        h, w = coords.shape[:2]
        coords = np.reshape(coords, (h * w, -1))
        coords = np.transpose(coords, (1, 0))
        transformed_coords_vector = np.matmul(trans, coords)
        transformed_coords_vector = np.transpose(transformed_coords_vector, (1, 0))
        return np.reshape(transformed_coords_vector, (h, w, -1))

    def _pixel_to_world_coords(self, pixel_coords, cam_proj_mat_inv):
        """
        Convert pixel coordinates to world coordinates.

        Args:
            pixel_coords (numpy.ndarray): An array of shape (h, w) representing the pixel coordinates.
            cam_proj_mat_inv (numpy.ndarray): The inverse of the camera projection matrix.

        Returns:
            numpy.ndarray: An array of shape (h, w, 4) representing the world coordinates in homogeneous form.
        """
        h, w = pixel_coords.shape[:2]
        pixel_coords = np.concatenate([pixel_coords, np.ones((h, w, 1))], -1)
        world_coords = self._transform(pixel_coords, cam_proj_mat_inv)
        world_coords_homo = np.concatenate([world_coords, np.ones((h, w, 1))], axis=-1)
        return world_coords_homo

    def _pointcloud_from_depth_and_camera_params(
        self,
        depth_image: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        offset: np.ndarray = np.eye(4),
    ) -> np.ndarray:
        """Converts depth image (in mm) to point cloud in world frame.
        :return: A numpy array of size (width, height, 3)
        """
        upc = self._create_uniform_pixel_coords_image(depth_image.shape)
        pc = upc * np.expand_dims(depth_image, -1)
        C = np.expand_dims(extrinsics[:3, 3], 0).T
        R = extrinsics[:3, :3]
        R_inv = R.T  # inverse of rot matrix is transpose
        R_inv_C = np.matmul(R_inv, C)
        extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
        cam_proj_mat = np.matmul(intrinsics, extrinsics)
        cam_proj_mat_homo = np.concatenate([cam_proj_mat, [np.array([0, 0, 0, 1])]])
        cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
        world_coords_homo = np.expand_dims(
            self._pixel_to_world_coords(pc, cam_proj_mat_inv), 0
        )
        world_coords = world_coords_homo[..., :-1][0]
        # convert to meters
        world_coords /= 1000.0

        # resize the pcds
        world_coords = cv2.resize(
            world_coords,
            (self._output_img_size, self._output_img_size),
            interpolation=cv2.INTER_NEAREST,
        )
        world_coords = np.concatenate(
            [world_coords, np.ones((self._output_img_size, self._output_img_size, 1))],
            -1,
        )
        trans_world_coords = self._transform(world_coords, offset)

        return trans_world_coords[:, :, :3]
