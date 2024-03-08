"""Trainer for diffusion models."""

import copy
import datetime
import os
from multiprocessing import Manager, Process

import numpy as np
import plotly.graph_objects as go
import torch
import tqdm
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader

import wandb
from rk_diffuser.dataset.rl_bench_dataset import RLBenchDataset
from rk_diffuser.models.diffusion import GaussianDynDiffusion
from rk_diffuser.robot import DiffRobot
from rk_diffuser.utils import (
    batch_to_device,
    concatenate_tensors_dict_list,
    keypoint_discovery,
    proc_quaternion,
)

HOLD_TASKS = [
    "take_lid_off_saucepan",
    "pick_up_cup",
    "toilet_seat_up",
    "open_grill",
    "open_box",
    "open_oven",
]


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA:
    """
    empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self._beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self._update_average(old_weight, up_weight)

    def _update_average(self, old, new):
        if old is None:
            return new
        return old * self._beta + (1 - self._beta) * new


class Trainer(object):
    def __init__(
        self,
        cfgs: DictConfig,
        diffusion_model: GaussianDynDiffusion,
        dataset: RLBenchDataset,
        eval_dataset: RLBenchDataset,
        scene_bounds: list,
        ema_decay: float = 0.995,
        train_batch_size: int = 32,
        train_lr: float = 2e-5,
        gradient_accumulate_every: int = 2,
        step_start_ema: int = 2000,
        update_ema_every: int = 10,
        log_freq: int = 1000,
        save_freq: int = 10000,
        train_device: str = "cuda",
        log: bool = False,
        project_name: str = "HDP",
        online_eval: bool = False,
        headless: bool = True,
        rank_bins: int = 10,
        robot: DiffRobot = None,
        diffusion_var: str = "gripper_poses",
        online_eval_start: int = 0,
        action_mode: str = "joints",
    ):
        """
        Initializes an instance of the trainer.

        Parameters:
        - cfgs (DictConfig): The configurations for the trainer.
        - diffusion_model (GaussianDynDiffusion): The diffusion model.
        - dataset (RLBenchDataset): The training dataset.
        - eval_dataset (RLBenchDataset): The evaluation dataset.
        - scene_bounds (list): The bounds of the scene.
        - ema_decay (float, optional): The exponential moving average decay rate. Defaults to 0.995.
        - train_batch_size (int, optional): The batch size for training. Defaults to 32.
        - train_lr (float, optional): The learning rate for training. Defaults to 2e-5.
        - gradient_accumulate_every (int, optional): The number of gradient accumulation steps. Defaults to 2.
        - step_start_ema (int, optional): The step at which to start updating the exponential moving average. Defaults to 2000.
        - update_ema_every (int, optional): The frequency of updating the exponential moving average. Defaults to 10.
        - log_freq (int, optional): The frequency of logging. Defaults to 1000.
        - save_freq (int, optional): The frequency of saving checkpoints. Defaults to 10000.
        - train_device (str, optional): The device for training. Defaults to "cuda".
        - log (bool, optional): Whether to log. Defaults to False.
        - project_name (str, optional): The name of the project. Defaults to "HDP".
        - online_eval (bool, optional): Whether to perform online evaluation. Defaults to False.
        - headless (bool, optional): Whether to run in headless mode. Defaults to True.
        - rank_bins (int, optional): The number of rank bins. Defaults to 10.
        - robot (DiffRobot, optional): The differentiable robot. Defaults to None.
        - diffusion_var (str, optional): The type of diffusion model. Defaults to "gripper_poses".
        - online_eval_start (int, optional): The start time for online evaluation. Defaults to 0.
        - action_mode (str, optional): The action mode. Defaults to "joints".

        Returns:
        - None
        """
        super().__init__()
        self._cfgs = cfgs
        self._model = diffusion_model
        self._ema = EMA(ema_decay)
        self._ema_model = copy.deepcopy(self._model)
        self._update_ema_every = update_ema_every

        self._scene_bounds = scene_bounds

        self._step_start_ema = step_start_ema
        self._log_freq = log_freq
        self._save_freq = save_freq

        self._batch_size = train_batch_size
        self._gradient_accumulate_every = gradient_accumulate_every
        self._log = log
        self._project_name = project_name
        self._online_eval = online_eval
        self._headless = headless
        self._rank_bins = rank_bins
        self._online_eval_start = online_eval_start

        self._dataset = dataset
        self._eval_dataset = eval_dataset
        self._action_mode = action_mode
        self._diffusion_var = diffusion_var

        self._robot = robot
        self._dataloader = cycle(
            DataLoader(
                self._dataset,
                batch_size=train_batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=True,
            )
        )
        self._eval_dataloader = cycle(
            DataLoader(
                self._eval_dataset,
                batch_size=10,
                num_workers=0,
                pin_memory=True,
                shuffle=True,
            )
        )

        self._optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=train_lr)

        self._reset_parameters()
        self._step = 0

        self._device = train_device
        if self._log:
            self._set_logger()

        if self._online_eval:
            self.create_envs()

    def create_envs(self) -> None:
        """
        Generates the environment instances for each task and launches the processes
        that will run the environments.

        Returns:
            None
        """
        use_traj = self._diffusion_var in ["joint_positions", "multi"]
        if self._diffusion_var == "multi":
            if self._action_mode == "traj":
                use_traj = False
            else:
                use_traj = True
        envs = {
            task: self._dataset._get_env(
                task,
                headless=True,
                use_traj=use_traj,
            )
            for task in self._cfgs.env.tasks
        }

        manager = Manager()
        returns = manager.dict()
        queue = manager.Queue()

        def env_fn(env, task, queue, returns):
            env.launch()

            while True:
                msg = queue.get()

                if msg == -1:
                    env.env.shutdown()

                demo, trajs = msg
                env.reset_to_demo(demo)

                for i, traj in enumerate(trajs):
                    traj = traj.reshape(-1)

                    if i == len(trajs) - 2:
                        gripper = np.array([0], dtype=np.float32)
                    elif i == len(trajs) - 1:
                        if task in HOLD_TASKS:
                            gripper = np.array([0], dtype=np.float32)
                        else:
                            gripper = np.array([1], dtype=np.float32)
                    else:
                        gripper = np.array([1], dtype=np.float32)

                    action = np.concatenate([traj, gripper], axis=0)

                    ts = env.step(action, record=True)

                returns.put(ts)

        self._env_queues = dict()
        self._env_returns = dict()
        self._env_proc = dict()
        for task, env in envs.items():
            returns = manager.Queue()
            queue = manager.Queue()

            self._env_proc[task] = Process(
                target=env_fn, args=(env, task, queue, returns)
            )
            self._env_queues[task] = queue
            self._env_returns[task] = returns

        for proc in self._env_proc.values():
            proc.start()

    def _set_logger(self) -> None:
        """
        Set up the logger.

        This function sets up the logger for the object by creating a log directory and initializing the logging configuration. The log directory is created based on the current date and time, and the task names. The logging configuration is initialized using the project name and the configuration parameters.

        Parameters:
            None

        Returns:
            None
        """
        task_names = "_".join(self._cfgs.env.tasks)
        dt = datetime.datetime.now()
        self._log_path = os.path.join(
            "snapshots",
            "low_level",
            task_names,
            str(dt)[:19].replace(" ", "_"),
        )
        configs = dict(self._cfgs)
        configs["log_path"] = self._log_path

        os.makedirs(self._log_path, exist_ok=True)
        self._run = wandb.init(
            project=self._project_name,
            config=configs,
            name=self._cfgs.run_name,
        )

    def _reset_parameters(self) -> None:
        """
        Reset the parameters of the ema model.
        """
        self._ema_model.load_state_dict(self._model.state_dict())

    def _step_ema(self):
        """
        Updates the exponential moving average (EMA) of the model if the current step is greater than or equal to the start step for EMA.

        Parameters:
            None

        Returns:
            None
        """
        if self._step < self._step_start_ema:
            self._reset_parameters()
            return
        self._ema.update_model_average(self._ema_model, self._model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def generate_traj(self, **kwargs) -> dict:
        """
        Generates a diffusion trajectory.

        Args:
            cond (type): The conditions for generating the trajectory.
            **kwargs: Additional keyword arguments for generating the trajectory.

        Returns:
            dict: A dictionary containing the generated trajectory results.
        """
        cond = kwargs["cond"]
        del kwargs["cond"]
        if "x" in kwargs:
            del kwargs["x"]

        kwargs["robot"] = self._robot
        traj_results = self._model.conditional_sample(cond=cond, **kwargs)
        ret_results = {}
        for k, v in traj_results.items():
            ret_results[k] = {kk: vv.detach().cpu().numpy() for kk, vv in v.items()}
        return ret_results

    def get_eval_log(
        self,
        pcds: np.ndarray,
        rgbs: np.ndarray,
        predicted_trajs: np.ndarray,
        diffusion_var: str,
        gt_trajs: np.ndarray = None,
    ) -> dict:
        """
        Generates a dictionary containing evaluation metrics for predicted trajectories.

        Parameters:
            pcds (np.ndarray): The point clouds. Shape: (N, 3).
            rgbs (np.ndarray): The RGB values. Shape: (N, 3).
            predicted_trajs (np.ndarray): The predicted trajectories. Shape: (M, T, 3).
            diffusion_var (str): The diffusion variance.
            gt_trajs (np.ndarray, optional): The ground truth trajectories. Shape: (M, T, 3).

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        metrics = {}

        if gt_trajs is not None:
            dist = (gt_trajs - predicted_trajs) ** 2
            trans_dist = np.sqrt(dist[..., :3].sum(axis=-1)).mean()
            rot_dist = np.sqrt(dist[..., 3:].sum(axis=-1)).mean()

            metrics[f"eval_trans_dist_{diffusion_var}"] = trans_dist
            metrics[f"eval_rot_dist_{diffusion_var}"] = rot_dist

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
                    marker=dict(size=6, color="red"),
                )
            ]
            plot_data += gt_plot

        fig = go.Figure(plot_data)
        # fig.show()
        metrics[f"trajectories_{diffusion_var}"] = fig
        return metrics

    def train(self, n_train_steps: int, train: bool = True) -> None:
        """
        Train the model for a specified number of steps.

        Args:
            n_train_steps (int): The number of training steps.
            train (bool, optional): Whether to train the model or not. Defaults to True.

        Returns:
            None
        """
        for _ in tqdm.tqdm(range(n_train_steps)):
            if train:
                for i in range(self._gradient_accumulate_every):
                    batch = next(self._dataloader)
                    batch = batch_to_device(batch, device=self._device)

                    loss, infos = self._model.loss(**batch, robot=self._robot)

                    loss = loss / self._gradient_accumulate_every
                    loss.backward()

                self._optimizer.step()
                self._optimizer.zero_grad()

                if self._step % self._update_ema_every == 0:
                    self._step_ema()

                metrics = {k: v.detach().item() for k, v in infos.items()}
                metrics["steps"] = self._step
                metrics["loss"] = loss.detach().item()
            else:
                metrics = {}

            if self._step % self._log_freq == 0 or not train:
                if self._online_eval and self._step > self._online_eval_start:
                    online_eval_logs = self.online_eval()
                    metrics.update(online_eval_logs)

                test_traj = next(self._eval_dataloader)
                test_traj = batch_to_device(test_traj, device=self._device)
                joint_pos = test_traj["joint_positions"]

                gt_trajs = (
                    self._robot.forward_kinematics_batch(joint_pos.view(-1, 7))
                    .view(*joint_pos.shape)
                    .cpu()
                    .detach()
                    .numpy()
                )

                eval_output = self.generate_traj(**test_traj)

                for k, v in eval_output.items():
                    eval_log = self.get_eval_log(
                        test_traj["pcds"].detach().cpu().numpy()[0],
                        test_traj["rgbs"].detach().cpu().numpy()[0],
                        v["traj"],
                        k,
                        gt_trajs=gt_trajs,
                    )
                    metrics.update({"eval/" + k: v for k, v in eval_log.items()})

                if self._log:
                    wandb.log(metrics)

            if (self._step + 1) % self._save_freq == 0 and train and self._log:
                torch.save(
                    self._model.state_dict(),
                    os.path.join(
                        self._log_path,
                        f"model_{self._step + 1}.pt",
                    ),
                )

            self._step += 1

    def _prepare_demo_data(self, task: str) -> tuple:
        """
        Prepares demo data for a given task.

        Args:
            task (str): The name of the task.

        Returns:
            tuple: A tuple containing the following:
                - demo: The selected demo.
                - data_dicts: A list of dictionaries containing the prepared data for each frame.
                - task_names: A list of task names.
                - gts: A dictionary containing the ground truth joint positions and gripper poses.
        """
        demos = self._eval_dataset._demos[task]
        demo_id = np.random.randint(len(demos))
        demo = demos[demo_id]

        key_frames = keypoint_discovery(demo=demo, stopping_delta=0.01)
        key_frames = [0] + key_frames

        pcds = []
        rgbs = []
        poses = []
        proprios = []
        joints = []
        gt_joints = []
        gt_poses = []

        obss = [demo._observations[k] for k in key_frames]

        for i in range(len(key_frames) - 1):
            _temp = []
            _temp_poses = []
            for j in range(key_frames[i], key_frames[i + 1]):
                _temp.append(demo._observations[j].joint_positions)
                _temp_poses.append(demo._observations[j].gripper_pose)
            gt_joints.append(np.array(_temp, dtype=np.float32))
            gt_poses.append(np.array(_temp_poses, dtype=np.float32))

        def proc_gt_trajs(traj):
            if len(traj) >= 64:
                indices = np.random.choice(len(traj), 64, replace=False)
                indices = np.sort(indices)
                traj = traj[indices]
            else:
                traj = np.concatenate(
                    [traj, traj[-1][None].repeat(64 - len(traj), axis=0)], axis=0
                )

            return traj

        gt_joints = [proc_gt_trajs(jts) for jts in gt_joints]
        gt_poses = [proc_gt_trajs(poses) for poses in gt_poses]

        for obs in obss:
            rgbs.append(obs.front_rgb)
            pcds.append(obs.front_point_cloud)
            poses.append(obs.gripper_pose)
            proprios.append(obs.get_low_dim_data())
            joints.append(obs.joint_positions)

        poses = torch.tensor(poses, dtype=torch.float32, device="cuda")
        poses = proc_quaternion(poses)
        joints = torch.tensor(joints, dtype=torch.float32, device="cuda")
        pcds = torch.tensor(pcds, dtype=torch.float32, device="cuda")
        rgbs = torch.tensor(rgbs, dtype=torch.float32, device="cuda")
        proprios = torch.tensor(proprios, dtype=torch.float32, device="cuda")
        rank = torch.zeros([1, self._rank_bins], dtype=torch.float32, device="cuda")
        rank[0, -1] = 1

        data_dicts = []
        task_names = []
        for i in range(len(pcds) - 1):
            if self._diffusion_var == "gripper_poses":
                cond = {0: poses[i : i + 1], -1: poses[i + 1 : i + 2]}
            else:
                cond = {0: joints[i : i + 1]}

            data_dicts.append(
                dict(
                    cond=cond,
                    pcds=pcds[i : i + 1],
                    rgbs=rgbs[i : i + 1],
                    proprios=proprios[i : i + 1],
                    rank=rank,
                    joint_positions=joints[i : i + 2][None],
                    start=poses[i : i + 1],
                    end=poses[i + 1 : i + 2],
                    gripper_poses=poses[i : i + 2][None],
                )
            )
            task_names.append(task)

        gts = {"joint_positions": gt_joints, "gripper_poses": gt_poses}
        return demo, data_dicts, task_names, gts

    def _dispatch_eval_trajs(self, trajs: list, task_names: list) -> dict:
        """
        Dispatches the given data to their corresponding task names and returns a dictionary.

        Args:
            trajs (list): A list of trajectories to be dispatched.
            task_names (list): A list of task names corresponding to the trajectories.

        Returns:
            dict: A dictionary where the keys are the task names and the values are lists of trajectories.
        """
        ret_dict = {k: [] for k in set(task_names)}

        for traj, task in zip(trajs, task_names):
            ret_dict[task].append(traj)

        return ret_dict

    def online_eval(self) -> dict:
        """
        Performs online evaluation of the model.

        Returns:
            dict: A dictionary containing various evaluation metrics.
        """
        metrics = {}
        data_dicts, tasks_names = [], []
        demos = {}
        gts = {}
        for t in self._cfgs.tasks:
            demo, data_dict, task_name, gt = self._prepare_demo_data(t)
            demos[t] = demo
            data_dicts.extend(data_dict)
            tasks_names.extend(task_name)
            gts[t] = gt

        # data_dicts = recursive_concatenate(data_dicts, dim=0)
        data_dicts = concatenate_tensors_dict_list(data_dicts, dim=0)
        trajs_dict_all = self.generate_traj(**data_dicts)
        trajs_dict = trajs_dict_all[self._diffusion_var]
        if self._diffusion_var in ["joint_positions", "multi"]:
            trajs = trajs_dict["joint_positions"]
        else:
            trajs = trajs_dict["traj"]

        if self._diffusion_var == "multi":
            if self._action_mode == "traj":
                trajs = trajs_dict_all["gripper_poses"]["traj"]
            elif self._action_mode == "joints":
                trajs = trajs_dict_all["joint_positions"]["joint_positions"]
            else:
                trajs = trajs_dict_all["multi"]["joint_positions"]

        task_trajs = self._dispatch_eval_trajs(trajs, tasks_names)
        vis_trajs = self._dispatch_eval_trajs(trajs_dict["traj"], tasks_names)
        vis_rgbs = self._dispatch_eval_trajs(data_dicts["rgbs"], tasks_names)
        vis_pcds = self._dispatch_eval_trajs(data_dicts["pcds"], tasks_names)

        vis_trajs = {k: np.concatenate(v, axis=0)[None] for k, v in vis_trajs.items()}
        vis_rgbs = {k: v[0].cpu().detach().numpy() for k, v in vis_rgbs.items()}
        vis_pcds = {k: v[0].cpu().detach().numpy() for k, v in vis_pcds.items()}
        gt_vis_trajs = {
            k: np.concatenate(v["gripper_poses"], axis=0)[None] for k, v in gts.items()
        }

        for k in vis_trajs:
            traj, rgb, pcd = vis_trajs[k], vis_rgbs[k], vis_pcds[k]
            vis_metrics = self.get_eval_log(
                pcd, rgb, traj, self._diffusion_var, gt_trajs=gt_vis_trajs[k]
            )
            vis_metrics = {f"{k}/{kk}": vv for kk, vv in vis_metrics.items()}
            metrics.update(vis_metrics)

        for task in self._cfgs.tasks:
            self._env_queues[task].put((demos[task], task_trajs[task]))

        returns = dict()
        for task in self._cfgs.tasks:
            returns[task] = self._env_returns[task].get()

        successes = {k: v.reward > 0 for k, v in returns.items()}

        for k, v in successes.items():
            metrics[f"{k}/success"] = int(v)

        metrics["eval/overall_success"] = np.mean(
            np.array(list(successes.values()), dtype=np.float32)
        )

        ik_errors = []
        for task, v in returns.items():
            try:
                vid = v.summaries[0]
                name = vid.name
                video = vid.value[1:]
                metrics[task + "/" + name] = wandb.Video(video, fps=10)
                ik_error = np.array(v.summaries[1], dtype=np.float32)
                metrics[f"{task}/ik_error"] = ik_error
                ik_errors.append(ik_error)
            except Exception as e:
                print(e)

        metrics["eval/overall_ik_error_rate"] = np.mean(ik_errors)

        return metrics

    def __exit__():
        """Finish the wandb run and clean up any resources."""
        wandb.finish()
