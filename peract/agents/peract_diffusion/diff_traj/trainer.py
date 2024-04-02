"""Trainer for diffusion models."""

import copy
import datetime
import os
from multiprocessing import Manager, Process

import numpy as np
import plotly.graph_objects as go
import torch
import tqdm
import wandb
from dataset.rl_bench_dataset import RLBenchDataset
from torch.utils.data import DataLoader, RandomSampler

from agents.peract_diffusion.diff_traj.diffusion import GaussianInvDynDiffusion
from agents.peract_diffusion.diff_traj.utils import (
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
        cfgs,
        diffusion_model: GaussianInvDynDiffusion,
        dataset: RLBenchDataset,
        eval_dataset: RLBenchDataset,
        scene_bounds,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=1000,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device="cuda",
        save_checkpoints=False,
        log=False,
        project_name="TrajDiffusion",
        online_eval=False,
        headless=True,
        rank_bins=10,
        pred_joints=False,
        inv_ik=True,
        robot=None,
        diffusion_var="gripper_poses",
        online_eval_start=0,
        action_mode="joints",
    ):
        super().__init__()
        self._cfgs = cfgs
        self._model = diffusion_model
        self._ema = EMA(ema_decay)
        self._ema_model = copy.deepcopy(self._model)
        self._update_ema_every = update_ema_every
        self._save_checkpoints = save_checkpoints

        self._scene_bounds = scene_bounds

        self._step_start_ema = step_start_ema
        self._log_freq = log_freq
        self._sample_freq = sample_freq
        self._save_freq = save_freq
        self._label_freq = label_freq
        self._save_parallel = save_parallel

        self._batch_size = train_batch_size
        self._gradient_accumulate_every = gradient_accumulate_every
        self._log = log
        self._project_name = project_name
        self._online_eval = online_eval
        self._headless = headless
        self._rank_bins = rank_bins
        self._pred_joints = pred_joints
        self._online_eval_start = online_eval_start

        self._dataset = dataset
        self._eval_dataset = eval_dataset
        self._inv_ik = inv_ik
        self._action_mode = action_mode
        self._diffusion_var = diffusion_var

        self._robot = robot
        self._dataloader = cycle(
            DataLoader(
                self._dataset,
                batch_size=train_batch_size,
                num_workers=0,
                pin_memory=True,
            )
        )
        self._eval_dataloader = cycle(
            DataLoader(
                self._eval_dataset,
                batch_size=10,
                num_workers=0,
                pin_memory=True,
            )
        )

        self._optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=train_lr)

        self._bucket = bucket
        self._n_reference = n_reference

        self._reset_parameters()
        self._step = 0

        self._device = train_device
        if self._log:
            self._set_logger()

        if self._online_eval:
            self.create_envs()

    def create_envs(self):
        use_traj = (
            self._pred_joints
            or self._inv_ik
            or self._diffusion_var in ["joint_positions", "multi"]
        )
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
            for task in self._cfgs.tasks
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

    def _set_logger(self):
        task_names = "_".join(self._cfgs.tasks)
        dt = datetime.datetime.now()
        self._log_path = os.path.join(
            "logs",
            task_names,
            str(dt)[:19].replace(" ", "_"),
        )
        configs = dict(self._cfgs)
        configs["log_path"] = self._log_path

        os.makedirs(self._log_path, exist_ok=True)
        self._run = wandb.init(
            project=self._project_name,
            config=configs,
        )

    def _reset_parameters(self):
        self._ema_model.load_state_dict(self._model.state_dict())

    def _step_ema(self):
        if self._step < self._step_start_ema:
            self._reset_parameters()
            return
        self._ema.update_model_average(self._ema_model, self._model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def generate_traj(self, **kwargs):
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

    def get_eval_log(self, pcds, rgbs, predicted_trajs, diffusion_var, gt_trajs=None):
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
        metrics[f"trajectories_{diffusion_var}"] = fig
        return metrics

    def train(self, n_train_steps, train=True):
        for step in tqdm.tqdm(range(n_train_steps)):
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

                gt_trajs = test_traj["gripper_poses"].detach().cpu().numpy()

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
                    torch.save(
                        self._model.state_dict(),
                        os.path.join(
                            self._log_path,
                            "model.pt",
                        ),
                    )

            self._step += 1

    def _prepare_demo_data(self, task):
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

    def _dispatch_eval_trajs(self, trajs, task_names):
        ret_dict = {k: [] for k in set(task_names)}

        for traj, task in zip(trajs, task_names):
            ret_dict[task].append(traj)

        return ret_dict

    def online_eval(self):
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
        if (
            self._pred_joints
            or self._inv_ik
            or self._diffusion_var in ["joint_positions", "multi"]
        ):
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
        wandb.finish()
