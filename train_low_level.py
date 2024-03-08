"""Main script."""

import os

import hydra
import torch

from rk_diffuser.dataset.rl_bench_dataset import RLBenchDataset
from rk_diffuser.models.multi_level_diffusion import MultiLevelDiffusion
from rk_diffuser.robot import DiffRobot
from rk_diffuser.trainer import Trainer
from rk_diffuser.utils import load_checkpoint

VALID_DIFFUSION_VARS = ["gripper_poses", "joint_positions", "multi"]


def _create_agent_fn(
    cfgs,
    device,
    diffusion_var="multi",
    sim=True,
    diff_optim=True,
    diff_optim_steps=100,
    diff_lr=10,
    pose_augment=False,
):
    robot = None
    if diffusion_var in ["joint_positions", "multi"]:
        robot = DiffRobot("./rk_diffuser/panda_urdf/panda.urdf", "Pandatip")
    if robot is not None:
        robot.to(device)

    assert diffusion_var in VALID_DIFFUSION_VARS
    if diffusion_var == "multi":
        diffusion_pose = hydra.utils.instantiate(
            cfgs,
            diffusion_var="gripper_poses",
        )
        diffusion_joints = hydra.utils.instantiate(
            cfgs,
            diffusion_var="joint_positions",
        )

        diffusion = MultiLevelDiffusion(
            {
                "gripper_poses": diffusion_pose,
                "joint_positions": diffusion_joints,
            },
            diff_optim=diff_optim,
            diff_optim_steps=diff_optim_steps,
            diff_lr=diff_lr,
            pose_augment=pose_augment,
            sim=sim,
        )
    else:
        diffusion = hydra.utils.instantiate(
            cfgs,
            diffusion_var=diffusion_var,
        )

    diffusion.to(device)
    return robot, diffusion


@hydra.main(
    config_path="cfgs",
    config_name="diffuser_config",
    version_base=None,
)
def main(cfgs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    robot, diffusion = _create_agent_fn(
        cfgs.method,
        device,
        diffusion_var=cfgs.diffusion_var,
        sim=cfgs.env.name == "sim",
        diff_optim=cfgs.diff_optim,
        diff_optim_steps=cfgs.diff_optim_steps,
        diff_lr=cfgs.diff_lr,
        pose_augment=False,
    )

    if os.path.exists(cfgs.load_model_path):
        load_checkpoint(diffusion, cfgs.load_model_path, cfgs.method.backbone)

    diffusion = diffusion.to(device)

    assert os.path.isdir(cfgs.env.data_path)
    if cfgs.env.name == "sim":
        dataset = RLBenchDataset(
            cfgs.env.tasks,
            cfgs.env.tasks_ratio,
            cfgs.env.cameras,
            cfgs.env.num_episodes,
            data_raw_path=os.path.join(cfgs.env.data_path, "train"),
            traj_len=cfgs.method.horizon,
            frame_skips=cfgs.frame_skips,
            observation_dim=cfgs.method.observation_dim,
            rank_bins=cfgs.method.rank_bins,
            robot=robot,
            diffusion_var=cfgs.diffusion_var,
            demo_aug_ratio=cfgs.env.demo_aug_ratio,
            demo_aug_min_len=cfgs.env.demo_aug_min_len,
            use_cached=cfgs.use_cached,
            ds_img_size=cfgs.ds_img_size,
        )

        eval_dataset = RLBenchDataset(
            cfgs.env.tasks,
            cfgs.env.tasks_ratio,
            cfgs.env.cameras,
            cfgs.env.num_episodes // 2,
            data_raw_path=os.path.join(cfgs.env.data_path, "eval"),
            traj_len=cfgs.method.horizon,
            frame_skips=cfgs.frame_skips,
            observation_dim=cfgs.method.observation_dim,
            rank_bins=cfgs.method.rank_bins,
            robot=robot,
            diffusion_var=cfgs.diffusion_var,
            training=False,
            demo_aug_ratio=cfgs.env.demo_aug_ratio,
            demo_aug_min_len=cfgs.env.demo_aug_min_len,
            use_cached=cfgs.use_cached,
            ds_img_size=cfgs.ds_img_size,
        )
    else:
        # RLBench complains when importing cv2 before it so we move it here
        from rk_diffuser.dataset.realworld_dataset import RealWorldDataset

        dataset = RealWorldDataset(
            cfgs.env.tasks,
            cfgs.env.cameras,
            cfgs.env.num_episodes,
            data_raw_path=cfgs.env.data_path,
            traj_len=cfgs.method.horizon,
            frame_skips=cfgs.frame_skips,
            observation_dim=cfgs.method.observation_dim,
            rank_bins=cfgs.method.rank_bins,
            robot=robot,
            diffusion_var=cfgs.diffusion_var,
            demo_aug_ratio=cfgs.env.demo_aug_ratio,
            demo_aug_min_len=cfgs.env.demo_aug_min_len,
            camera_extrinsics=cfgs.env.camera_extrinsics,
            load_processed_data=cfgs.env.load_processed_data,
            save_processed_data=cfgs.env.save_processed_data,
        )

        eval_dataset = RealWorldDataset(
            cfgs.env.tasks,
            cfgs.env.cameras,
            cfgs.env.num_episodes_eval,
            data_raw_path=cfgs.env.data_path,
            traj_len=cfgs.method.horizon,
            frame_skips=cfgs.frame_skips,
            observation_dim=cfgs.method.observation_dim,
            rank_bins=cfgs.method.rank_bins,
            robot=robot,
            diffusion_var=cfgs.diffusion_var,
            demo_aug_ratio=cfgs.env.demo_aug_ratio,
            demo_aug_min_len=cfgs.env.demo_aug_min_len,
            camera_extrinsics=cfgs.env.camera_extrinsics,
            training=False,
            load_processed_data=cfgs.env.load_processed_data,
            save_processed_data=cfgs.env.save_processed_data,
        )

    trainer = Trainer(
        cfgs=cfgs,
        diffusion_model=diffusion,
        dataset=dataset,
        eval_dataset=eval_dataset,
        train_batch_size=cfgs.batch_size,
        log=cfgs.log,
        log_freq=cfgs.log_freq,
        save_freq=cfgs.save_freq,
        scene_bounds=cfgs.env.scene_bounds,
        project_name=cfgs.project_name,
        online_eval=cfgs.online_eval,
        headless=cfgs.headless,
        rank_bins=cfgs.method.rank_bins,
        robot=robot,
        diffusion_var=cfgs.diffusion_var,
        online_eval_start=cfgs.online_eval_start,
        action_mode=cfgs.action_mode,
    )

    trainer.train(cfgs.n_epochs, not cfgs.eval_only)


if __name__ == "__main__":
    main()
