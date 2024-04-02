"""DDPM diffusion model."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import agents.peract_diffusion.diff_traj.utils as utils
from agents.peract_diffusion.diff_traj.helpers import (
    Losses,
    apply_conditioning,
    cosine_beta_schedule,
    extract,
)

DELTA_LOSS_SCALE = 100.0


class GaussianInvDynDiffusion(nn.Module):
    def __init__(
        self,
        model_fn,
        horizon,
        observation_dim,
        action_dim,
        scene_bounds,
        joint_limits,
        n_timesteps=1000,
        loss_type="l1",
        clip_denoised=False,
        predict_epsilon=True,
        hidden_dim=256,
        loss_discount=1.0,
        loss_weights=None,
        condition_guidance_w=0.1,
        train_only_inv=False,
        reverse_train=True,
        conditions=[],
        delta_actions=True,
        noise_init_method="normal",
        loss_fn="state_chamfer",
        coverage_weight=1.0,
        detach_reverse=True,
        pred_deltas=True,
        pred_joints=True,
        joint_weight=1.0,
        inv_ik=True,
        robot_offset=[],
        robot_rot=[],
        trans_loss_scale=100.0,
        rot_loss_scale=1.0,
        diffusion_var="gripper_poses",
        joint_pred_pose_loss=True,
        joint_loss_scale=100.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim
        self.model = model_fn()
        self.train_only_inv = train_only_inv
        self.condition_guidance_w = condition_guidance_w
        self.diffusion_var = diffusion_var
        self.delta_actions = delta_actions
        self.noise_init_method = noise_init_method
        self.detach_reverse = detach_reverse
        self.pred_deltas = pred_deltas
        self.pred_joints = pred_joints
        self.hidden_dim = hidden_dim
        self.joint_weight = joint_weight
        self.inv_ik = inv_ik
        self.joint_pred_pose_loss = joint_pred_pose_loss

        self.trans_loss_scale = trans_loss_scale
        self.rot_loss_scale = rot_loss_scale
        self.joint_loss_scale = joint_loss_scale

        robot_offset = torch.FloatTensor(robot_offset)
        scene_bounds = torch.FloatTensor(scene_bounds)
        joint_limits = torch.FloatTensor(joint_limits)
        robot_rot = torch.FloatTensor(robot_rot)

        self.conditions = conditions
        self.reverse_train = (
            reverse_train
            and diffusion_var == "gripper_poses"
            and "end" in self.conditions
        )

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("joint_limits", joint_limits)
        self.register_buffer("robot_offset", robot_offset)
        self.register_buffer("robot_rot", robot_rot)
        self.register_buffer("scene_bounds", scene_bounds)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        if self.pred_joints:
            self.inv_joints = nn.Sequential(
                nn.Linear(7 * 2, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.action_dim),
            )

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)

        if loss_fn == "state_chamfer":
            self.loss_fn = Losses[loss_fn](loss_weights, coverage_weight, loss_type)
        else:
            self.loss_fn = Losses[loss_fn](loss_weights)

    def gripper_pose_loss(self, p1, p2):
        trans, quat = p1[..., :3], p1[..., 3:]
        trans_recon, quat_recon = p2[..., :3], p2[..., 3:]

        trans_loss = F.mse_loss(trans, trans_recon) * self.trans_loss_scale
        rot_loss = (
            utils.geodesic_distance_between_quaternions(quat, quat_recon).mean()
            * self.rot_loss_scale
        )

        info = {
            "trans_loss": trans_loss,
            "rot_loss": rot_loss,
        }

        loss = trans_loss + rot_loss

        return loss, info

    def get_loss_weights(self, discount):
        """
        sets loss coefficients for trajectory

        action_weight   : float
            coefficient on first action loss
        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        weights_dict    : dict
            { i: c } multiplies dimension i of observation loss by c
        """
        self.action_weight = 1
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, **kwargs):
        if len(self.conditions) > 0:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, use_dropout=False, **kwargs)
            epsilon_uncond = self.model(
                x,
                cond,
                t,
                force_dropout=True,
                **kwargs,
            )
            epsilon = epsilon_uncond + self.condition_guidance_w * (
                epsilon_cond - epsilon_uncond
            )
        else:
            epsilon = self.model(x, cond, t, **kwargs)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            if self.diffusion_var == "gripper_poses":
                x_clipped = []
                for i in range(3):
                    x_clipped.append(
                        torch.clamp(
                            x_recon[..., i],
                            self.scene_bounds[0][i],
                            self.scene_bounds[1][i],
                        )
                    )
                xyz = torch.stack(x_clipped, dim=-1)
                quat = x_recon[..., 3:]
                quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
                x_recon = torch.cat([xyz, quat], dim=-1)
                x_recon = utils.proc_quaternion(x_recon)
            else:
                x_clipped = []
                for i in range(7):
                    x_clipped.append(
                        torch.clamp(
                            x_recon[..., i],
                            self.joint_limits[0][i],
                            self.joint_limits[1][i],
                        )
                    )
                x_recon = torch.stack(x_clipped, dim=-1)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, cond, t, **kwargs):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            cond=cond,
            t=t,
            **kwargs,
        )
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(
        self,
        shape,
        cond,
        verbose=True,
        return_diffusion=False,
        **kwargs,
    ):
        device = self.betas.device

        with torch.no_grad():
            batch_size = shape[0]
            x = 0.5 * self.init_noise(shape, device=device, cond=cond)
            x = apply_conditioning(x, cond, 0)

            if return_diffusion:
                diffusion = [x]

            progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
            for i in reversed(range(0, self.n_timesteps)):
                timesteps = torch.full(
                    (batch_size,), i, device=device, dtype=torch.long
                )
                x = self.p_sample(x, cond, timesteps, **kwargs)
                x = apply_conditioning(x, cond, 0)

                progress.update({"t": i})

                if return_diffusion:
                    diffusion.append(x)

            progress.close()

        if self.delta_actions:
            x = utils.cumsum_traj(x, cond)

        return_dict = {"traj": x}
        if self.inv_ik and self.diffusion_var == "gripper_poses":
            seq_len = shape[1]
            joints = kwargs["joint_positions"].unsqueeze(1).repeat(1, seq_len, 1)
            transposed_x = torch.cat(
                [
                    x[..., :3] + self.robot_offset[None, None, ...],
                    x[..., -1:],
                    x[..., 3:-1],
                ],
                dim=-1,
            ).detach()
            robot = kwargs["robot"]

            joints = joints.view(-1, 7)
            transposed_x.requires_grad_(False)
            transposed_x = transposed_x.view(-1, 7)
            for i in range(1000):
                joints = robot.inverse_kinematics_autodiff_single_step_batch_adam(
                    transposed_x, joints
                )

            predicted_x = robot.forward_kinematics_batch(joints)
            return_dict["traj"] = predicted_x.view(batch_size, seq_len, -1)
            return_dict["joint_positions"] = joints.view(batch_size, seq_len, -1)

        elif self.diffusion_var == "joint_positions":
            seq_len = shape[1]
            robot = kwargs["robot"]
            joints = x.view(-1, 7)
            predicted_traj = robot.forward_kinematics_batch(joints).view(
                batch_size, seq_len, 7
            )

            return_dict["traj"] = predicted_traj
            return_dict["joint_positions"] = x

        elif self.pred_joints:
            joints = self.inv_joints(torch.cat([x[:, :-1], x[:, 1:]], dim=-1))
            joints = torch.cat([joints, joints[:, -1:]], dim=1)
            return_dict["joint_positions"] = joints

        if return_diffusion:
            return_dict["diffusion_hist"] = torch.stack(diffusion, dim=1)

        return return_dict

    # @torch.compile
    def conditional_sample(self, cond, horizon=None, *args, **kwargs):
        """
        conditions : [ (time, state), ... ]
        """
        cond = self.proc_cond(cond)
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        ret_dict = self.p_sample_loop(shape, cond, *args, **kwargs)

        return {self.diffusion_var: ret_dict}

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def init_noise(self, shape, device, cond=None):
        if self.noise_init_method == "bounded_uniform":
            noise = torch.rand(shape, device=device)

            xyz = noise[..., :3]
            quat = noise[..., 3:]

            scene_size = (self.scene_bounds[1] - self.scene_bounds[0])[None, None]
            xyz = xyz * scene_size + self.scene_bounds[0][None, None]
            quat = (quat - 0.5) * 2

            return torch.cat([xyz, quat], dim=-1)
        elif self.noise_init_method == "normal":
            return torch.randn(shape, device=device)
        elif self.noise_init_method == "linear":
            assert cond is not None
            start, end = cond[0], cond[-1]
            steps = shape[1]
            res = (end - start) / steps
            noise = torch.arange(steps, device=device)[None, :, None]
            noise = start.unsqueeze(1) + res.unsqueeze(1) * noise
            xyz = noise[..., :3]
            quat = noise[..., 3:]
            quat = quat / torch.linalg.norm(quat, keepdims=True)

            return torch.cat([xyz, quat], dim=-1)
        else:
            raise NotImplementedError

    def p_losses(self, x_start, cond, t, robot=None, **kwargs):
        noise = self.init_noise(x_start.shape, x_start.device, cond)
        batch_size = x_start.size(0)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        conditions = kwargs
        if self.reverse_train:
            reversed_x = torch.flip(x_noisy, dims=(1,))
            reversed_cond = {-1: cond[0], 0: cond[-1]}
            x_noisy = torch.cat([x_noisy, reversed_x], dim=0)
            cond = {k: torch.cat([v, reversed_cond[k]], dim=0) for k, v in cond.items()}

            reversed_noise = torch.flip(noise, dims=(1,))
            noise = torch.cat([noise, reversed_noise], dim=0)

            reversed_start = torch.flip(x_start, dims=(1,))
            x_start = torch.cat([x_start, reversed_start], dim=0)

            new_conditions = {}
            for k, v in conditions.items():
                if k == "start":
                    new_conditions[k] = torch.cat(
                        [conditions["start"], conditions["end"]], dim=0
                    )
                elif k == "end":
                    new_conditions[k] = torch.cat(
                        [conditions["end"], conditions["start"]], dim=0
                    )
                else:
                    new_conditions[k] = v.repeat(
                        2, *[1 for _ in range(len(v.shape[1:]))]
                    )

            conditions = new_conditions

            t = t.repeat(
                2,
            )

        x_noisy = apply_conditioning(x_noisy, cond, 0)

        x_recon = self.model(x_noisy, cond, t, **conditions)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, 0)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        elif self.delta_actions:
            x_recon_cumsum = utils.cumsum_traj(x_recon, cond)
            loss, info = self.loss_fn(x_recon_cumsum, x_start)

            if self.pred_deltas:
                deltas_hat = x_recon[:, 1:-1]
                deltas = x_start[:, 1:-1] - x_start[:, :-2]
                delta_loss = F.mse_loss(deltas_hat, deltas) * DELTA_LOSS_SCALE

                info["delta_loss"] = delta_loss

                loss += delta_loss
        else:
            if self.diffusion_var == "gripper_poses":
                loss = F.mse_loss(x_start, x_recon) * self.trans_loss_scale
                info = {"pose_loss": loss}
            else:
                loss = F.mse_loss(x_start, x_recon)
                loss = loss * self.joint_loss_scale
                info = {"joint_loss": loss}

                if self.joint_pred_pose_loss:
                    assert robot is not None
                    predicted_poses = robot.forward_kinematics_batch(
                        x_recon.contiguous().view(-1, 7)
                    ).view(batch_size, -1, 7)
                    gt_poses = kwargs["gripper_poses"]

                    pose_loss = F.mse_loss(predicted_poses[..., :3], gt_poses[..., :3])
                    pose_loss = pose_loss * self.trans_loss_scale

                    loss += pose_loss
                    info["pose_loss"] = pose_loss

        if self.pred_joints and self.diffusion_var == "gripper_poses":
            joints_hat = self.inv_joints(
                torch.cat([x_start[:, :-1], x_start[:, 1:]], dim=-1)
            )
            joints = conditions["joint_positions"]
            loss_hat = F.mse_loss(joints_hat, joints[:, 1:])

            loss += loss_hat * self.joint_weight
            info["joint_loss"] = loss_hat

        if self.reverse_train:
            x_forward, x_backward = torch.split(x_recon, batch_size, dim=0)
            x_backward = torch.flip(x_backward, dims=(1,))
            if self.detach_reverse:
                x_backward = x_backward.detach()

            consistency_loss, _ = self.gripper_pose_loss(x_forward, x_backward)
            loss += consistency_loss
            info["consistency_loss"] = consistency_loss

        return loss, info

    def proc_cond(self, cond: dict) -> dict:
        new_cond = {k: v for k, v in cond.items()}

        if -1 in new_cond and "end" not in self.conditions:
            del new_cond[-1]

        return new_cond

    # @torch.compile
    def loss(self, x, cond, **kwargs):
        cond = self.proc_cond(cond)
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, **kwargs)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)
