"""DDPM diffusion model."""

import torch
from torch import nn
from torch.nn import functional as F

import rk_diffuser.utils as utils
from rk_diffuser.models.helpers import (
    Losses,
    apply_conditioning,
    cosine_beta_schedule,
    extract,
)
from rk_diffuser.models.temporal import Temporal
from rk_diffuser.robot import DiffRobot

DELTA_LOSS_SCALE = 100.0


def _create_model_fn(
    horizon: int,
    transition_dim: int,
    dim: int,
    dim_mults: list,
    scene_bounds: list,
    condition_dropout: bool,
    conditions: list,
    proprio_dim: int,
    hard_conditions: list,
    rank_bins: int,
    backbone: str,
    num_encoder_layers: int,
    num_decoder_layers: int,
    n_head: int,
    causal_attn: bool,
    depth_proc: str,
    rgb_encoder: str,
):
    model = Temporal(
        horizon=horizon,
        transition_dim=transition_dim,
        dim=dim,
        dim_mults=dim_mults,
        scene_bounds=scene_bounds,
        condition_dropout=condition_dropout,
        conditions=conditions,
        proprio_dim=proprio_dim,
        hard_conditions=hard_conditions,
        rank_bin=rank_bins,
        backbone=backbone,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        n_head=n_head,
        causal_attn=causal_attn,
        depth_proc=depth_proc,
        rgb_encoder=rgb_encoder,
    )

    return model


class GaussianDynDiffusion(nn.Module):
    def __init__(
        self,
        horizon: int,
        observation_dim: int,
        dim_mults: list,
        action_dim: int,
        scene_bounds: list,
        joint_limits: list,
        n_timesteps: int,
        loss_type: str,
        clip_denoised: bool,
        predict_epsilon: bool,
        hidden_dim: int,
        loss_discount: float,
        condition_guidance_w: float,
        reverse_train: bool,
        conditions: list,
        hard_conditions: list,
        noise_init_method: str,
        loss_fn: str,
        coverage_weight: float,
        detach_reverse: bool,
        joint_weight: float,
        robot_offset: list,
        trans_loss_scale: float,
        rot_loss_scale: float,
        diffusion_var: str,
        joint_pred_pose_loss: bool,
        joint_loss_scale: float,
        rank_bins: int,
        backbone: str,
        num_decoder_layers: int,
        num_encoder_layers: int,
        n_head: int,
        causal_attn: bool,
        depth_proc: str,
        rgb_encoder: str,
    ):
        """
        Initializes the class with the given parameters.

        Args:
            horizon (int): The number of timesteps in the diffusion process.
            observation_dim (int): The dimension of the observation space.
            dim_mults (list): The dimension multipliers for unet.
            action_dim (int): The dimension of the action space.
            scene_bounds (list): The bounds of the scene.
            joint_limits (list): The limits of the joints.
            n_timesteps (int): The number of diffusion timesteps.
            loss_type (str): The type of loss function.
            clip_denoised (bool): Whether to clip the denoised states.
            predict_epsilon (bool): Whether to predict epsilon.
            hidden_dim (int): The dimension of the hidden layers.
            loss_discount (float): The discount factor for the loss.
            condition_guidance_w (float): The weight of the condition guidance.
            reverse_train (bool): Whether to reverse the training process.
            conditions (list): The conditions.
            hard_conditions (list): The conditions for which we don't do classifier-free guidance.
            noise_init_method (str): The method for initializing noise.
            loss_fn (str): The loss function.
            coverage_weight (float): The weight of the coverage.
            detach_reverse (bool): Whether to detach the reverse process.
            joint_weight (float): The weight of the joints.
            robot_offset (list): The offset of the robot.
            trans_loss_scale (float): The scale for the translation loss.
            rot_loss_scale (float): The scale for the rotation loss.
            diffusion_var (str): The diffusion variable.
            joint_pred_pose_loss (bool): Whether to use joint predicted pose loss.
            joint_loss_scale (float): The scale for the joint loss.
            rank_bins (int): The number of discrete bins for traj rank.
            backbone (str): The Temporal Net backbone.
            num_decoder_layers (int): Number of decoder layers for the transformer backbone.
            num_encoder_layers (int): Number of decoder layers for the transformer backbone.
            n_heads (int): Number of transformer heads.
            causal_attn (bool): Whether to use causal attention in the transformer backbone.
            depth_proc (str): Depth processing method.
            rgb_encoder (str): RGB encoder type.

        Returns:
            None
        """
        super().__init__()
        self._horizon = horizon
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._transition_dim = observation_dim
        self._model = _create_model_fn(
            horizon=horizon,
            transition_dim=observation_dim,
            dim=hidden_dim,
            dim_mults=dim_mults,
            scene_bounds=scene_bounds,
            condition_dropout=True,
            conditions=conditions,
            proprio_dim=24,
            hard_conditions=hard_conditions,
            rank_bins=rank_bins,
            backbone=backbone,
            num_decoder_layers=num_decoder_layers,
            num_encoder_layers=num_encoder_layers,
            n_head=n_head,
            causal_attn=causal_attn,
            depth_proc=depth_proc,
            rgb_encoder=rgb_encoder,
        )
        self._condition_guidance_w = condition_guidance_w
        self._diffusion_var = diffusion_var
        self._noise_init_method = noise_init_method
        self._detach_reverse = detach_reverse
        self._hidden_dim = hidden_dim
        self._joint_weight = joint_weight
        self._joint_pred_pose_loss = joint_pred_pose_loss

        self._trans_loss_scale = trans_loss_scale
        self._rot_loss_scale = rot_loss_scale
        self._joint_loss_scale = joint_loss_scale

        robot_offset = torch.FloatTensor(robot_offset)
        scene_bounds = torch.FloatTensor(scene_bounds)
        joint_limits = torch.FloatTensor(joint_limits)

        self._conditions = conditions
        self._reverse_train = (
            reverse_train
            and diffusion_var == "gripper_poses"
            and "end" in self._conditions
        )

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self._n_timesteps = int(n_timesteps)
        self._clip_denoised = clip_denoised
        self._predict_epsilon = predict_epsilon

        self.register_buffer("_joint_limits", joint_limits)
        self.register_buffer("_robot_offset", robot_offset)
        self.register_buffer("_scene_bounds", scene_bounds)
        self.register_buffer("_betas", betas)
        self.register_buffer("_alphas_cumprod", alphas_cumprod)
        self.register_buffer("_alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("_sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "_sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "_log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "_sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "_sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("_posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "_posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "_posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "_posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)

        if loss_fn == "state_chamfer":
            self._loss_fn = Losses[loss_fn](loss_weights, coverage_weight, loss_type)
        else:
            self._loss_fn = Losses[loss_fn](loss_weights)

    def gripper_pose_loss(self, p1, p2):
        trans, quat = p1[..., :3], p1[..., 3:]
        trans_recon, quat_recon = p2[..., :3], p2[..., 3:]

        trans_loss = F.mse_loss(trans, trans_recon) * self._trans_loss_scale
        rot_loss = (
            utils.geodesic_distance_between_quaternions(quat, quat_recon).mean()
            * self._rot_loss_scale
        )

        info = {
            "trans_loss": trans_loss,
            "rot_loss": rot_loss,
        }

        loss = trans_loss + rot_loss

        return loss, info

    def get_loss_weights(self, discount: float) -> torch.Tensor:
        """
        Sets loss coefficients for trajectory.

        Args:
            discount (float): The discount factor.

        Returns:
            torch.Tensor: The loss weights for each timestep and dimension.
        """
        dim_weights = torch.ones(self._observation_dim, dtype=torch.float32)

        # Decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self._horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)

        # Set loss weights to 0 for t=0 if predict_epsilon is True
        if self._predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(
        self, x_t: torch.tensor, t: torch.tensor, noise: torch.tensor
    ) -> torch.tensor:
        """
        Predicts the starting point from noise.

        Args:
            x_t (torch.tensor): The input tensor.
            t (torch.tensor): The time tensor.
            noise (torch.tensor): The noise tensor.

        Returns:
            torch.tensor: The predicted x_0.
        """
        if self._predict_epsilon:
            return (
                extract(self._sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self._sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(
        self, x_start: torch.tensor, x_t: torch.tensor, t: torch.tensor
    ) -> tuple:
        """
        Compute the posterior distribution of a time-dependent variable.

        Args:
            x_start (torch.tensor): The initial value of the variable.
            x_t (torch.tensor): The current value of the variable.
            t (torch.tensor): The time at which the posterior is computed.

        Returns:
            tuple: A tuple containing the posterior mean, posterior variance, and
                   posterior log variance clipped.
        """
        posterior_mean = (
            extract(self._posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self._posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self._posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self._posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, x: torch.tensor, cond: dict, t: torch.tensor, **kwargs
    ) -> tuple:
        """
        Calculates the mean, variance, and log variance of the posterior distribution
        given the input tensor `x`, condition dictionary `cond`, and time tensor `t`.

        Parameters:
            x (torch.tensor): The input tensor.
            cond (dict): The condition dictionary.
            t (torch.tensor): The time tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the model mean, posterior variance, and
            posterior log variance.
        """
        if len(self._conditions) > 0:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self._model(x, cond, t, use_dropout=False, **kwargs)
            epsilon_uncond = self._model(
                x,
                cond,
                t,
                force_dropout=True,
                **kwargs,
            )
            epsilon = epsilon_uncond + self._condition_guidance_w * (
                epsilon_cond - epsilon_uncond
            )
        else:
            epsilon = self._model(x, cond, t, **kwargs)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self._clip_denoised:
            if self._diffusion_var == "gripper_poses":
                x_clipped = []
                for i in range(3):
                    x_clipped.append(
                        torch.clamp(
                            x_recon[..., i],
                            self._scene_bounds[0][i],
                            self._scene_bounds[1][i],
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
                            self._joint_limits[0][i],
                            self._joint_limits[1][i],
                        )
                    )
                x_recon = torch.stack(x_clipped, dim=-1)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(
        self, x: torch.tensor, cond: dict, t: torch.tensor, **kwargs
    ) -> torch.tensor:
        """
        Calculate the conditional probability distribution for the given input tensor.

        Args:
            x (torch.tensor): The input tensor.
            cond (dict): A dictionary of conditional variables.
            t (torch.tensor): The target tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.tensor: The conditional probability distribution tensor.
        """
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
        shape: list,
        cond: dict,
        verbose: bool = True,
        return_diffusion: bool = True,
        **kwargs,
    ) -> dict:
        """
        Sampling from the diffusion model.

        Args:
            shape (list): The shape of the input.
            cond (dict): The conditioning variables.
            verbose (bool, optional): Whether to display progress information. Defaults to True.
            return_diffusion (bool, optional): Whether to return the diffusion history. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The dictionary containing the trajectory, diffusion history, and joint positions (if applicable).
        """
        device = self._betas.device

        with torch.no_grad():
            batch_size = shape[0]
            x = 0.5 * self.init_noise(shape, device=device, cond=cond)
            x = apply_conditioning(x, cond, 0)

            if return_diffusion:
                diffusion = [x]

            progress = utils.Progress(self._n_timesteps) if verbose else utils.Silent()
            for i in reversed(range(0, self._n_timesteps)):
                timesteps = torch.full(
                    (batch_size,), i, device=device, dtype=torch.long
                )
                x = self.p_sample(x, cond, timesteps, **kwargs)
                x = apply_conditioning(x, cond, 0)

                progress.update({"t": i})

                if return_diffusion:
                    diffusion.append(x)

            progress.close()

        return_dict = {"traj": x}

        if self._diffusion_var == "joint_positions":
            seq_len = shape[1]
            robot = kwargs["robot"]
            joints = x.view(-1, 7)
            predicted_traj = robot.forward_kinematics_batch(joints).view(
                batch_size, seq_len, 7
            )

            return_dict["traj"] = predicted_traj
            return_dict["joint_positions"] = x

        if return_diffusion:
            diffusion_hist = torch.stack(diffusion, dim=1)
            if self._diffusion_var == "joint_positions":
                shape = diffusion_hist.shape
                diffusion_hist = diffusion_hist.view(-1, 7)
                diffusion_hist = robot.forward_kinematics_batch(diffusion_hist).view(
                    *shape
                )
            return_dict["diffusion_hist"] = diffusion_hist

        return return_dict

    def conditional_sample(
        self, cond: dict, horizon: int = None, *args, **kwargs
    ) -> dict:
        """
        A function that performs a conditional sample of the diffusion model.

        Args:
            cond (dict): The conditions for the sample.
            horizon (int, optional): The horizon for the sample. Defaults to None.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the sampled values.
        """
        cond = self.proc_cond(cond)
        batch_size = len(cond[0])
        horizon = horizon or self._horizon
        shape = (batch_size, horizon, self._observation_dim)

        ret_dict = self.p_sample_loop(shape, cond, *args, **kwargs)

        return {self._diffusion_var: ret_dict}

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(
        self, x_start: torch.tensor, t: torch.tensor, noise: torch.tensor = None
    ) -> torch.tensor:
        """
        Generate a sample from the quadratic potential function.

        Args:
            x_start (torch.tensor): The starting point of the sample.
            t (torch.tensor): The time at which to evaluate the sample.
            noise (torch.tensor, optional): The noise to add to the sample. Defaults to None.

        Returns:
            torch.tensor: The generated sample.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self._sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self._sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def init_noise(
        self, shape: tuple, device: torch.device, cond: dict = None
    ) -> torch.tensor:
        """
        Initializes and returns noise tensor based on the given shape, device, and noise initialization method.

        Parameters:
            shape (tuple): The shape of the noise tensor.
            device (torch.device): The device where the noise tensor will be initialized.
            cond (dict, optional): A dictionary containing conditional values for noise initialization. Defaults to None.

        Returns:
            torch.tensor: The initialized noise tensor.

        Raises:
            NotImplementedError: If the noise initialization method is not supported.
        """
        if self._noise_init_method == "bounded_uniform":
            noise = torch.rand(shape, device=device)

            xyz = noise[..., :3]
            quat = noise[..., 3:]

            scene_size = (self._scene_bounds[1] - self._scene_bounds[0])[None, None]
            xyz = xyz * scene_size + self._scene_bounds[0][None, None]
            quat = (quat - 0.5) * 2

            return torch.cat([xyz, quat], dim=-1)
        elif self._noise_init_method == "normal":
            return torch.randn(shape, device=device)
        elif self._noise_init_method == "linear":
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

    def p_losses(
        self,
        x_start: torch.tensor,
        cond: dict,
        t: torch.tensor,
        robot: DiffRobot = None,
        **kwargs,
    ) -> tuple:
        """
        Computes the losses for the diffusion model.

        Args:
            x_start (torch.tensor): The original sample x_0. Shape: (batch_size, state_dim).
            cond (dict): The conditions for the system. Contains keys "start" and "end", each with shape (batch_size, condition_dim).
            t (torch.tensor): The time steps for the system. Shape: (batch_size,).
            robot (DiffRobot, optional): The robot object. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the loss and a dictionary of additional information.

        Raises:
            AssertionError: If the shapes of the noise and reconstructed states do not match.

        Examples:
            loss, info = p_losses(x_start, cond, t)
        """
        noise = self.init_noise(x_start.shape, x_start.device, cond)
        batch_size = x_start.size(0)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        conditions = kwargs
        if self._reverse_train:
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

        x_recon = self._model(x_noisy, cond, t, **conditions)

        if not self._predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, 0)

        assert noise.shape == x_recon.shape

        if self._predict_epsilon:
            loss, info = self._loss_fn(x_recon, noise)

        else:
            if self._diffusion_var == "gripper_poses":
                loss = F.mse_loss(x_start, x_recon) * self._trans_loss_scale
                info = {"pose_loss": loss}
            else:
                loss = F.mse_loss(x_start, x_recon)
                loss = loss * self._joint_loss_scale
                info = {"joint_loss": loss}

                if self._joint_pred_pose_loss:
                    assert robot is not None
                    predicted_poses = robot.forward_kinematics_batch(
                        x_recon.contiguous().view(-1, 7)
                    ).view(batch_size, -1, 7)
                    gt_poses = kwargs["gripper_poses"]

                    pose_loss = F.mse_loss(predicted_poses[..., :3], gt_poses[..., :3])
                    pose_loss = pose_loss * self._trans_loss_scale

                    loss += pose_loss
                    info["pose_loss"] = pose_loss

        if self._reverse_train:
            x_forward, x_backward = torch.split(x_recon, batch_size, dim=0)
            x_backward = torch.flip(x_backward, dims=(1,))
            if self._detach_reverse:
                x_backward = x_backward.detach()

            consistency_loss, _ = self.gripper_pose_loss(x_forward, x_backward)
            loss += consistency_loss
            info["consistency_loss"] = consistency_loss

        return loss, info

    def proc_cond(self, cond: dict) -> dict:
        """
        Process the given condition dictionary.

        Parameters:
            cond (dict): The condition dictionary to be processed.

        Returns:
            dict: The processed condition dictionary.
        """
        new_cond = {k: v for k, v in cond.items()}

        if -1 in new_cond and "end" not in self._conditions:
            del new_cond[-1]

        return new_cond

    def loss(self, x: torch.tensor, cond: dict, **kwargs) -> tuple:
        """
        Computes the loss for the given input tensor and condition dictionary.

        Args:
            x (torch.tensor): The input tensor.
            cond (dict): The condition dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the loss values.

        """
        cond = self.proc_cond(cond)
        batch_size = len(x)
        t = torch.randint(0, self._n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, **kwargs)

    def forward(self, cond: dict, *args, **kwargs) -> dict:
        """
        Forward the given `cond` dictionary along with any additional `args` and `kwargs` to the `conditional_sample` method and return the result.

        Parameters:
            cond (dict): A dictionary containing the condition to be passed to the `conditional_sample` method.
            *args: Additional positional arguments to be passed to the `conditional_sample` method.
            **kwargs: Additional keyword arguments to be passed to the `conditional_sample` method.

        Returns:
            dict: The result of the `conditional_sample` method.
        """
        return self.conditional_sample(cond=cond, *args, **kwargs)
