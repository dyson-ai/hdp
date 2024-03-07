"""Util functions for diffusion models."""

import logging
import math
import time
from typing import List, Optional

import numpy as np
import pytorch_kinematics as pk
import torch
import torch.nn.functional as F
from rlbench.demo import Demo

DTYPE = torch.float
DEVICE = "cuda"


class Timer:
    def __init__(self):
        self._start = time.time()

    def __call__(self, reset=True):
        now = time.time()
        diff = now - self._start
        if reset:
            self._start = now
        return diff


class Progress:
    def __init__(
        self,
        total,
        name="Progress",
        ncol=3,
        max_length=20,
        indent=0,
        line_width=100,
        speed_update_freq=100,
    ):
        self.total = total
        self.name = name
        self.ncol = ncol
        self.max_length = max_length
        self.indent = indent
        self.line_width = line_width
        self._speed_update_freq = speed_update_freq

        self._step = 0
        self._prev_line = "\033[F"
        self._clear_line = " " * self.line_width

        self._pbar_size = self.ncol * self.max_length
        self._complete_pbar = "#" * self._pbar_size
        self._incomplete_pbar = " " * self._pbar_size

        self.lines = [""]
        self.fraction = "{} / {}".format(0, self.total)

        self.resume()

    def update(self, description, n=1):
        self._step += n
        if self._step % self._speed_update_freq == 0:
            self._time0 = time.time()
            self._step0 = self._step
        self.set_description(description)

    def resume(self):
        self._skip_lines = 1
        print("\n", end="")
        self._time0 = time.time()
        self._step0 = self._step

    def pause(self):
        self._clear()
        self._skip_lines = 1

    def set_description(self, params=[]):
        if type(params) == dict:
            params = sorted([(key, val) for key, val in params.items()])

        ############
        # Position #
        ############
        self._clear()

        ###########
        # Percent #
        ###########
        percent, fraction = self._format_percent(self._step, self.total)
        self.fraction = fraction

        #########
        # Speed #
        #########
        speed = self._format_speed(self._step)

        ##########
        # Params #
        ##########
        num_params = len(params)
        nrow = math.ceil(num_params / self.ncol)
        params_split = self._chunk(params, self.ncol)
        params_string, lines = self._format(params_split)
        self.lines = lines

        description = "{} | {}{}".format(percent, speed, params_string)
        print(description)
        self._skip_lines = nrow + 1

    def append_description(self, descr):
        self.lines.append(descr)

    def _clear(self):
        position = self._prev_line * self._skip_lines
        empty = "\n".join([self._clear_line for _ in range(self._skip_lines)])
        print(position, end="")
        print(empty)
        print(position, end="")

    def _format_percent(self, n, total):
        if total:
            percent = n / float(total)

            complete_entries = int(percent * self._pbar_size)
            incomplete_entries = self._pbar_size - complete_entries

            pbar = (
                self._complete_pbar[:complete_entries]
                + self._incomplete_pbar[:incomplete_entries]
            )
            fraction = "{} / {}".format(n, total)
            string = "{} [{}] {:3d}%".format(fraction, pbar, int(percent * 100))
        else:
            fraction = "{}".format(n)
            string = "{} iterations".format(n)
        return string, fraction

    def _format_speed(self, n):
        num_steps = n - self._step0
        t = time.time() - self._time0
        speed = num_steps / t
        string = "{:.1f} Hz".format(speed)
        if num_steps > 0:
            self._speed = string
        return string

    def _chunk(self, l, n):
        return [l[i : i + n] for i in range(0, len(l), n)]

    def _format(self, chunks):
        lines = [self._format_chunk(chunk) for chunk in chunks]
        lines.insert(0, "")
        padding = "\n" + " " * self.indent
        string = padding.join(lines)
        return string, lines

    def _format_chunk(self, chunk):
        line = " | ".join([self._format_param(param) for param in chunk])
        return line

    def _format_param(self, param):
        k, v = param
        return "{} : {}".format(k, v)[: self.max_length]

    def stamp(self):
        if self.lines != [""]:
            params = " | ".join(self.lines)
            string = "[ {} ] {}{} | {}".format(
                self.name, self.fraction, params, self._speed
            )
            self._clear()
            print(string, end="\n")
            self._skip_lines = 1
        else:
            self._clear()
            self._skip_lines = 0

    def close(self):
        self.pause()


class Silent:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        return lambda *args: None


def to_np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def to_torch(x, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
        # import pdb; pdb.set_trace()
    return torch.tensor(x, dtype=dtype, device=device)


def to_device(x, device=DEVICE):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f"Unrecognized type in `to_device`: {type(x)}")


# def atleast_2d(x, axis=0):
# 	'''
# 		works for both np arrays and torch tensors
# 	'''
# 	while len(x.shape) < 2:
# 		shape = (1, *x.shape) if axis == 0 else (*x.shape, 1)
# 		x = x.reshape(*shape)
# 	return x

# def to_2d(x):
# 	dim = x.shape[-1]
# 	return x.reshape(-1, dim)


def batchify(batch, device):
    """
    convert a single dataset item to a batch suitable for passing to a model by
            1) converting np arrays to torch tensors and
            2) and ensuring that everything has a batch dimension
    """
    fn = lambda x: to_torch(x[None], device=device)

    batched_vals = []
    for field in batch._fields:
        val = getattr(batch, field)
        val = apply_dict(fn, val) if type(val) is dict else fn(val)
        batched_vals.append(val)
    return type(batch)(*batched_vals)


def apply_dict(fn, d, *args, **kwargs):
    return {k: fn(v, *args, **kwargs) for k, v in d.items()}


def normalize(x):
    """
    scales `x` to [0, 1]
    """
    x = x - x.min()
    x = x / x.max()
    return x


def to_img(x):
    normalized = normalize(x)
    array = to_np(normalized)
    array = np.transpose(array, (1, 2, 0))
    return (array * 255).astype(np.uint8)


def set_device(device):
    DEVICE = device
    if "cuda" in device:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)


def batch_to_device(batch, device="cuda:0"):
    return to_device(batch, device)


def _to_str(num):
    if num >= 1e6:
        return f"{(num/1e6):.2f} M"
    else:
        return f"{(num/1e3):.2f} k"


# -----------------------------------------------------------------------------#
# ----------------------------- parameter counting ----------------------------#
# -----------------------------------------------------------------------------#


def param_to_module(param):
    module_name = param[::-1].split(".", maxsplit=1)[-1][::-1]
    return module_name


def report_parameters(model, topk=10):
    counts = {k: p.numel() for k, p in model.named_parameters()}
    n_parameters = sum(counts.values())
    print(f"[ utils/arrays ] Total parameters: {_to_str(n_parameters)}")

    modules = dict(model.named_modules())
    sorted_keys = sorted(counts, key=lambda x: -counts[x])
    for i in range(topk):
        key = sorted_keys[i]
        count = counts[key]
        module = param_to_module(key)
        print(" " * 8, f"{key:10}: {_to_str(count)} | {modules[module]}")

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    print(
        " " * 8,
        f"... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters",
    )
    return n_parameters


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[i - 1].gripper_open
        and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped


def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)
    logging.debug("Found %d keypoints." % len(episode_keypoints), episode_keypoints)
    return episode_keypoints


def cumsum_traj(x, cond):
    x = torch.cat([cond[0].unsqueeze(1), x[:, 1:-1]], dim=1)
    x = torch.cumsum(x, dim=1)
    x = torch.cat([x, cond[-1].unsqueeze(1)], dim=1)

    return x


def concatenate_tensors_dict_list(dict_list, dim=0):
    concatenated_dict = {}

    # Iterate over keys in the first dictionary to initialize the keys in concatenated_dict
    for key in dict_list[0]:
        if isinstance(dict_list[0][key], torch.Tensor):
            concatenated_dict[key] = torch.cat([d[key] for d in dict_list], dim=dim)
        else:
            concatenated_dict[key] = [d[key] for d in dict_list]

    # Concatenate tensors for each key across dictionaries
    for key in concatenated_dict:
        if isinstance(concatenated_dict[key], list):
            concatenated_dict[key] = concatenate_tensors_dict_list(
                concatenated_dict[key], dim=dim
            )

    return concatenated_dict


def geodesic_distance_between_quaternions(
    q1: torch.tensor,
    q2: torch.tensor,
    acos_epsilon: Optional[float] = None,
) -> torch.tensor:
    """
    Given rows of quaternions q1 and q2, compute the geodesic distance between each
    """
    if len(q1.shape) == 3:
        q1 = q1.contiguous().view(-1, q1.size(-1))
    if len(q2.shape) == 3:
        q2 = q2.contiguous().view(-1, q2.size(-1))

    assert q1.shape[1] == 4, f"q1.shape[1] is {q1.shape[1]}, should be 4"
    assert q1.shape == q2.shape
    acos_clamp_epsilon = 1e-7
    if acos_epsilon is not None:
        acos_clamp_epsilon = acos_epsilon

    dot = torch.clip(
        torch.sum(q1 * q2, dim=1), -1 + acos_clamp_epsilon, 1 - acos_clamp_epsilon
    )
    distance = 2 * torch.acos(torch.abs(dot))

    # dot = torch.clip(torch.sum(q1 * q2, dim=1), -1, 1)
    # distance = 2 * torch.acos(
    #     torch.clamp(dot, -1 + acos_clamp_epsilon, 1 - acos_clamp_epsilon)
    # )
    # distance = torch.abs(torch.remainder(distance + torch.pi, 2 * torch.pi) - torch.pi)
    # assert distance.numel() == q1.shape[0], (
    #     f"Error, {distance.numel()} distance values calculated - should be {q1.shape[0]} (distance.shape ="
    #     f" {distance.shape})"
    # )
    return distance


def diff_kinematics_pose_to_coppelia(diff_k_pose, offset):
    shape = diff_k_pose.shape
    if len(diff_k_pose.shape) == 3:
        diff_k_pose = diff_k_pose.view(-1, 7)

    diff_k_pose = torch.cat(
        [
            diff_k_pose[:, :3] - offset[None, ...],
            diff_k_pose[:, 4:],
            diff_k_pose[:, 3:4],
        ],
        dim=-1,
    )

    return diff_k_pose.view(*shape)


def proc_quaternion(poses):
    trans, quat = poses[..., :3], poses[..., 3:]
    mask = quat[..., -1:] > 0
    quat = mask * quat + (~mask) * (-quat)

    if torch.is_tensor(quat):
        return torch.cat([trans, quat], dim=-1)
    else:
        return np.concatenate([trans, quat], axis=-1)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def rlb_pose_to_matrix(pose):
    bs = pose.size(0)
    matrix = torch.zeros([bs, 4, 4], dtype=pose.dtype, device=pose.device)
    xyz = pose[:, :3]
    quat = pose[:, 3:]
    quat = torch.cat([quat[:, 3:4], quat[:, :3]], dim=-1)
    rot_matrix = pk.quaternion_to_matrix(quat)
    matrix[:, :3, :3] = rot_matrix
    matrix[:, :3, 3] = xyz
    return matrix


def matrix_to_rlb_pose(matrix):
    rot_matrix = matrix[:, :3, :3]
    xyz = matrix[:, :3, 3]
    quat = pk.matrix_to_quaternion(rot_matrix)
    quat = torch.cat([quat[:, 1:4], quat[:, :1]], dim=-1)

    return torch.cat([xyz, quat], dim=-1)


def load_low_level_ckpt(path):
    state_dict = torch.load(path)
    del_keys = [k for k in state_dict if "loss_fn.weights" in k]
    for k in del_keys:
        del state_dict[k]

    return state_dict, del_keys


def load_checkpoint(diffusion, load_model_path, backbone):
    try:
        state_dict, del_keys = load_low_level_ckpt(load_model_path)

        if backbone == "unet":
            model_keys = diffusion.state_dict()
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

        diffusion.load_state_dict(state_dict, strict=False)
        print("model loaded")
    except Exception as e:
        print("load model failed")
        print(e)
