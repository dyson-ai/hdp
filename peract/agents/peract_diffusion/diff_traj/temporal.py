"""Temporal UNet to process trajectories."""

import einops
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.distributions import Bernoulli

from agents.peract_diffusion.diff_traj.helpers import (
    Conv1dBlock,
    Downsample1d,
    SinusoidalPosEmb,
    Upsample1d,
)
from agents.peract_diffusion.diff_traj.pointnet import PointNetfeat
from agents.peract_diffusion.diff_traj.resnet import ResnetEncoder


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=128):
        super().__init__()
        self._heads = heads
        hidden_dim = dim_head * heads
        self._to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self._to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self._to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self._heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self._heads, h=h, w=w
        )
        return self._to_out(out)


class GlobalMixing(nn.Module):
    def __init__(self, dim, heads=4, dim_head=128):
        super().__init__()
        self._heads = heads
        hidden_dim = dim_head * heads
        self._to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self._to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self._to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self._heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self._heads, h=h, w=w
        )
        return self._to_out(out)


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5, mish=True
    ):
        super().__init__()

        self._blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size, mish),
                Conv1dBlock(out_channels, out_channels, kernel_size, mish),
            ]
        )

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self._time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self._blocks[0](x) + self._time_mlp(t)
        out = self._blocks[1](out)

        return out + self.residual_conv(x)


class Temporal(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        scene_bounds,
        proprio_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        condition_dropout=0.1,
        calc_energy=False,
        kernel_size=5,
        conditions=[],
        hard_conditions=[],
        rank_bin=10,
        num_encoder_layers=4,
        num_decoder_layers=4,
        n_head=4,
        causal_attn=True,
        backbone="unet",
        depth_proc="pointnet",
        rgb_encoder="resnet18",
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        self._depth_proc = depth_proc

        if calc_energy:
            mish = False
            act_fn = nn.SiLU()
        else:
            mish = True
            act_fn = nn.Mish()

        self._time_dim = dim
        self._returns_dim = dim
        scene_bounds = torch.tensor(
            scene_bounds,
            dtype=torch.float32,
        )
        self._proprio_dim = proprio_dim
        self._rank_bins = rank_bin

        self.register_buffer("_scene_bounds", scene_bounds)

        self._time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self._conditions = conditions
        self._hard_conditions = hard_conditions

        self._condition_dropout = condition_dropout
        self._calc_energy = calc_energy

        condition_fns = {}
        mlp_shape_mapping = {
            "return": 1,
            "start": 7,
            "end": 7,
            "proprios": self._proprio_dim,
            "rank": self._rank_bins,
        }

        embed_dim = dim
        for cond_name in self._conditions:
            if cond_name == "pcds":
                if depth_proc == "pointnet":
                    condition_fns[cond_name] = nn.Sequential(
                        PointNetfeat(global_feat=True),
                        nn.Mish(),
                        nn.Linear(1024, 256),
                        nn.Mish(),
                        nn.Linear(256, dim),
                    )
                elif depth_proc == "resnet":
                    resnet = ResnetEncoder(
                        input_shape=(4, 64, 64), pretrained=False, freeze=False
                    )
                    condition_fns[cond_name] = nn.Sequential(
                        resnet,
                        nn.Mish(),
                        nn.Linear(resnet.n_channel, dim),
                    )
                else:
                    raise NotImplementedError
            elif cond_name == "rgbs":
                resnet = ResnetEncoder(
                    rgb=True, freeze=False, pretrained=True, model=rgb_encoder
                )
                condition_fns[cond_name] = nn.Sequential(
                    resnet,
                    nn.Mish(),
                    nn.Linear(resnet.n_channel, dim),
                )
            else:
                condition_fns[cond_name] = nn.Sequential(
                    nn.Linear(mlp_shape_mapping[cond_name], dim),
                    act_fn,
                    nn.Linear(dim, dim * 4),
                    act_fn,
                    nn.Linear(dim * 4, dim),
                )
        self._cond_fns = nn.ModuleDict(condition_fns)
        embed_dim += len(self._conditions) * dim

        self._mask_dist = Bernoulli(probs=1 - self._condition_dropout)

        if backbone == "unet":
            self._backbone = UnetBackbone(
                horizon=horizon,
                transition_dim=transition_dim,
                cond_dim=len(self._conditions) * dim,
                dim=dim,
                dim_mults=dim_mults,
                kernel_size=kernel_size,
                mish=mish,
            )
        elif backbone == "transformer":
            self._backbone = TransformerBackbone(
                horizon=horizon,
                transition_dim=transition_dim,
                cond_dim=embed_dim,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                n_head=n_head,
                dim=dim,
                causal_attn=causal_attn,
            )
        else:
            raise NotImplementedError

    def forward(
        self,
        x,
        cond,
        time,
        use_dropout=True,
        force_dropout=False,
        **kwargs,
    ):
        """
        x : [ batch x horizon x transition ]
        returns : [batch x horizon]
        """
        if self._calc_energy:
            x_inp = x

        t = self._time_mlp(time)

        for cond_name in sorted(self._conditions):
            cond_var = kwargs[cond_name]

            if (
                cond_name == "pcds"
                and self._depth_proc == "resnet"
                or cond_name == "rgbs"
            ):
                cond_var = cond_var.permute(0, 3, 1, 2)

            cond_emb = self._cond_fns[cond_name](cond_var)
            allow_dropout = cond_name not in self._hard_conditions
            if use_dropout and allow_dropout:
                mask = self._mask_dist.sample(sample_shape=(cond_emb.size(0), 1)).to(
                    cond_emb.device
                )
                cond_emb = mask * cond_emb
            if force_dropout and allow_dropout:
                cond_emb = 0 * cond_emb

            t = torch.cat([t, cond_emb], dim=-1)

        x = self._backbone(x, t)

        if self._calc_energy:
            # Energy function
            energy = ((x - x_inp) ** 2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x_inp, create_graph=True)
            return grad[0]
        else:
            return x


class UnetBackbone(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        kernel_size=5,
        mish=False,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        embed_dim = dim
        embed_dim += cond_dim

        self._downs = nn.ModuleList([])
        self._ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self._downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in,
                            dim_out,
                            embed_dim=embed_dim,
                            horizon=horizon,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            embed_dim=embed_dim,
                            horizon=horizon,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self._mid_block1 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=embed_dim,
            horizon=horizon,
            kernel_size=kernel_size,
            mish=mish,
        )
        self._mid_block2 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=embed_dim,
            horizon=horizon,
            kernel_size=kernel_size,
            mish=mish,
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self._ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2,
                            dim_in,
                            embed_dim=embed_dim,
                            horizon=horizon,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        ResidualTemporalBlock(
                            dim_in,
                            dim_in,
                            embed_dim=embed_dim,
                            horizon=horizon,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon * 2

        self._final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, t):
        """
        x : [ batch x horizon x transition ]
        returns : [batch x horizon]
        """
        x = einops.rearrange(x, "b h t -> b t h")
        h = []

        for resnet, resnet2, downsample in self._downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self._mid_block1(x, t)
        x = self._mid_block2(x, t)

        for resnet, resnet2, upsample in self._ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self._final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")

        return x


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        num_encoder_layers,
        num_decoder_layers,
        n_head=4,
        dim=128,
        dropout_rate=0.1,
        causal_attn=False,
    ):
        super().__init__()

        self._obs_emb = nn.Sequential(
            nn.Linear(transition_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim + cond_dim,
            dim_feedforward=2 * dim,
            nhead=n_head,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self._encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim + cond_dim,
            dim_feedforward=2 * dim,
            nhead=n_head,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self._decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
        )

        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = horizon
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        self._final = nn.Sequential(
            nn.Linear(dim + cond_dim, transition_dim),
            nn.GELU(),
            nn.Linear(transition_dim, transition_dim),
        )

    def forward(self, x, t):
        """
        x : [ batch x horizon x transition ]
        returns : [batch x horizon]
        """
        x = self._obs_emb(x)
        x = torch.cat([x, t.unsqueeze(1).repeat(1, x.size(1), 1)], dim=-1)

        memory = self._encoder(x)
        x = self._decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.mask,
            memory_mask=None,
        )
        x = self._final(x)

        return x
