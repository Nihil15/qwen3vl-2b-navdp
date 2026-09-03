"""Standalone NavDP policy (System-1 of InternVLA-N1) — no Qwen VLM required.

Architecture clone of internnav's `NavDP_Policy_DPT_CriticSum_DAT`
(third_party/InternNav/internnav/model/basemodel/internvla_n1/navdp.py) with
the `diffusion_policy` dependency replaced by the local SinusoidalPosEmb and
extra inference heads exposed:

  * predict_pointgoal(goal_point, images, depths)  -> ranked trajectories
  * predict_pixelgoal(goal_pixel, images, depths)  -> ranked trajectories
  * predict_nogoal(images, depths)                 -> ranked trajectories

Weights come from checkpoints/navdp_extracted.pth (extracted `model.navdp.*`
tensors of InternVLA-N1-w-NavDP; see scripts/extract_navdp_weights.py).

Input conventions (canonical NavDP):
  images: (B, memory_size, 224, 224, 3) float, RGB / 255.0
  depths: (B, memory_size, 224, 224, 1) float, meters; invalid/inf -> 0,
          values < 0.1 or > 5.0 zeroed
  goal_point: (B, 3) [x, y, z] in robot frame (x forward, y left), meters
  goal_pixel: (B, 2) pixel goal in the current image, normalized to [0, 1]

Output: trajectories (K, predict_size, 3) of cumulative [x, y, yaw] waypoints
in the robot frame (cumsum of per-step deltas / 4), ranked by the critic.
"""

import os

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .navdp_backbone import DAT_RGBD_Patch_Backbone, SinusoidalPosEmb, TokenCompressor

_DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "navdp_extracted.pth"
)
_DEFAULT_DA_CKPT = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "depth_anything_v2_vits.pth"
)


class NavDPStandalone(nn.Module):
    def __init__(
        self,
        image_size=224,
        memory_size=2,
        predict_size=32,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        vlm_token_dim=3584,
        dropout=0.1,
        input_dtype="bf16",
        navdp_version=0.1,
        da_checkpoint=_DEFAULT_DA_CKPT,
        device="cuda:0",
    ):
        super().__init__()
        self.image_size = image_size
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.token_dim = token_dim
        self.vlm_token_dim = vlm_token_dim
        self.dropout = dropout
        self.input_dtype = torch.bfloat16 if input_dtype == "bf16" else torch.float32
        self._device = torch.device(device)

        self.rgbd_encoder = DAT_RGBD_Patch_Backbone(
            image_size,
            token_dim,
            memory_size=memory_size,
            finetune=False,
            checkpoint=da_checkpoint,
            input_dtype=input_dtype,
            version=navdp_version,
        )
        self.point_encoder = nn.Linear(3, token_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=temporal_depth)

        self.input_embed = nn.Linear(3, token_dim)
        self.cond_pos_embed = nn.Parameter(
            torch.zeros((1, memory_size * 16 + 2, token_dim), dtype=self.input_dtype)
        )
        self.out_pos_embed = nn.Parameter(torch.zeros((1, predict_size, token_dim), dtype=self.input_dtype))
        self.drop = nn.Dropout(dropout)
        self.time_emb = SinusoidalPosEmb(token_dim)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=20, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon"
        )

        self.layernorm = nn.LayerNorm(token_dim)
        self.action_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)

        self.tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = (
            self.tgt_mask.float()
            .masked_fill(self.tgt_mask == 0, float("-inf"))
            .masked_fill(self.tgt_mask == 1, float(0.0))
        ).to(dtype=self.input_dtype)

        self.cond_critic_mask = torch.zeros((predict_size, 2 + memory_size * 16))
        self.cond_critic_mask[:, 0:2] = float("-inf")
        self.cond_critic_mask = self.cond_critic_mask.to(dtype=self.input_dtype)

        self.vlm_embed_mlp = nn.Sequential(
            nn.Linear(vlm_token_dim, vlm_token_dim // 4),
            nn.ReLU(),
            nn.Linear(vlm_token_dim // 4, vlm_token_dim // 8),
            nn.ReLU(),
            nn.Linear(vlm_token_dim // 8, token_dim),
        )
        self.goal_compressor = TokenCompressor(token_dim, 8, 1)

        self.pg_embed_mlp = nn.Sequential(
            nn.Linear(2, token_dim // 2), nn.ReLU(), nn.Linear(token_dim // 2, token_dim)
        )
        self.pg_pred_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(),
            nn.Linear(token_dim // 2, token_dim // 4),
            nn.ReLU(),
            nn.Linear(token_dim // 4, 2),
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, weights_path=_DEFAULT_WEIGHTS, device="cuda:0", **kwargs):
        model = cls(device=device, **kwargs)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        real_missing = [k for k in missing if not k.startswith("rgbd_encoder.rgb_model.")]
        if real_missing or unexpected:
            print(f"[NavDP] missing={real_missing[:8]} unexpected={unexpected[:8]}")
        model = model.to(device=torch.device(device), dtype=model.input_dtype)
        model.tgt_mask = model.tgt_mask.to(device)
        model.cond_critic_mask = model.cond_critic_mask.to(device)
        model.eval()
        return model

    # ------------------------------------------------------------------ #
    def _predict_noise(self, last_actions, timestep, goal_embed, rgbd_embed):
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep.to(self._device)).unsqueeze(1).to(dtype=last_actions.dtype)
        cond_embedding = (
            torch.cat([time_embeds, goal_embed, rgbd_embed], dim=1)
            + self.cond_pos_embed[:, : self.memory_size * 16 + 2, :]
        )
        cond_embedding = cond_embedding.repeat(action_embeds.shape[0], 1, 1)
        input_embedding = action_embeds + self.out_pos_embed[:, : self.predict_size, :]
        output = self.decoder(tgt=input_embedding, memory=cond_embedding, tgt_mask=self.tgt_mask)
        output = self.layernorm(output)
        return self.action_head(output)

    def _predict_critic(self, trajectories, rgbd_embed):
        repeat_rgbd = rgbd_embed.repeat(trajectories.shape[0], 1, 1)
        nogoal = torch.zeros_like(repeat_rgbd[:, 0:1])
        act = self.input_embed(trajectories)
        act = act + self.out_pos_embed[:, : self.predict_size, :]
        cond = torch.cat([nogoal, nogoal, repeat_rgbd], dim=1) + self.cond_pos_embed[
            :, : self.memory_size * 16 + 2, :
        ]
        out = self.decoder(tgt=act, memory=cond, memory_mask=self.cond_critic_mask)
        out = self.layernorm(out)
        return self.critic_head(out.mean(dim=1))[:, 0]

    def _denoise(self, goal_embed, rgbd_embed, sample_num):
        naction = torch.randn(
            (sample_num, self.predict_size, 3), dtype=goal_embed.dtype, device=self._device
        )
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        for k in self.noise_scheduler.timesteps:
            noise_pred = self._predict_noise(naction, k.unsqueeze(0), goal_embed, rgbd_embed)
            naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
        return naction

    def _rank(self, naction, rgbd_embed, topk=8):
        critic = self._predict_critic(naction, rgbd_embed).float()
        trajs = torch.cumsum(naction.float() / 4.0, dim=1)
        best = trajs[(-critic).argsort()[:topk]]
        worst = trajs[critic.argsort()[:topk]]
        return best, worst, critic

    @torch.no_grad()
    def sample_pointgoal(self, goal_point, images, depths, sample_num=32):
        """Point-goal sampling that returns ALL trajectories + critic scores.

        goal_point uses the checkpoint's native convention: [x fwd, y RIGHT, z].
        Returns (trajs (N, predict_size, 3) float32 cumulative waypoints,
                 critic (N,) float32).
        """
        goal = torch.as_tensor(goal_point, dtype=self.input_dtype, device=self._device).reshape(1, 3)
        rgbd_embed = self.rgbd_encoder(images.to(self._device), depths.to(self._device))
        goal_embed = self.point_encoder(goal).unsqueeze(1)
        naction = self._denoise(goal_embed, rgbd_embed, sample_num)
        critic = self._predict_critic(naction, rgbd_embed).float()
        trajs = torch.cumsum(naction.float() / 4.0, dim=1)
        return trajs, critic

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_pointgoal(self, goal_point, images, depths, sample_num=32, topk=8):
        """goal_point: (1,3) [x fwd, y left, z] meters, robot frame."""
        goal = torch.as_tensor(goal_point, dtype=self.input_dtype, device=self._device).reshape(1, 3)
        rgbd_embed = self.rgbd_encoder(images.to(self._device), depths.to(self._device))
        goal_embed = self.point_encoder(goal).unsqueeze(1)
        naction = self._denoise(goal_embed, rgbd_embed, sample_num)
        return self._rank(naction, rgbd_embed, topk)

    @torch.no_grad()
    def predict_pixelgoal(self, goal_pixel, images, depths, sample_num=32, topk=8):
        """goal_pixel: (1,2) normalized [x, y] in [0,1] of current image."""
        goal = torch.as_tensor(goal_pixel, dtype=self.input_dtype, device=self._device).reshape(1, 2)
        rgbd_embed = self.rgbd_encoder(images.to(self._device), depths.to(self._device))
        goal_embed = self.pg_embed_mlp(goal).unsqueeze(1)
        naction = self._denoise(goal_embed, rgbd_embed, sample_num)
        return self._rank(naction, rgbd_embed, topk)

    @torch.no_grad()
    def predict_nogoal(self, images, depths, sample_num=32, topk=8):
        rgbd_embed = self.rgbd_encoder(images.to(self._device), depths.to(self._device))
        goal_embed = torch.zeros_like(rgbd_embed[:, 0:1])
        naction = self._denoise(goal_embed, rgbd_embed, sample_num)
        return self._rank(naction, rgbd_embed, topk)


def trajectory_to_vw(trajectory, kp=1.0, max_lin=0.5, max_ang=0.5):
    """Last waypoint of a ranked trajectory -> (linear, angular) velocity."""
    import numpy as np

    subgoal = trajectory[-1]
    linear = float(np.clip(kp * np.linalg.norm(subgoal[:2]), 0.0, max_lin))
    angular = float(np.clip(kp * subgoal[2], -max_ang, max_ang))
    return linear, angular
