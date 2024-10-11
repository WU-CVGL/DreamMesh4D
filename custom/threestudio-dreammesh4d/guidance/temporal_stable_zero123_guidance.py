import importlib
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from tqdm import tqdm

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *

import clip
from extern.ldm_zero123.modules.diffusionmodules.util import GroupNorm32

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# load model
def load_model_from_config(config, ckpt, device, vram_O=True, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")

    if "global_step" in pl_sd and verbose:
        print(f'[INFO] Global Step: {pl_sd["global_step"]}')

    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("[INFO] missing keys: \n", m)
    if len(u) > 0 and verbose:
        print("[INFO] unexpected keys: \n", u)

    # manually load ema and delete it to save GPU memory
    if model.use_ema:
        if verbose:
            print("[INFO] loading EMA...")
        model.model_ema.copy_to(model.model)
        del model.model_ema

    if vram_O:
        # we don't need decoder
        del model.first_stage_model.decoder

    torch.cuda.empty_cache()

    model.eval().to(device)

    return model


@threestudio.register("temporal-stable-zero123-guidance")
class TemporalStableZero123Guidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "load/zero123/stable-zero123.ckpt"
        pretrained_config: str = "load/zero123/sd-objaverse-finetune-c_concat-256.yaml"
        vram_O: bool = True

        num_frames: int = 14
        cond_video_dir: str = "load/videos/anya"
        cond_elevation_deg: float = 0.0
        cond_azimuth_deg: float = 0.0
        cond_camera_distance: float = 1.2

        guidance_scale: float = 5.0

        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        chunk_size: Optional[int] = None

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Zero123 ...")

        self.config = OmegaConf.load(self.cfg.pretrained_config)
        # TODO: seems it cannot load into fp16...
        self.weights_dtype = torch.float16 if self.cfg.half_precision_weights else torch.float32
        
        self.model = load_model_from_config(
            self.config,
            self.cfg.pretrained_model_name_or_path,
            device=self.device,
            vram_O=self.cfg.vram_O,
        )
        self.model.to(dtype=self.weights_dtype)
        # self.model.cond_stage_model.model.visual.ln_pre.to(dtype=torch.float32)
        def recursive_layernorm_fp32(model):
            for attr in model.__dir__():
                if attr.startswith("_"):
                    continue
                try:
                    module = getattr(model, attr)
                except:
                    continue  # ignore attributes like property, which can't be retrived using getattr?
                if isinstance(module, clip.model.LayerNorm):
                    module.to(dtype=torch.float32)
                    
                elif isinstance(module, torch.nn.Sequential) or isinstance(module, torch.nn.ModuleList):
                    for sub_module in module:
                        recursive_layernorm_fp32(sub_module)
                elif isinstance(module, torch.nn.Module):
                    recursive_layernorm_fp32(module)
        recursive_layernorm_fp32(self.model)

        for p in self.model.parameters():
            p.requires_grad_(False)

        # timesteps: use diffuser for convenience... hope it's alright.
        self.num_train_timesteps = self.config.model.params.timesteps

        self.scheduler = DDIMScheduler(
            self.num_train_timesteps,
            self.config.model.params.linear_start,
            self.config.model.params.linear_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        self.num_frames = self.cfg.num_frames

        self.prepare_embeddings_video(self.cfg.cond_video_dir)

        threestudio.info(f"Loaded Stable Zero123!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def prepare_embeddings(self, image_path: str) -> None:
        # load cond image for zero123
        assert os.path.exists(image_path)
        rgba = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        rgba = (
            cv2.resize(rgba, (256, 256), interpolation=cv2.INTER_AREA).astype(
                np.float32
            )
            / 255.0
        )
        rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
        rgb_256: Float[Tensor, "1 3 H W"] = (
            torch.from_numpy(rgb)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(self.device)
        )
        c_crossattn, c_concat = self.get_img_embeds(rgb_256)
        return (rgb_256, c_crossattn, c_concat)

    @torch.cuda.amp.autocast(enabled=False)
    def prepare_embeddings_video(self, video_dir: str) -> None:
        assert os.path.exists(video_dir)
        rgb_256 = []
        c_crossattn = []
        c_concat = []
        for i in range(self.num_frames):
            image_path = os.path.join(video_dir, f"{i:03}_rgba.png")
            if not os.path.exists(image_path):
                image_path = os.path.join(video_dir, f"{i}.png")
            outs = self.prepare_embeddings(image_path)
            rgb_256.append(outs[0])
            c_crossattn.append(outs[1])
            c_concat.append(outs[2])
        
        self.rgb_256 = torch.cat(rgb_256, dim=0)
        self.c_crossattn = torch.cat(c_crossattn, dim=0)
        self.c_concat = torch.cat(c_concat, dim=0)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_img_embeds(
        self,
        img: Float[Tensor, "B 3 256 256"],
    ) -> Tuple[Float[Tensor, "B 1 768"], Float[Tensor, "B 4 32 32"]]:
        img = img * 2.0 - 1.0
        c_crossattn = self.model.get_learned_conditioning(img.to(self.weights_dtype)).to(self.weights_dtype)
        c_concat = self.model.encode_first_stage(img.to(self.weights_dtype)).mode()
        return c_crossattn, c_concat

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs.to(self.weights_dtype))
        )
        return latents.to(input_dtype)  # [B, 4, 32, 32] Latent space image

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        image = self.model.decode_first_stage(latents)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_cond(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        frame_indices: Int[Tensor, "B"],
        c_crossattn=None,
        c_concat=None,
        **kwargs,
    ) -> dict:
        T = torch.stack(
            [
                torch.deg2rad(
                    (90 - elevation) - (90 - self.cfg.cond_elevation_deg)
                ),  # Zero123 polar is 90-elevation
                torch.sin(torch.deg2rad(azimuth - self.cfg.cond_azimuth_deg)),
                torch.cos(torch.deg2rad(azimuth - self.cfg.cond_azimuth_deg)),
                torch.deg2rad(
                    90 - torch.full_like(elevation, self.cfg.cond_elevation_deg)
                ),
            ],
            dim=-1,
        )[:, None, :].to(self.device, dtype=self.weights_dtype)
        cond = {}
        clip_emb = self.model.cc_projection(
            torch.cat(
                [
                    (self.c_crossattn if c_crossattn is None else c_crossattn)[frame_indices],
                    T,
                ],
                dim=-1,
            )
        )
        cond["c_crossattn"] = [
            torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)
        ]
        cond["c_concat"] = [
            torch.cat(
                [
                    torch.zeros(
                        len(T), *self.c_concat.shape[1:], dtype=self.c_concat.dtype, device=self.device
                    ),
                    (self.c_concat if c_concat is None else c_concat)[frame_indices],
                ],
                dim=0,
            )
        ]
        return cond

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        frame_indices: Int[Tensor, "B"],
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = (
                F.interpolate(rgb_BCHW, (32, 32), mode="bilinear", align_corners=False)
                * 2
                - 1
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (256, 256), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        cond = self.get_cond(elevation, azimuth, camera_distances, frame_indices)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)

            noise_pred = self.model.apply_model(x_in.to(self.weights_dtype), t_in, cond)

        # perform guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        w = (1 - self.alphas[t]).reshape(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        return guidance_out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
