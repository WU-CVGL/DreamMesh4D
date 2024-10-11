import math
from dataclasses import dataclass

import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *

from .gaussian_batch_renderer import GaussianBatchRenderer
from ..geometry.dynamic_sugar import DynamicSuGaRModel
import time

def basicfunction(x):
    return torch.exp(-1*x.pow(2))

class Depth2Normal(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delzdelxkernel = torch.tensor(
            [
                [0.00000, 0.00000, 0.00000],
                [-1.00000, 0.00000, 1.00000],
                [0.00000, 0.00000, 0.00000],
            ]
        )
        self.delzdelykernel = torch.tensor(
            [
                [0.00000, -1.00000, 0.00000],
                [0.00000, 0.00000, 0.00000],
                [0.0000, 1.00000, 0.00000],
            ]
        )

    def forward(self, x):
        B, C, H, W = x.shape
        delzdelxkernel = self.delzdelxkernel.view(1, 1, 3, 3).to(x.device)
        delzdelx = F.conv2d(
            x.reshape(B * C, 1, H, W), delzdelxkernel, padding=1
        ).reshape(B, C, H, W)
        delzdelykernel = self.delzdelykernel.view(1, 1, 3, 3).to(x.device)
        delzdely = F.conv2d(
            x.reshape(B * C, 1, H, W), delzdelykernel, padding=1
        ).reshape(B, C, H, W)
        normal = -torch.cross(delzdelx, delzdely, dim=1)
        return normal

@threestudio.register("diff-sugar-rasterizer-temporal")
class DiffGaussian(Rasterizer, GaussianBatchRenderer):
    @dataclass
    class Config(Rasterizer.Config):
        debug: bool = False
        invert_bg_prob: float = 1.0
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)

    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        threestudio.info(
            "[Note] Gaussian Splatting doesn't support material and background now."
        )
        super().configure(geometry, material, background)
        self.normal_module = Depth2Normal()
        self.background_tensor = torch.tensor(
            self.cfg.back_ground_color, dtype=torch.float32, device="cuda"
        )

    def forward(
        self,
        viewpoint_camera,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
        compute_normal_from_dist=True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        if self.training:
            # debug
            invert_bg_color = False
            # invert_bg_color = np.random.rand() > self.cfg.invert_bg_prob
        else:
            invert_bg_color = True

        bg_color = bg_color if not invert_bg_color else (1.0 - bg_color)

        pc: DynamicSuGaRModel = self.geometry
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        pts = pc.get_xyz
        screenspace_points = (
            torch.zeros_like(
                pts, dtype=pts.dtype, requires_grad=True, device="cuda"
            )
            + 0
        )
        # pointtimes = (
        #     torch.ones(
        #         (pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda"
        #     )
        #     + 0
        # )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # means3D = pc.get_xyz
        means2D = screenspace_points

        static = False
        if viewpoint_camera.timestamp is None and viewpoint_camera.frame_idx is None:
            static = True
            means3D = pc.get_xyz
            opacity = pc.get_opacity
            scales = pc.get_scaling
            rotations = pc.get_rotation
            shs = pc.get_features
            colors_precomp = None
        else:
            # debug time
            # start_time = time.time_ns()
            means3D, scales, rotations, opacity, colors_precomp = pc.get_timed_gs_all_single_time(viewpoint_camera.timestamp, viewpoint_camera.frame_idx)
            # print(f"{time.time_ns() - start_time} ns")
            shs = None

        # print(means3D.shape)
        cov3D_precomp = None

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        mask = rendered_alpha > 0.99
        rendered_depth[~mask] = rendered_depth[~mask].detach()

        if compute_normal_from_dist and not static:
            batch_idx = kwargs["batch_idx"]
            rays_d = kwargs["rays_d"][batch_idx]
            rays_o = kwargs["rays_o"][batch_idx]
            xyz_map = rays_o + rendered_depth.permute(1, 2, 0) * rays_d
            normal_from_dist = self.normal_module(xyz_map.permute(2, 0, 1).unsqueeze(0))[0]
            normal_from_dist = F.normalize(normal_from_dist, dim=0)
            normal_map_from_dist = normal_from_dist * 0.5 * rendered_alpha + 0.5
            normal_mask = mask.repeat(3, 1, 1)
            normal_from_dist[~normal_mask] = normal_from_dist[~normal_mask].detach()
            normal_map_from_dist[~normal_mask] = normal_map_from_dist[~normal_mask].detach()
        else:
            normal_from_dist = None
            normal_map_from_dist = None

        if not static:
            point_normals = pc.get_timed_gs_normals(
                viewpoint_camera.timestamp[None], viewpoint_camera.frame_idx[None]
            )[0]
            normal, _, _, _ = rasterizer(
                means3D=means3D,
                means2D=torch.zeros_like(means2D),
                shs=None,
                colors_precomp=point_normals,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
            normal = F.normalize(normal, dim=0)

            normal_map = normal * 0.5 * rendered_alpha + 0.5
            normal_mask = mask.repeat(3, 1, 1)
            normal[~normal_mask] = normal[~normal_mask].detach()
            normal_map[~normal_mask] = normal_map[~normal_mask].detach()
        else:
            normal = None
            normal_map = None

        # Retain gradients of the 2D (screen-space) means for batch dim
        if self.training:
            screenspace_points.retain_grad()

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image.clamp(0, 1),
            "normal": normal_map,
            "normal_from_dist": normal_map_from_dist,
            "depth": rendered_depth,
            "mask": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "raw_normal": normal,
            "raw_normal_from_dist": normal_from_dist,
        }
