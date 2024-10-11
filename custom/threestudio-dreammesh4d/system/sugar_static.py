import os
import random
from easydict import EasyDict
from dataclasses import dataclass, field

import numpy as np
import threestudio
import torch
import torch.nn.functional as F

from threestudio.systems.utils import parse_optimizer
from threestudio.utils.loss import tv_loss
from threestudio.utils.typing import *


from torchmetrics import PearsonCorrCoef
from pytorch3d.loss import mesh_normal_consistency, mesh_laplacian_smoothing
from .base import BaseSuGaRSystem
from ..geometry.gaussian_base import BasicPointCloud
from ..geometry.sugar import SuGaRModel
from ..utils.sugar_utils import SuGaRRegularizer


@threestudio.register("sugar-static-system")
class SuGaRStaticSystem(BaseSuGaRSystem):
    @dataclass
    class Config(BaseSuGaRSystem.Config):
        stage: str = "gaussian"
        freq: dict = field(default_factory=dict)
        ambient_ratio_min: float = 0.5
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)

        # ==== SuGaR regularization configs for Gaussian stage ==== #
        use_sugar_reg: bool = True
        knn_to_track: int = 16
        n_samples_for_sugar_sdf_reg: int = 500000
        # min_opac_prune: Any = 0.5

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.stage = self.cfg.stage
        if self.stage == "gaussian":
            self.automatic_optimization = False


    def configure_optimizers(self):
        optim = self.geometry.optimizer
        if hasattr(self, "merged_optimizer"):
            return [optim]
        
        if hasattr(self.cfg.optimizer, "name"):
            net_optim = parse_optimizer(self.cfg.optimizer, self)
            optim = self.geometry.merge_optimizer(net_optim)
            self.merged_optimizer = True
        else:
            self.merged_optimizer = False
        return [optim]

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._render_type = "rgb"
        self.sugar_reg = None
        self.pearson = PearsonCorrCoef().to(self.device)

        # Zero123
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def on_load_checkpoint(self, checkpoint):
        if self.stage == "gaussian":
            num_pts = checkpoint["state_dict"]["geometry._xyz"].shape[0]
            pcd = BasicPointCloud(
                points=np.zeros((num_pts, 3)),
                colors=np.zeros((num_pts, 3)),
                normals=np.zeros((num_pts, 3)),
            )
            self.geometry.create_from_pcd(pcd, 10)
            self.geometry.training_setup()
            # return
            super().on_load_checkpoint(checkpoint)

    def forward(
        self,
        batch: Dict[str, Any],
        compute_color_in_rasterizer: bool = False
    ) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)
        if self.stage == "sugar" and not compute_color_in_rasterizer:
            self.geometry: SuGaRModel
            batch.update(
                {"override_color": self.geometry.get_points_rgb()}
            )
        outputs = self.renderer.batch_forward(batch)
        return outputs

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        super().on_train_batch_start(batch, batch_idx, unused)
        if self.stage == "gaussain" and self.cfg.use_sugar_reg and self.global_step >= self.cfg.freq.milestone_sugar_reg:
            self.sugar_reg = SuGaRRegularizer(
                self.geometry, keep_track_of_knn=True, knn_to_track=self.cfg.knn_to_track
            )
            self.sugar_reg.reset_neighbors(self.cfg.knn_to_track)

        if self.sugar_reg is not None:
            if (
                self.global_step % self.cfg.freq.reset_neighbors == 0
                or self.geometry.pruned_or_densified
            ):
                self.sugar_reg.reset_neighbors(self.cfg.knn_to_track)

            # self.geometry.min_opac_prune = self.C(self.cfg.min_opac_prune)

    def training_substep(self, batch, batch_idx, guidance: str):
        """
        Args:
            guidance: one of "ref" (reference image supervision), "rand"
        """
        if guidance == "ref":
            ambient_ratio = 1.0
            shading = "diffuse"
            batch["shading"] = shading
        elif guidance == "rand":
            batch = batch["random_camera"]
            ambient_ratio = (
                self.cfg.ambient_ratio_min
                + (1 - self.cfg.ambient_ratio_min) * random.random()
            )

        batch["ambient_ratio"] = ambient_ratio

        out = self(batch)

        loss_prefix = f"loss_{guidance}_"

        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        guidance_eval = (
            guidance == "rand"
            and self.cfg.freq.guidance_eval > 0
            and self.true_global_step % self.cfg.freq.guidance_eval == 0
        )

        if guidance == "ref":
            gt_mask = batch["mask"]
            gt_rgb = batch["rgb"]

            # color loss
            gt_rgb = gt_rgb * gt_mask.float()
            set_loss("rgb", F.mse_loss(gt_rgb, out["comp_rgb"] * gt_mask.float()))

            # mask loss
            set_loss("mask", F.mse_loss(gt_mask.float(), out["comp_mask"]))

            # depth loss
            if self.C(self.cfg.loss.lambda_depth) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)].unsqueeze(1)
                valid_pred_depth = out["comp_depth"][gt_mask].unsqueeze(1)
                with torch.no_grad():
                    A = torch.cat(
                        [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                    )  # [B, 2]
                    X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                    valid_gt_depth = A @ X  # [B, 1]
                set_loss("depth", F.mse_loss(valid_gt_depth, valid_pred_depth))

            # relative depth loss
            if self.C(self.cfg.loss.lambda_depth_rel) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)]  # [B,]
                valid_pred_depth = out["comp_depth"][gt_mask]  # [B,]
                set_loss(
                    "depth_rel", 1 - self.pearson(valid_pred_depth, valid_gt_depth)
                )

            # normal loss
            if self.C(self.cfg.loss.lambda_normal) > 0:
                valid_gt_normal = (
                    1 - 2 * batch["ref_normal"][gt_mask.squeeze(-1)]
                )  # [B, 3]
                valid_pred_normal = (
                    2 * out["comp_normal"][gt_mask.squeeze(-1)] - 1
                )  # [B, 3]
                set_loss(
                    "normal",
                    1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean(),
                )

        elif guidance == "rand":
            # zero123
            guidance_out = self.guidance(
                out["comp_rgb"],
                **batch,
                rgb_as_latents=False,
                guidance_eval=guidance_eval,
            )
            # claforte: TODO: rename the loss_terms keys
            set_loss("sds", guidance_out["loss_sds"])

            if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
                if "comp_normal" not in out:
                    raise ValueError(
                        "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                    )
                normal = out["comp_normal"]
                set_loss(
                    "normal_smooth",
                    (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                    + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
                )

            if self.stage == "gaussian" and self.sugar_reg is not None:
                ## cross entropy loss for opacity to make it binary
                if self.C(self.cfg.loss.lambda_opacity_binary, interpolation='interval') > 0:
                    # only use in static stage
                    visibility_filter = out["visibility_filter"]
                    opacity = self.geometry.get_opacity.unsqueeze(0).repeat(len(visibility_filter), 1, 1)
                    vis_opacities = opacity[torch.stack(visibility_filter)]
                    set_loss(
                        "opacity_binary",
                        -(vis_opacities * torch.log(vis_opacities + 1e-10)
                        + (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)).mean()
                    )

                if self.C(self.cfg.loss.lambda_sugar_density_reg, interpolation='interval') > 0:
                    use_sdf_normal_reg = self.C(self.cfg.loss.lambda_sugar_sdf_normal_reg, interpolation='interval') > 0
                    coarse_args = EasyDict(
                        {
                            # "outputs": out,
                            "n_samples_for_sdf_regularization": self.cfg.n_samples_for_sugar_sdf_reg,
                            "use_sdf_better_normal_loss": use_sdf_normal_reg,
                        }
                    )
                    dloss = self.sugar_reg.coarse_density_regulation(coarse_args)
                    set_loss("sugar_density_reg", dloss["density_regulation"])
                    if use_sdf_normal_reg:
                        set_loss("sugar_sdf_normal_reg", dloss["normal_regulation"])


            if self.stage == "sugar":
                surface_mesh = self.geometry.surface_mesh
                if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
                    set_loss(
                        "normal_consistency",
                        mesh_normal_consistency(surface_mesh)
                    )
                if self.C(self.cfg.loss.lambda_laplacian_smoothing) > 0:
                    set_loss(
                        "laplacian_smoothing",
                        mesh_laplacian_smoothing(surface_mesh, "uniform")
                    )

                if self.C(self.cfg.loss.lambda_opacity_max) > 0:
                    set_loss(
                        "opacity_max",
                        (self.geometry.get_opacity - 1).abs().mean()
                    )

                if self.C(self.cfg.loss.lambda_normal_depth_consistency) > 0:
                    if "comp_normal_from_dist" not in out:
                        raise ValueError(
                            "comp_normal_from_dist is required for normal-depth consistency loss!"
                        )
                    raw_normal = out["comp_normal"] * 2 - 1
                    raw_normal_from_dist = out["comp_normal_from_dist"] * 2 - 1
                    # loss_normal_depth_consistency = F.mse_loss(raw_normal, raw_normal_from_dist)
                    loss_normal_depth_consistency = (1 - (raw_normal * raw_normal_from_dist).sum(dim=-1)).mean()
                    set_loss("normal_depth_consistency", loss_normal_depth_consistency)

            if self.cfg.loss["lambda_rgb_tv"] > 0.0:
                loss_rgb_tv = tv_loss(out["comp_rgb"].permute(0, 3, 1, 2))
                set_loss("rgb_tv", loss_rgb_tv)

            if (
                out.__contains__("comp_depth")
                and self.cfg.loss["lambda_depth_tv"] > 0.0
            ):
                loss_depth_tv = tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
                set_loss("depth_tv", loss_depth_tv)

            if (
                out.__contains__("comp_normal")
                and self.cfg.loss["lambda_normal_tv"] > 0.0
            ):
                loss_normal_tv = tv_loss(out["comp_normal"].permute(0, 3, 1, 2))
                set_loss("normal_tv", loss_normal_tv)

        loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss += loss_weighted

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log(f"train/loss_{guidance}", loss)

        out.update({"loss": loss})
        return out

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        total_loss = 0.0

        if self.stage == "gaussian":
            self.log(
                "gauss_num",
                int(self.geometry.get_xyz.shape[0]),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        out_ref = self.training_substep(batch, batch_idx, guidance="ref")
        total_loss += out_ref["loss"]

        out_rand = self.training_substep(batch, batch_idx, guidance="rand")
        total_loss += out_rand["loss"]


        self.log("train/loss", total_loss, prog_bar=True)


        if self.stage == "gaussian":
            total_loss.backward()

            visibility_filter = out_rand["visibility_filter"]
            radii = out_rand["radii"]
            viewspace_point_tensor = out_rand["viewspace_points"]

            self.geometry.update_states(
                self.global_step,
                visibility_filter,
                radii,
                viewspace_point_tensor,
            )

            opt.step()
            opt.zero_grad(set_to_none=True)

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):

        def save_out_to_image_grid(filename, out):
            self.save_image_grid(
                # f"it{self.true_global_step}-val/{batch['index'][0]}.png",
                filename,
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal_from_dist"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal_from_dist" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["comp_mask"][0, :, :, 0],
                            "kwargs": {"cmap": None, "data_range": (0, 1)},
                        }
                    ]
                    if "comp_mask" in out
                    else []
                ),
                # claforte: TODO: don't hardcode the frame numbers to record... read them from cfg instead.
                name=None,
                step=self.true_global_step,
            )

        batch.update(
            {
                "override_bg_color": torch.ones([1, 3], dtype=torch.float32, device=self.device)
            }
        )
        out = self(batch)
        save_out_to_image_grid(f"it{self.true_global_step}-val/{batch['index'][0]}.png", out)

            
        
    def on_validation_epoch_end(self):
        filestem = f"it{self.true_global_step}-val"
        self.save_img_sequence(
            filestem,
            filestem,
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="validation_epoch_end",
            step=self.true_global_step,
        )

        # Compute quantile of gaussian opacities
        n_quantiles = 10
        for i in range(n_quantiles):
            quant = self.geometry.get_opacity.quantile(i/n_quantiles).item()
            threestudio.info(f'Quantile {i/n_quantiles}: {quant:.04f}')

    def test_step(self, batch, batch_idx):
        batch.update(
            {
                "override_bg_color": torch.ones([1, 3], dtype=torch.float32, device=self.device)
            }
        )
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )

        # Save the current mesh as .ply file
        if self.stage == "gaussian":
            pc_save_path = os.path.join(
                self.get_save_dir(), f"exported_gs_step{self.global_step}.ply"
            )
            self.geometry.save_ply(pc_save_path)
        else:
            self.export_mesh(format="ply")

