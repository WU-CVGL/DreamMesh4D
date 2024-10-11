import bisect
import math
import os
from dataclasses import dataclass, field
from PIL import Image
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)

from threestudio.utils.typing import *

from threestudio.data.image import (
    SingleImageDataModuleConfig,
    SingleImageIterableDataset,
    SingleImageDataset,
)
from .uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
    RandomCameraArbiraryDataset,
)

@dataclass
class TemporalRandomImageDataModuleConfig(SingleImageDataModuleConfig):
    video_frames_dir: Optional[str] = None
    video_length: int = 14
    num_frames: int = 14
    norm_timestamp: bool = False
    white_background: bool = True

class TemporalRandomImageIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg = parse_structured(
            TemporalRandomImageDataModuleConfig, cfg
        )
        self.video_length = self.cfg.video_length
        self.num_frames = self.cfg.num_frames

        if self.cfg.use_random_camera:
            self.rand_cam_bs = self.cfg.random_camera.batch_size
            self.cfg.random_camera.update(
                {"batch_size": self.num_frames * self.rand_cam_bs}
            )
        self.setup(self.cfg, split)
        # self.single_image_dataset = SingleImageIterableDataset(self.cfg, split)

    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: TemporalRandomImageDataModuleConfig = cfg

        assert self.cfg.use_random_camera  # Fix this later
        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )

        elevation_deg = torch.FloatTensor([self.cfg.default_elevation_deg])
        azimuth_deg = torch.FloatTensor([self.cfg.default_azimuth_deg])
        camera_distance = torch.FloatTensor([self.cfg.default_camera_distance])

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        camera_position: Float[Tensor, "1 3"] = torch.stack(
            [
                camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                camera_distance * torch.sin(elevation),
            ],
            dim=-1,
        )

        center: Float[Tensor, "1 3"] = torch.zeros_like(camera_position)
        up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

        light_position: Float[Tensor, "1 3"] = camera_position
        lookat: Float[Tensor, "1 3"] = F.normalize(center - camera_position, dim=-1)
        right: Float[Tensor, "1 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        self.c2w: Float[Tensor, "1 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
            dim=-1,
        )
        self.c2w4x4: Float[Tensor, "B 4 4"] = torch.cat(
            [self.c2w, torch.zeros_like(self.c2w[:, :1])], dim=1
        )
        self.c2w4x4[:, 3, 3] = 1.0

        self.camera_position = camera_position
        self.light_position = light_position
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distance = camera_distance
        self.fovy = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg]))

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.focal_lengths = [
            0.5 * height / torch.tan(0.5 * self.fovy) for height in self.heights
        ]

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.focal_length = self.focal_lengths[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.set_rays()
        self.load_video_frames()
        self.prev_height = self.height

        self.frame_indices = torch.arange(self.video_length, dtype=torch.long)
        self.timestamps = torch.as_tensor(
            np.linspace(0, 1, self.video_length+2, endpoint=True), dtype=torch.float32
        )[1:-1]

    def set_rays(self):
        # get directions by dividing directions_unit_focal by focal length
        directions: Float[Tensor, "1 H W 3"] = self.directions_unit_focal[None]
        directions[:, :, :, :2] = directions[:, :, :, :2] / self.focal_length

        rays_o, rays_d = get_rays(
            directions,
            self.c2w4x4,
            keepdim=True,
            noise_scale=self.cfg.rays_noise_scale,
            normalize=self.cfg.rays_d_normalize,
        )

        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.1, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(self.c2w4x4, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx

    # Copied from threestudio.data.image.SingleImageDataBase.load_images
    def load_single_frame(self, frame_path):
        # load image
        assert os.path.exists(frame_path), f"Could not find image {frame_path}!"
        rgba = cv2.cvtColor(
            cv2.imread(frame_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        rgba = (
            cv2.resize(
                rgba, (self.width, self.height), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            / 255.0
        )
        rgb = rgba[..., :3]
        rgb: Float[Tensor, "1 H W 3"] = (
            torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank)
        )
        mask: Float[Tensor, "1 H W 1"] = (
            torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0).to(self.rank)
        )
        if self.cfg.white_background:
            rgb[~mask[..., 0], :] = 1.0
        self.rgbs.append(rgb)
        self.masks.append(mask)

        # filename = os.path.basename(frame_path)
        # maskname = filename.replace(".png", "_mask.png")
        # Image.fromarray((rgb[0].cpu().numpy() * 255.).astype(np.uint8)).save(f".cache/{filename}")
        # Image.fromarray(mask.squeeze().cpu().numpy()).save(f".cache/{maskname}")

        print(
            f"[INFO] single image dataset: load image {frame_path} {rgb.shape}"
        )

        # load depth
        if self.cfg.requires_depth:
            depth_path = frame_path.replace("_rgba.png", "_depth.png")
            assert os.path.exists(depth_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

            depth = cv2.resize(
                depth, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
            depth: Float[Tensor, "1 H W 1"] = (
                torch.from_numpy(depth.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(self.rank)
            )
            self.depths.append(depth)
            print(
                f"[INFO] single image dataset: load depth {depth_path} {depth.shape}"
            )
        else:
            depth = None

        # load normal
        if self.cfg.requires_normal:
            normal_path = frame_path.replace("_rgba.png", "_normal.png")
            assert os.path.exists(normal_path)
            normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
            normal = cv2.resize(
                normal, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
            normal: Float[Tensor, "1 H W 3"] = (
                torch.from_numpy(normal.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(self.rank)
            )
            self.normals.append(normal)
            print(
                f"[INFO] single image dataset: load normal {normal_path} {normal.shape}"
            )
        else:
            normal = None

    def load_video_frames(self):
        assert os.path.exists(self.cfg.video_frames_dir), f"Could not find image {self.cfg.video_frames_dir}!"
        self.rgbs = []
        self.masks = []
        if self.cfg.requires_depth:
            self.depths = []
        if self.cfg.requires_normal:
            self.normals = []

        # all_frame_paths = glob.glob(os.path.join(self.cfg.video_frames_dir, "*_rgba.png"))
        # self.video_length = len(all_frame_paths)

        for idx in range(self.video_length):
            try:
                frame_path = os.path.join(self.cfg.video_frames_dir, f"{idx:03}_rgba.png")
                self.load_single_frame(frame_path)
            except:
                frame_path = os.path.join(self.cfg.video_frames_dir, f"{idx}.png")
                self.load_single_frame(frame_path)

        self.rgbs = torch.cat(self.rgbs, dim=0)
        self.masks = torch.cat(self.masks, dim=0)
        if self.cfg.requires_depth:
            self.depths = torch.cat(self.depths, dim=0)
        else:
            self.depths = None

        if self.cfg.requires_normal:
            self.normals = torch.cat(self.normals, dim=0)
        else:
            self.normals = None

    def get_all_images(self):
        return self.rgbs

    def collate(self, batch) -> Dict[str, Any]:
        rand_frame_idx = np.random.choice(self.video_length, (self.num_frames,), replace=False, )
        timestamps = self.timestamps[rand_frame_idx]
        # Add noise to timestamps
        # timestamps = timestamps + 0.1 * (torch.rand_like(timestamps) * 2 - 1) / (self.video_length + 1)

        frame_indices = self.frame_indices[rand_frame_idx]
        batch = {
            "rays_o": self.rays_o.repeat(self.num_frames, 1, 1, 1),
            "rays_d": self.rays_d.repeat(self.num_frames, 1, 1, 1),
            "mvp_mtx": self.mvp_mtx.repeat(self.num_frames, 1, 1),
            "camera_positions": self.camera_position,
            "light_positions": self.light_position,
            "elevation": self.elevation_deg,
            "azimuth": self.azimuth_deg,
            "camera_distances": self.camera_distance,
            "rgb": self.rgbs[rand_frame_idx],
            "ref_depth": self.depths[rand_frame_idx] if self.depths is not None else None,
            "ref_normal": self.normals[rand_frame_idx] if self.normals is not None else None,
            "mask": self.masks[rand_frame_idx],
            "height": self.height,
            "width": self.width,
            "c2w": self.c2w4x4.repeat(self.num_frames, 1, 1),
            "fovy": self.fovy.repeat(self.num_frames),
            "timestamp": timestamps,
            "frame_indices": frame_indices
        }
        if self.cfg.use_random_camera:
            batch_rand_cam = self.random_pose_generator.collate(None)
            batch_rand_cam["timestamp"] = timestamps.repeat_interleave(self.rand_cam_bs)
            batch_rand_cam["frame_indices"] = frame_indices.repeat_interleave(self.rand_cam_bs)
            batch["random_camera"] = batch_rand_cam

        return batch

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        if self.height == self.prev_height:
            return

        self.prev_height = self.height
        self.width = self.widths[size_ind]
        self.focal_length = self.focal_lengths[size_ind]
        threestudio.debug(f"Training height: {self.height}, width: {self.width}")
        self.load_video_frames()

        self.random_pose_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}


class TemporalRandomCameraDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: TemporalRandomImageDataModuleConfig = cfg
        self.split = split

        self.video_length = self.cfg.video_length

        self.frame_indices = torch.arange(self.video_length, dtype=torch.long)
        self.timestamps = torch.as_tensor(
            np.linspace(0, 1, self.video_length+2, endpoint=True), dtype=torch.float32
        )[1:-1]

        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            # make sure the first and last view are not the same
            # azimuth_deg = torch.linspace(0, 360.0, self.n_views + 1)[: self.n_views]
            azimuth_deg = torch.tensor([0., -75., 15., 105., 195.])
        else:
            # azimuth_deg = torch.linspace(0, 360.0, self.n_views)
            azimuth_deg = torch.tensor([0., -75., 15., 105., 195.])

        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180
        light_positions: Float[Tensor, "B 3"] = camera_positions

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, normalize=self.cfg.rays_d_normalize
        )
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances
        self.fovy = fovy

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "height": self.cfg.eval_height,
            "width": self.cfg.eval_width,
            "fovy": self.fovy[index],
            # "proj_mtx": self.proj_mtx[index],
            "n_all_views": self.n_views,
            "timestamps": self.timestamps,
            "frame_indices": self.frame_indices,
            "video_length": self.video_length
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch



@register("temporal-image-datamodule")
class TemporalRandomImageDataModule(pl.LightningDataModule):
    cfg: TemporalRandomImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(TemporalRandomImageDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = TemporalRandomImageIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            val_config = self.cfg.get("random_camera", {})
            val_config.update({"video_length": self.cfg.video_length,})
            self.val_dataset = TemporalRandomCameraDataset(val_config, "val")
        if stage in [None, "test"]:
            val_config = self.cfg.get("random_camera", {})
            val_config.update({"video_length": self.cfg.video_length,})
            self.test_dataset = TemporalRandomCameraDataset(val_config, "test")
        if stage in [None, "predict"]:
            cfg = self.cfg.get("random_camera", {})
            # cfg.update(
            #     {
            #         "batch_size": 1,
            #         "height": 512,
            #         "width": 512,
            #     }
            # )
            # self.predict_dataset = RandomCameraIterableDataset(cfg)
            cfg.update(
                {
                    "predict_height": 1024,
                    "predict_width": 1024,
                    "predict_azimuth_range": (-180, 180),
                    "predict_elevation_range": (-10, 80),
                    "predict_camera_distance_range": (3.8, 3.8),
                    "n_predict_views": 120,
                }
            )
            self.predict_dataset = RandomCameraArbiraryDataset(cfg, "predict")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.predict_dataset, batch_size=1)
