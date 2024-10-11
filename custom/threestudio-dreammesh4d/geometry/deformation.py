import itertools
from typing import Optional, Sequence, Iterable, Collection

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import torch.nn.init as init

from argparse import ArgumentParser, Namespace
import time

class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64
        self.timebase_pe = 4
        self.defor_depth = 1
        self.posebase_pe = 10
        self.scale_rotation_pe = 2
        self.opacity_pe = 2
        self.timenet_width = 64
        self.timenet_output = 32
        self.bounds = 1.0
        self.plane_tv_weight = 0.0001
        self.time_smoothness_weight = 0.01
        self.l1_time_planes = 0.0001
        self.kplanes_config = {
            'grid_dimensions': 2,
            'input_coordinate_dim': 4,
            'output_coordinate_dim': 32,
            'resolution': [64, 64, 64, 25]
        }
        self.multires = [1, 2, 4, 8]
        self.no_grid = False
        self.no_ds = False
        self.no_dr = False
        self.no_do = True
        self.use_res = True


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp


def init_grid_param(
    grid_nd: int,
    in_dim: int,
    out_dim: int,
    reso: Sequence[int],
    a: float = 0.1,
    b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class HexPlaneField(nn.Module):
    def __init__(
        self,

        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds, bounds, bounds],
                             [-bounds, -bounds, -bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config = [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                                       r * res for r in config["resolution"][:3]
                                   ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)
        # print(f"Initialized model grids: {self.grids}")
        print("feature_dim:", self.feat_dim)

    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ])
        self.aabb = nn.Parameter(aabb, requires_grad=True)
        print("Voxel Plane: set aabb=", self.aabb)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""

        pts = normalize_aabb(pts, self.aabb)
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)

        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps)

        return features


class Linear_Res(nn.Module):
    def __init__(self, W):
        super(Linear_Res, self).__init__()
        self.main_stream = nn.Linear(W, W)

    def forward(self, x):
        x = F.relu(x)
        return x + self.main_stream(x)


class Feat_Res_Net(nn.Module):
    def __init__(self, W, D):
        super(Feat_Res_Net, self).__init__()
        self.D = D
        self.W = W

        self.feature_out = [Linear_Res(self.W)]
        for i in range(self.D - 2):
            self.feature_out.append(Linear_Res(self.W))
        self.feature_out = nn.Sequential(*self.feature_out)

    def initialize_weights(self, ):
        for m in self.feature_out.modules():
            if isinstance(m, nn.Linear):
                init.constant_(m.weight, 0)
                # init.xavier_uniform_(m.weight,gain=1)
                if m.bias is not None:
                    # init.xavier_uniform_(m.bias,gain=1)
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.feature_out(x)


class Head_Res_Net(nn.Module):
    def __init__(self, W, H):
        super(Head_Res_Net, self).__init__()
        self.W = W
        self.H = H

        self.feature_out = [Linear_Res(self.W)]
        self.feature_out.append(nn.Linear(W, self.H))
        self.feature_out = nn.Sequential(*self.feature_out)

    def initialize_weights(self, ):
        for m in self.feature_out.modules():
            if isinstance(m, nn.Linear):
                init.constant_(m.weight, 0)
                # init.xavier_uniform_(m.weight,gain=1)
                if m.bias is not None:
                    # init.xavier_uniform_(m.bias,gain=1)
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.feature_out(x)


class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, skips=[], args=None, use_res=False):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips

        self.no_grid = args.no_grid  # False
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)

        self.use_res = use_res
        if not self.use_res:
            self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform = self.create_net()
        else:
            self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform = self.create_res_net()
        self.args = args

    def create_net(self):

        mlp_out_dim = 0
        if self.no_grid:
            self.feature_out = [nn.Linear(4, self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim, self.W)]

        for i in range(self.D - 1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        output_dim = self.W
        return \
            nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3)), \
                nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 6)), \
                nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 4)), \
                nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1))

    def create_res_net(self, ):

        mlp_out_dim = 0

        if self.no_grid:
            self.feature_out = [nn.Linear(4, self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim, self.W)]

        for i in range(self.D - 1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W))
        self.feature_out = nn.Sequential(*self.feature_out)

        # self.feature_in = nn.Linear(mlp_out_dim + self.grid.feat_dim ,self.W)
        # self.feature_out = Feat_Res_Net(self.W, self.D)

        output_dim = self.W
        return \
            Head_Res_Net(self.W, 3), \
                Head_Res_Net(self.W, 6), \
                Head_Res_Net(self.W, 4), \
                Head_Res_Net(self.W, 1)

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_emb):
        if not self.use_res:
            if self.no_grid:
                h = torch.cat([rays_pts_emb[:, :3], time_emb[:, :1]], -1)
            else:
                grid_feature = self.grid(rays_pts_emb[:, :3], time_emb[:, :1])

                h = grid_feature

            h = self.feature_out(h)
        else:
            # start_time = time.time_ns()
            # debug
            # rays_pts_emb = torch.randn_like(rays_pts_emb)
            # time_emb = torch.randn_like(time_emb)

            grid_feature = self.grid(rays_pts_emb[:, :3], time_emb[:, :1])
            # print(f"grid time: {time.time_ns() - start_time} ns")

            # h =  self.feature_out(self.feature_in(grid_feature))
            # print(grid_feature.shape)
            h = self.feature_out(grid_feature)
            # exit()
        return h

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:, :3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:, :3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx

    def forward_dynamic(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity_emb=None, time_emb=None):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_emb).float()
        dx = self.pos_deform(hidden)
        pts = rays_pts_emb[:, :3] + dx
        if self.args.no_ds:  # False
            scales = scales_emb[:, :3]
        else:
            ds = self.scales_deform(hidden)
            scales = scales_emb[:, :3] + ds
        if self.args.no_dr:  # False
            rotations = rotations_emb[:, :4]
        else:
            dr = self.rotations_deform(hidden)
            rotations = rotations_emb[:, :4] + dr
        if self.args.no_do:  # True
            opacity = opacity_emb[:, :1]
        else:
            do = self.opacity_deform(hidden)
            opacity = opacity_emb[:, :1] + do
        # + do
        # print("deformation value:","pts:",torch.abs(dx).mean(),"rotation:",torch.abs(dr).mean())

        return pts, scales, rotations, opacity
    
    def forward_dynamic_delta(self, rays_pts_emb, time_emb=None):
        hidden = self.query_time(rays_pts_emb, None, None, time_emb).float()
        dx = self.pos_deform(hidden)
        ds = None if self.args.no_ds else self.scales_deform(hidden)
        dr = None if self.args.no_dr else self.rotations_deform(hidden)
        do = None if self.args.no_do else self.opacity_deform(hidden)
        return dx, dr, ds, do

    def forward_dynamic_xyz(self, rays_pts_emb, time_emb):
        hidden = self.query_time(rays_pts_emb, None, None, time_emb).float()
        dx = self.pos_deform(hidden)
        pts = rays_pts_emb[:, :3] + dx
        return pts

    def forward_dynamic_xyz_and_rotation(self, rays_pts_emb, time_emb):
        hidden = self.query_time(rays_pts_emb, None, None, time_emb).float()
        dx = self.pos_deform(hidden)
        # pts = rays_pts_emb[:, :3] + dx
        dr = self.rotations_deform(hidden)
        return dx, dr

    def forward_dynamic_scale(self, rays_pts_emb, time_emb, hidden=None):
        if hidden is None:
            assert rays_pts_emb is not None and time_emb is not None
            hidden = self.query_time(rays_pts_emb, None, None, time_emb).float()
        ds = self.scales_deform(hidden)
        return ds

    def forward_dynamic_rot(self, rays_pts_emb, time_emb, hidden=None):
        if hidden is None:
            assert rays_pts_emb is not None and time_emb is not None
            hidden = self.query_time(rays_pts_emb, None, None, time_emb).float()
        dr = self.rotations_deform(hidden)
        return dr

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" not in name:
                parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        return list(self.grid.parameters())
    # + list(self.timegrid.parameters())


class DeformationNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth = args.defor_depth
        posbase_pe = args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        times_ch = 2 * timebase_pe + 1
        self.timenet = nn.Sequential(
            nn.Linear(times_ch, timenet_width),
            nn.ReLU(),
            nn.Linear(timenet_width, timenet_output)
        )

        self.use_res = args.use_res
        if self.use_res:
            print("Using zero-init and residual")
        self.deformation_net = Deformation(W=net_width, D=defor_depth,
                                           input_ch=(4 + 3) + ((4 + 3) * scale_rotation_pe) * 2,
                                           input_ch_time=timenet_output, args=args, use_res=self.use_res)
        self.register_buffer('time_poc', torch.FloatTensor([(2 ** i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2 ** i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2 ** i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2 ** i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)

        if self.use_res:
            # self.deformation_net.feature_out.initialize_weights()
            self.deformation_net.pos_deform.initialize_weights()
            self.deformation_net.scales_deform.initialize_weights()
            self.deformation_net.rotations_deform.initialize_weights()
            self.deformation_net.opacity_deform.initialize_weights()

        # self.deformation_net.feature_out.apply(initialize_zeros_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, times_sel=None):
        if times_sel is not None:
            return self.forward_dynamic(point, scales, rotations, opacity, times_sel)
        else:
            return self.forward_static(point)

    def forward_static(self, points):
        points = self.deformation_net(points)
        return points

    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)

        means3D, scales, rotations, opacity = self.deformation_net(point,
                                                                   scales,
                                                                   rotations,
                                                                   opacity,
                                                                   # times_feature,
                                                                   times_sel)
        return means3D, scales, rotations, opacity
    
    def forward_dynamic_delta(self, point, times_sel):
        return self.deformation_net.forward_dynamic_delta(point, times_sel)

    def forward_dynamic_xyz(self, point, times_sel):
        return self.deformation_net.forward_dynamic_xyz(point, times_sel)

    def forward_dynamic_scale(self, point, times_sel, hidden=None):
        return self.deformation_net.forward_dynamic_scale(point, times_sel, hidden=hidden)

    def forward_dg_trans_and_rotation(self, point, times_sel):
        return self.deformation_net.forward_dynamic_xyz_and_rotation(point, times_sel)

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())

    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            # ? bug with initialize weight again
            init.xavier_uniform_(m.weight, gain=1)
            # init.xavier_uniform_(m.bias,gain=1)
            # init.constant_(m.bias, 0)


def initialize_zeros_weights(m):
    if isinstance(m, nn.Linear):
        init.constant_(m.weight, 0)
        # init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            # init.xavier_uniform_(m.bias,gain=1)
            init.constant_(m.bias, 0)
