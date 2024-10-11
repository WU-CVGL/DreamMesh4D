import time
from dataclasses import dataclass, field


import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_knn._C import distCUDA2
from threestudio.utils.misc import C
from threestudio.utils.typing import *

from einops import rearrange
import pypose as pp
import open3d as o3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from .sugar import SuGaRModel
from .deformation import DeformationNetwork, ModelHiddenParams
from ..utils.dual_quaternions import DualQuaternion

from tqdm import trange

import potpourri3d as pp3d


def strain_tensor_to_matrix(strain_tensor: Float[Tensor, "... 6"]):
    strain_matrix = torch.zeros(*strain_tensor.shape[:-1], 3, 3)
    strain_matrix[..., 0, 0] += 1.
    strain_matrix[..., 1, 1] += 1.
    strain_matrix[..., 2, 2] += 1.
    strain_matrix = strain_matrix.to(strain_tensor).flatten(-2, -1)
    strain_matrix[..., [0, 4, 8]] += strain_tensor[..., :3]
    strain_matrix[..., [1, 2, 5]] += strain_tensor[..., 3:]
    strain_matrix[..., [3, 6, 7]] += strain_tensor[..., 3:]
    strain_matrix = strain_matrix.reshape(*strain_tensor.shape[:-1], 3, 3)
    return strain_matrix


@threestudio.register("dynamic-sugar")
class DynamicSuGaRModel(SuGaRModel):
    @dataclass
    class Config(SuGaRModel.Config):
        num_frames: int = 14
        static_learnable: bool = False
        use_deform_graph: bool = True
        dynamic_mode: str = "deformation"  # 'discrete', 'deformation'

        n_dg_nodes: int = 1000
        dg_node_connectivity: int = 8

        dg_trans_lr: Any = 0.001
        dg_rot_lr: Any = 0.001
        dg_scale_lr: Any = 0.001

        vert_trans_lr: Any = 0.001
        vert_rot_lr: Any = 0.001
        vert_scale_lr: Any = 0.001

        deformation_lr: Any = 0.001
        grid_lr: Any = 0.001

        d_xyz: bool = True
        d_rotation: bool = True
        d_opacity: bool = False
        d_scale: bool = True

        dist_mode: str = 'eucdisc'

        skinning_method: str = "hybrid"  # "lbs"(linear blending skinning) or "dqs"(dual-quaternion skinning) or "hybrid"


    cfg: Config

    def configure(self) -> None:
        super().configure()
        if not self.cfg.static_learnable:
            self._points.requires_grad_(False)
            self._quaternions.requires_grad_(False)
            self.all_densities.requires_grad_(False)
            self._sh_coordinates_dc.requires_grad_(False)
            self._sh_coordinates_rest.requires_grad_(False)
            self._scales.requires_grad_(False)
            self._quaternions.requires_grad_(False)
            self.surface_mesh_thickness.requires_grad_(False)

        self.num_frames = self.cfg.num_frames
        self.dynamic_mode = self.cfg.dynamic_mode

        if self.cfg.use_deform_graph:

            self.build_deformation_graph(
                self.cfg.n_dg_nodes,
                None,
                nodes_connectivity=self.cfg.dg_node_connectivity,
                mode=self.cfg.dist_mode
            )

        if self.dynamic_mode == "discrete":
            # xyz
            if self.cfg.use_deform_graph:  # True
                self._dg_node_trans = nn.Parameter(
                    torch.zeros((self.num_frames, self.cfg.n_dg_nodes, 3), device="cuda"), requires_grad=True
                )
                dg_node_rots = torch.zeros((self.num_frames, self.cfg.n_dg_nodes, 4), device="cuda")
                dg_node_rots[..., -1] = 1
                self._dg_node_rots = nn.Parameter(dg_node_rots, requires_grad=True)
                if self.cfg.d_scale:
                    self._dg_node_scales = nn.Parameter(
                        torch.zeros((self.num_frames, self.cfg.n_dg_nodes, 3), device="cuda"), requires_grad=True
                    )
                else:
                    self._dg_node_scales = None

                if self.cfg.skinning_method == "hybrid":
                    self._dg_node_lbs_weights = nn.Parameter(
                        torch.zeros((self.num_frames, self.cfg.n_dg_nodes, 1), device="cuda"), reuqires_grad=True
                    )
                else:
                    self._dg_node_lbs_weights = None

            else:
                self._vert_trans = nn.Parameter(
                    torch.zeros(
                        (self.num_frames, *self._points.shape), device=self.device
                    ).requires_grad_(True)
                )
                vert_rots = torch.zeros((self.num_frames, self.n_verts, 4), device="cuda")
                vert_rots[..., -1] = 1
                self._vert_rots = nn.Parameter(vert_rots, requires_grad=True)
                if self.cfg.d_scale:
                    self._vert_scales = nn.Parameter(
                        torch.zeros((self.num_frames, self.n_verts, 3), device=self.device),
                        requires_grad=True
                    )
                else:
                    self._vert_scales = None

        elif self.dynamic_mode == "deformation":
            deformation_args = ModelHiddenParams(None)
            deformation_args.no_dr = False
            deformation_args.no_ds = not (self.cfg.d_scale or self.cfg.skinning_method == "hybrid" or self.cfg.skinning_method == "lbs")
            deformation_args.no_do = not (self.cfg.skinning_method == "hybrid")

            self._deformation = DeformationNetwork(deformation_args)
            self._deformation_table = torch.empty(0)
        else:
            raise ValueError(f"Unimplemented dynamic mode {self.dynamic_mode}.")

        self.training_setup_dynamic()

        self._gs_bary_weights = torch.cat(
            [self.surface_triangle_bary_coords] * self.n_faces, dim=0
        )
        self._gs_vert_connections = self._surface_mesh_faces.repeat_interleave(
            self.cfg.n_gaussians_per_surface_triangle, dim=0
        )

        # debug
        self.val_step = 0
        self.global_step = 0
        self.save_path = None


    def training_setup_dynamic(self):
        training_args = self.cfg

        l = []
        if self.dynamic_mode == "discrete":
            if self.cfg.use_deform_graph:
                l += [
                    # xyz
                    {
                        "params": [self._dg_node_trans],
                        "lr": C(training_args.dg_trans_lr, 0, 0) * self.spatial_lr_scale,
                        "name": "dg_trans"
                    },
                    {
                        "params": [self._dg_node_rots],
                        "lr": C(training_args.dg_rot_lr, 0, 0),
                        "name": "dg_rotation"
                    },
                ]
                if self.cfg.d_scale:
                    l += [
                        {
                            "params": [self._dg_node_scales],
                            "lr": C(training_args.dg_scale_lr, 0, 0),
                            "name": "dg_scale"
                        },
                    ]
            else:
                l += [
                    {
                        "params": [self._vert_trans],
                        "lr": C(training_args.vert_trans_lr, 0, 0) * self.spatial_lr_scale,
                        "name": "vert_trans",
                    },
                    {
                        "params": [self._vert_rots],
                        "lr": C(training_args.vert_rot_lr, 0, 0),
                        "name": "vert_rotation",
                    },
                ]
                if self.cfg.d_scale:
                    l += [
                        {
                            "params": [self._vert_scales],
                            "lr": C(training_args.vert_scale_lr, 0, 0),
                            "name": "vert_scale"
                        }
                    ]

        elif self.dynamic_mode == "deformation":
            l += [
                {
                    "params": list(self._deformation.get_mlp_parameters()),
                    "lr": C(training_args.deformation_lr, 0, 0) * self.spatial_lr_scale,
                    "name": "deformation"
                },
                {
                    "params": list(self._deformation.get_grid_parameters()),
                    "lr": C(training_args.grid_lr, 0, 0) * self.spatial_lr_scale,
                    "name": "grid"
                }
            ]

        # a = self._deformation.get_grid_parameters()
        # b = self._deformation.get_mlp_parameters()

        self.optimize_list = l
        self.optimize_params = [d["name"] for d in l]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if not ("name" in param_group):
                continue

            if self.dynamic_mode == "discrete":
                if self.cfg.use_deform_graph:
                    if param_group["name"] == "dg_trans":
                        param_group["lr"] = C(
                            self.cfg.dg_trans_lr, 0, iteration, interpolation="exp"
                        ) * self.spatial_lr_scale
                    if param_group["name"] == "dg_rotation":
                        param_group["lr"] = C(
                            self.cfg.dg_rot_lr, 0, iteration, interpolation="exp"
                        )
                    if param_group["name"] == "dg_scale":
                        param_group["lr"] = C(
                            self.cfg.dg_scale_lr, 0, iteration, interpolation="exp"
                        )
                else:
                    if param_group["name"] == "vert_trans":
                        param_group["lr"] = C(
                            self.cfg.vert_trans_lr, 0, iteration, interpolation="exp"
                        ) * self.spatial_lr_scale
                    if param_group["name"] == "vert_rotation":
                        param_group["lr"] = C(
                            self.cfg.vert_rot_lr, 0, iteration, interpolation="exp"
                        )
                    if param_group["name"] == "vert_scale":
                        param_group["lr"] = C(
                            self.cfg.vert_scale_lr, 0, iteration, interpolation="exp"
                        )
            elif self.dynamic_mode == "deformation":
                if "grid" in param_group["name"]:
                    param_group["lr"] = C(
                        self.cfg.grid_lr, 0, iteration, interpolation="exp"
                    ) * self.spatial_lr_scale
                elif param_group["name"] == "deformation":
                    param_group["lr"] = C(
                        self.cfg.deformation_lr, 0, iteration, interpolation="exp"
                    ) * self.spatial_lr_scale

        self.color_clip = C(self.cfg.color_clip, 0, iteration)

    def get_timed_vertex_xyz(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Float[Tensor, "N_t N_v 3"]:
        n_t = len(timestamp) if timestamp is not None else len(frame_idx)
        deformed_vert_pos = []
        for i in range(n_t):
            t = timestamp[i] if timestamp is not None else None
            f = frame_idx[i] if frame_idx is not None else None
            key = dict_temporal_key(t, f)
            if self._deformed_vert_positions.__contains__(key):
                vert_pos = self._deformed_vert_positions[key]
            else:
                vert_pos = self.get_timed_vertex_attributes(
                    t[None] if t is not None else t,
                    f[None] if f is not None else f
                )["xyz"][0]
            deformed_vert_pos.append(vert_pos)
        deformed_vert_pos = torch.stack(deformed_vert_pos, dim=0)
        return deformed_vert_pos

    def get_timed_vertex_rotation(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None,
        return_matrix: bool = False,
    ) -> Union[Float[pp.LieTensor, "N_t N_v 4"], Float[Tensor, "N_t N_v 3 3"]]:
        n_t = len(timestamp) if timestamp is not None else len(frame_idx)
        deformed_vert_rot = []
        for i in range(n_t):
            t = timestamp[i] if timestamp is not None else None
            f = frame_idx[i] if frame_idx is not None else None
            key = dict_temporal_key(t, f)
            if self._deformed_vert_rotations.__contains__(key):
                vert_rot = self._deformed_vert_rotations[key]
            else:
                vert_rot = self.get_timed_vertex_attributes(
                    t[None] if t is not None else t,
                    f[None] if f is not None else f
                )["rotation"][0]
            deformed_vert_rot.append(vert_rot)
        deformed_vert_rot = torch.stack(deformed_vert_rot, dim=0)
        if return_matrix:
            return deformed_vert_rot.matrix()
        else:
            return deformed_vert_rot

    # ============= Functions to compute normals ============= #
    def get_timed_surface_mesh(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Meshes:
        n_t = len(timestamp) if timestamp is not None else len(frame_idx)
        deformed_vert_pos = self.get_timed_vertex_xyz(timestamp, frame_idx)
        surface_mesh = Meshes(
            # verts=self.get_timed_xyz_vertices(timestamp, frame_idx),
            verts=deformed_vert_pos,
            faces=torch.stack([self._surface_mesh_faces] * n_t, dim=0),
            textures=TexturesVertex(
                verts_features=torch.stack([self._vertex_colors] * n_t, dim=0).clamp(0, 1).to(self.device)
            )
        )
        return surface_mesh

    def get_timed_face_normals(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Float[Tensor, "N_t N_faces 3"]:
        return F.normalize(
            self.get_timed_surface_mesh(timestamp, frame_idx).faces_normals_padded(),
            dim=-1
        )

    def get_timed_gs_normals(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Float[Tensor, "N_t N_gs 3"]:
        return self.get_timed_face_normals(timestamp, frame_idx).repeat_interleave(
            self.cfg.n_gaussians_per_surface_triangle, dim=1
        )

    # ========= Compute deformation nodes' attributes ======== #
    def get_timed_dg_attributes(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None,
    ) -> Dict[str, Union[Float[Tensor, "N_t N_p C"], pp.LieTensor]]:
        n_t = timestamp.shape[0] if timestamp is not None else frame_idx.shape[0]
        timed_attr_list = []
        for i in range(n_t):
            key_timestamp = timestamp[i].item() if timestamp is not None else 0
            key_frame = frame_idx[i].float().item() if frame_idx is not None else 0
            key = key_timestamp + key_frame

            if self.dg_timed_attrs.__contains__(key):
                attrs = self.dg_timed_attrs[key]
            else:
                attrs = self._get_timed_dg_attributes(
                    timestamp=timestamp[i:i + 1] if timestamp is not None else None,
                    frame_idx=frame_idx[i:i + 1] if frame_idx is not None else None,
                )
                self.dg_timed_attrs[key] = attrs
            timed_attr_list.append(attrs)

        timed_attrs = {}
        timed_attrs["xyz"] = torch.cat(
            [attr_dict["xyz"] for attr_dict in timed_attr_list], dim=0
        )
        timed_attrs["rotation"] = pp.SO3(
            torch.cat(
                [attr_dict["rotation"].tensor() for attr_dict in timed_attr_list],
                dim=0
            )
        )
        timed_attrs["scale"] = torch.cat(
            [attr_dict["scale"] for attr_dict in timed_attr_list], dim=0
        ) if self.cfg.d_scale or self.cfg.skinning_method == "hybrid" or self.cfg.skinning_method == "lbs" else None
        timed_attrs["opacity"] = torch.cat(
            [attr_dict["opacity"] for attr_dict in timed_attr_list], dim=0
        ) if self.cfg.skinning_method == "hybrid" else None
        return timed_attrs


    def _get_timed_dg_attributes(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ):
        if self.dynamic_mode == "discrete":
            assert frame_idx is not None
            trans = self._dg_node_trans[frame_idx]
            rot = self._dg_node_rots[frame_idx]
            d_scale = self._dg_node_scales[frame_idx] if self.cfg.d_scale else None
            d_opacity = None

        elif self.dynamic_mode == "deformation":
            assert timestamp is not None
            pts = self._deform_graph_node_xyz

            num_pts = pts.shape[0]
            num_t = timestamp.shape[0]
            pts = torch.cat([pts] * num_t, dim=0)
            ts = timestamp.unsqueeze(-1).repeat_interleave(num_pts, dim=0)


            # start_time = time.time_ns()
            trans, rot, d_scale, d_opacity = self._deformation.forward_dynamic_delta(pts, ts * 2 - 1)
            # print(f"real MLP time: {time.time_ns() - start_time} ns")

            # debug
            # num_time = 500
            # pts_debug = torch.cat([pts for i in range(num_time)])
            # ts_debug = timestamp.unsqueeze(-1).repeat_interleave(num_pts*num_time, dim=0)
            # start_time = time.time_ns()
            # debug_trans, debug_rot, debug_d_scale, debug_d_opacity = self._deformation.forward_dynamic_delta(pts_debug, ts_debug * 2 - 1)
            # print(f"debug MLP time: {time.time_ns() - start_time} ns")



            # trans, rot = self._deformation.forward_dg_trans_and_rotation(pts, ts * 2 - 1)
            trans = trans.reshape(num_t, num_pts, 3)
            rot = rot.reshape(num_t, num_pts, 4)

            # NOTE: why do this?
            idt_quaternion = torch.zeros((1, num_pts, 4)).to(rot)
            idt_quaternion[..., -1] = 1
            rot = rot + idt_quaternion

            if d_scale is not None:
                d_scale = d_scale.reshape(num_t, num_pts, 6)
                # to shear matrix
                d_scale = strain_tensor_to_matrix(d_scale)
            if d_opacity is not None:
                d_opacity = d_opacity.reshape(num_t, num_pts, 1)
                d_opacity = F.sigmoid(d_opacity)
        # rot = rot / rot.norm(dim=-1, keepdim=True)
        rot = F.normalize(rot, dim=-1)
        attrs = {
            "xyz": trans, "rotation": pp.SO3(rot), "scale": d_scale, "opacity": d_opacity
        }
        return attrs

    # =========== Compute mesh vertices' attributes ========== #
    def get_timed_vertex_attributes(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None,
    ) -> Dict[str, Union[Float[Tensor, "N_t N_p C"], pp.LieTensor]]:
        if self.cfg.use_deform_graph:
            vert_attrs = self._get_timed_vertex_attributes_from_dg(timestamp, frame_idx)
        else:
            vert_attrs = self._get_timed_vertex_attributes(timestamp, frame_idx)
        # cache deformed mesh vert positions
        for i in range(vert_attrs["xyz"].shape[0]):
            t = timestamp[i] if timestamp is not None else None
            f = frame_idx[i] if frame_idx is not None else None
            key = dict_temporal_key(t, f)
            self._deformed_vert_positions[key] = vert_attrs["xyz"][i]
            self._deformed_vert_rotations[key] = vert_attrs["rotation"][i]

        return vert_attrs

    def _get_timed_vertex_attributes_from_dg(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None,
    ) -> Dict[str, Union[Float[Tensor, "N_t N_p C"], pp.LieTensor]]:
        n_t = len(timestamp) if timestamp is not None else len(frame_idx)
        neighbor_nodes_xyz: Float[Tensor, "N_p N_n 3"]
        neighbor_nodes_xyz = self._deform_graph_node_xyz[self._xyz_neighbor_node_idx]
        
        # debug time
        # start_time = time.time_ns()
        dg_node_attrs = self.get_timed_dg_attributes(timestamp, frame_idx)
        # print(f"MLP time: {time.time_ns() - start_time} ns")
        
        dg_node_trans, dg_node_rots = dg_node_attrs["xyz"], dg_node_attrs["rotation"]

        neighbor_nodes_trans: Float[Tensor, "N_t N_p N_n 3"]
        neighbor_nodes_rots: Float[pp.LieTensor, "N_t N_p N_n 4"]
        neighbor_nodes_trans = dg_node_trans[:, self._xyz_neighbor_node_idx]
        neighbor_nodes_rots = dg_node_rots[:, self._xyz_neighbor_node_idx]

        # debug time
        # start_time = time.time_ns()
        if self.cfg.d_scale or self.cfg.skinning_method == "hybrid" or self.cfg.skinning_method == "lbs":
            dg_node_scales = dg_node_attrs.get("scale")
            assert dg_node_scales is not None

            neighbor_nodes_scales: Float[Tensor, "N_t N_p N_n 3 3"]
            neighbor_nodes_scales = dg_node_scales[:, self._xyz_neighbor_node_idx]

        # deform vertex xyz
        if self.cfg.skinning_method == "lbs" or self.cfg.skinning_method == "hybrid":
            num_pts = self.get_xyz_verts.shape[0]
            # dists_vec: Float[Tensor, "N_t N_p N_n 3 1"]
            # dists_vec = (self.get_xyz_verts.unsqueeze(1) - neighbor_nodes_xyz).unsqueeze(-1)
            # dists_vec = torch.stack([dists_vec] * n_t, dim=0)

            # deformed_vert_xyz: Float[Tensor, "N_t N_p 3"]
            # deformed_xyz = torch.bmm(
            #     neighbor_nodes_rots.matrix().reshape(-1, 3, 3), dists_vec.reshape(-1, 3, 1)
            # ).squeeze(-1).reshape(n_t, num_pts, -1, 3)
            # deformed_xyz = deformed_xyz + neighbor_nodes_xyz.unsqueeze(0) + neighbor_nodes_trans

            deformed_xyz = torch.bmm(
                neighbor_nodes_scales.reshape(-1, 3, 3),
                self.get_xyz_verts.unsqueeze(0).unsqueeze(2).unsqueeze(-1).repeat(
                    n_t, 1, neighbor_nodes_xyz.shape[1], 1, 1).reshape(-1, 3, 1)
            )
            deformed_xyz = torch.bmm(
                neighbor_nodes_rots.matrix().reshape(-1, 3, 3), deformed_xyz
            ).squeeze(-1).reshape(n_t, num_pts, -1, 3)
            deformed_xyz = deformed_xyz + neighbor_nodes_trans


            # deformed_xyz = torch.bmm(
            #     neighbor_nodes_rots.matrix().reshape(-1, 3, 3),
            #     self.get_xyz_verts.unsqueeze(0).unsqueeze(2).unsqueeze(-1).repeat(
            #         n_t, 1, neighbor_nodes_xyz.shape[1], 1, 1).reshape(-1, 3, 1)
            # ).squeeze(-1).reshape(n_t, num_pts, -1, 3)
            # deformed_xyz = deformed_xyz + neighbor_nodes_trans

            nn_weights = self._xyz_neighbor_nodes_weights[None, :, :, None]
            deformed_vert_xyz_lbs = (nn_weights * deformed_xyz).sum(dim=2)

        if self.cfg.skinning_method == "dqs" or self.cfg.skinning_method == "hybrid":
            dual_quat = DualQuaternion.from_quat_pose_array(
                torch.cat([neighbor_nodes_rots.tensor(), neighbor_nodes_trans], dim=-1)
            )
            q_real: Float[Tensor, "N_t N_p N_n 4"] = dual_quat.q_r.tensor()
            q_dual: Float[Tensor, "N_t N_p N_n 4"] = dual_quat.q_d.tensor()
            nn_weights = self._xyz_neighbor_nodes_weights[None, :, :, None]
            weighted_sum_q_real: Float[Tensor, "N_t N_p 4"] = (q_real * nn_weights).sum(dim=-2)
            weighted_sum_q_dual: Float[Tensor, "N_t N_p 4"] = (q_dual * nn_weights).sum(dim=-2)
            weighted_sum_dual_quat = DualQuaternion.from_dq_array(
                torch.cat([weighted_sum_q_real, weighted_sum_q_dual], dim=-1)
            )
            dq_normalized = weighted_sum_dual_quat.normalized()
            deformed_vert_xyz_dqs = dq_normalized.transform_point_simple(self.get_xyz_verts)


        if self.cfg.skinning_method == "lbs":
            deformed_vert_xyz = deformed_vert_xyz_lbs
        elif self.cfg.skinning_method == "dqs":
            deformed_vert_xyz = deformed_vert_xyz_dqs
        elif self.cfg.skinning_method == "hybrid":
            neighbor_nodes_opacity = dg_node_attrs["opacity"][:, self._xyz_neighbor_node_idx]
            vert_lbs_weight = (
                self._xyz_neighbor_nodes_weights[None, ..., None] * neighbor_nodes_opacity
            ).sum(dim=-2)
            # debug
            vert_lbs_weight = torch.clamp(vert_lbs_weight + 0.4, max=1.0)

            deformed_vert_xyz = vert_lbs_weight * deformed_vert_xyz_lbs + (1 - vert_lbs_weight) * deformed_vert_xyz_dqs

        # deform vertex rotation
        deformed_vert_rots: Float[pp.LieTensor, "N_t N_p 4"]
        deformed_vert_rots = (
            self._xyz_neighbor_nodes_weights[None, ..., None] * neighbor_nodes_rots.Log()
        ).sum(dim=-2)
        deformed_vert_rots = pp.so3(deformed_vert_rots).Exp()
        # deformed_vert_rots = pp.SO3(deformed_vert_rots / deformed_vert_rots.norm(dim=-1, keepdim=True))

        # debug time
        # print(f"Skinning Vertex time: {time.time_ns() - start_time} ns")



        outs = {"xyz": deformed_vert_xyz, "rotation": deformed_vert_rots}
        if self.cfg.d_scale:
            deformed_vert_scales: Float[Tensor, "N_t N_p 3 3"]
            if self.cfg.skinning_method == "lbs":
                deformed_vert_scales = (
                    self._xyz_neighbor_nodes_weights[None, ..., None, None] * neighbor_nodes_scales
                ).sum(dim=-3)
                outs["scale"] = deformed_vert_scales
            elif self.cfg.skinning_method == "hybrid":
                deformed_vert_scales = (
                    self._xyz_neighbor_nodes_weights[None, ..., None, None]
                    * neighbor_nodes_opacity[..., None]
                    * neighbor_nodes_scales
                ).sum(dim=-3)
                deformed_vert_scales = (
                    deformed_vert_scales
                    + (1 - vert_lbs_weight)[..., None] * torch.eye(3).to(deformed_vert_scales)
                )
                outs["scale"] = deformed_vert_scales
        return outs

    def _get_timed_vertex_attributes(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Dict[str, Union[Float[Tensor, "N_t N_p C"], pp.LieTensor]]:
        if self.dynamic_mode == "discrete":
            assert frame_idx is not None
            trans = self._vert_trans[frame_idx]
            rot = self._vert_rots[frame_idx]
            d_scale = self._vert_scales[frame_idx] if self.cfg.d_scale else None
            d_opacity = None

        elif self.dynamic_mode == "deformation":
            assert timestamp is not None
            pts = self._points

            num_pts = pts.shape[0]
            num_t = timestamp.shape[0]
            pts = torch.cat([pts] * num_t, dim=0)
            ts = timestamp.unsqueeze(-1).repeat_interleave(num_pts, dim=0)
            trans, rot, d_scale, d_opacity = self._deformation.forward_dynamic_delta(pts, ts * 2 - 1)
            # trans, rot = self._deformation.forward_dg_trans_and_rotation(pts, ts * 2 - 1)
            trans = trans.reshape(num_t, num_pts, 3)
            rot = rot.reshape(num_t, num_pts, 4)

            # NOTE: why do this?
            idt_quaternion = torch.zeros((1, num_pts, 4)).to(rot)
            idt_quaternion[..., -1] = 1
            rot = rot + idt_quaternion

            if d_scale is not None:
                d_scale = d_scale.reshape(num_t, num_pts, 3)
            if d_opacity is not None:
                d_opacity = d_opacity.reshape(num_t, num_pts, 1)
        # rot = rot / rot.norm(dim=-1, keepdim=True)
        rot = F.normalize(rot, dim=-1)
        attrs = {
            "xyz": trans, "rotation": pp.SO3(rot), "scale": d_scale
        }
        return attrs

    # ========= Compute gaussian kernals' attributes ========= #
    def get_timed_gs_attributes(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None,
    ) -> Dict[str, Union[Float[Tensor, "N_t N_p C"], pp.LieTensor]]:
        vert_attrs = self.get_timed_vertex_attributes(timestamp, frame_idx)
        # xyz
        # debug time
        # start_time = time.time_ns()
        gs_timed_xyz = self._get_gs_xyz_from_vertex(vert_attrs["xyz"])

        # rotations
        gs_drots_q: Float[pp.LieTensor, "N_t N_g 4"] = fuse_rotations(
            self._gs_vert_connections, self._gs_bary_weights, vert_attrs["rotation"]
        )
        gs_rots_q_orig = pp.SO3(
            self.get_rotation[None, :, [1, 2, 3, 0]])  # NOTE: the quaternion order should be considered
        gs_timed_rots = gs_drots_q @ gs_rots_q_orig
        gs_timed_rots = gs_timed_rots.tensor()[..., [3, 0, 1, 2]]
        gs_timed_rots = F.normalize(gs_timed_rots, dim=-1)
        
        # debug time
        # print(f"Skinning GS time: {time.time_ns() - start_time} ns")


        gs_attrs = {"xyz": gs_timed_xyz, "rotation": gs_timed_rots}
        # scales
        if self.cfg.d_scale:
            # gs_scales_orig = self._scales
            # gs_scales_orig = torch.cat([
            #     torch.zeros(len(self._scales), 1, device=self.device),
            #     self._scales], dim=-1)
            # vert_timed_dscales = vert_attrs["scale"][:, self._gs_vert_connections, :]
            # gs_timed_dscale = (self._gs_bary_weights[None] * vert_timed_dscales).sum(dim=-2)
            # # gs_timed_scales = gs_scales_orig + gs_timed_dscale
            # gs_timed_scales = gs_scales_orig * (gs_timed_dscale + 1)
            # gs_timed_scales = torch.cat([
            #     self.surface_mesh_thickness * torch.ones(*gs_timed_scales.shape[:-1], 1, device=self.device),
            #     self.scale_activation(gs_timed_scales[..., 1:])], dim=-1)
            # gs_attrs["scale"] = gs_timed_scales

            gs_scales_orig = torch.stack([self.scaling]*gs_timed_xyz.shape[0], dim=0)
            vert_timed_dscales = vert_attrs["scale"][:, self._gs_vert_connections, ...]
            gs_timed_dscale = (self._gs_bary_weights[None, ..., None] * vert_timed_dscales).sum(dim=-3)
            gs_timed_scales = torch.einsum(
                "tpij,tpjk->tpik", gs_timed_dscale, gs_scales_orig.unsqueeze(-1)
            ).squeeze(-1)
            gs_attrs["scale"] = gs_timed_scales

        return gs_attrs

    def get_timed_gs_all_single_time(self, timestamp=None, frame_idx=None):
        if timestamp is not None and timestamp.ndim == 0:
            timestamp = timestamp[None]
        if frame_idx is not None and frame_idx.ndim == 0:
            frame_idx = frame_idx[None]

        gs_timed_attrs = self.get_timed_gs_attributes(timestamp, frame_idx)
        means3D = gs_timed_attrs["xyz"][0]
        rotations = gs_timed_attrs["rotation"][0]
        if self.cfg.d_scale:
            scales = gs_timed_attrs["scale"][0]
        else:
            scales = self.get_scaling

        opacity = self.get_opacity
        colors_precomp = self.get_points_rgb()
        return means3D, scales, rotations, opacity, colors_precomp

    def _get_gs_xyz_from_vertex(self, xyz_vert=None) -> Float[Tensor, "N_t N_gs 3"]:
        if self.binded_to_surface_mesh:
            if xyz_vert is None:
                xyz_vert = self._points
            # First gather vertices of all triangles
            if xyz_vert.ndim == 2:
                xyz_vert = xyz_vert[None]  # (n_t, n_v, 3)
            faces_verts = xyz_vert[:, self._surface_mesh_faces]  # (n_t, n_faces, 3, 3)

            # Then compute the points using barycenter coordinates in the surface triangles
            points = faces_verts[:, :, None] * self.surface_triangle_bary_coords[
                None, None]  # n_t, n_faces, n_gaussians_per_face, 3, n_coords
            points = points.sum(dim=-2)  # n_t, n_faces, n_gaussians_per_face, n_coords
            # points = points.reshape(self._n_points, 3)
            points = rearrange(points, "t f n c -> t (f n) c")
        else:
            raise ValueError("No vertices when with no mesh binded.")
        return points

    def build_deformation_graph(self, n_nodes, xyz_nodes=None, nodes_connectivity=6, mode="geodisc"):
        threestudio.info(f"Building deformation graph with `{mode}`...")
        device = self.device
        xyz_verts = self.get_xyz_verts
        self._xyz_cpu = xyz_verts.cpu().numpy()
        mesh = o3d.io.read_triangle_mesh(self.cfg.surface_mesh_to_bind_path)

        if xyz_nodes is None:
            downpcd = mesh.sample_points_uniformly(number_of_points=n_nodes)
            # downpcd = mesh.sample_points_poisson_disk(number_of_points=1000, pcl=downpcd)
        else:
            downpcd = o3d.geometry.PointCloud()
            downpcd.points = o3d.utility.Vector3dVector(xyz_nodes.cpu().numpy())

        downpcd.paint_uniform_color([0.5, 0.5, 0.5])
        self._deform_graph_node_xyz = torch.from_numpy(np.asarray(downpcd.points)).float().to(device)

        downpcd_tree = o3d.geometry.KDTreeFlann(downpcd)

        if mode == "eucdisc":
            downpcd_size, _ = self._deform_graph_node_xyz.size()
            # TODO delete unused connectivity attr
            deform_graph_connectivity = [
                torch.from_numpy(
                    np.asarray(
                        downpcd_tree.search_knn_vector_3d(downpcd.points[i], nodes_connectivity + 1)[1][1:]
                    )
                ).to(device)
                for i in range(downpcd_size)
            ]
            self._deform_graph_connectivity = torch.stack(deform_graph_connectivity).long().to(device)
            self._deform_graph_tree = downpcd_tree

            # build connections between the original point cloud to deformation graph node
            xyz_neighbor_node_idx = [
                torch.from_numpy(
                    np.asarray(downpcd_tree.search_knn_vector_3d(self._xyz_cpu[i], nodes_connectivity)[1])
                ).to(device)
                for i in range(self._xyz_cpu.shape[0])
            ]
            xyz_neighbor_nodes_weights = [
                torch.from_numpy(
                    np.asarray(downpcd_tree.search_knn_vector_3d(self._xyz_cpu[i], nodes_connectivity)[2])
                ).float().to(device)
                for i in range(self._xyz_cpu.shape[0])
            ]
        elif mode == "geodisc":
            # vertices = self._xyz_cpu
            # faces = self._surface_mesh_faces.cpu().numpy()
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)

            # init geodisc calculation algorithm (geodesic version)
            # geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)

            # init geodisc calculation algorithm (potpourri3d  version)
            geoalg = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)

            # 1. build a kd tree for all vertices in mesh
            mesh_pcl = o3d.geometry.PointCloud()
            mesh_pcl.points = o3d.utility.Vector3dVector(mesh.vertices)
            mesh_kdtree = o3d.geometry.KDTreeFlann(mesh_pcl)

            # 2. find the nearest vertex of all downsampled points and get their index.
            nearest_vertex = [
                np.asarray(mesh_kdtree.search_knn_vector_3d(downpcd.points[i], 1)[1])[0]
                for i in range(n_nodes)
            ]
            target_index = np.array(nearest_vertex)

            # 3. find k nearest neighbors(geodistance) of mesh vertices and downsample pointcloud
            xyz_neighbor_node_idx = []
            xyz_neighbor_nodes_weights = []
            downpcd_points = np.asarray(downpcd.points)
            for i in trange(self._xyz_cpu.shape[0], leave=False):
                source_index = np.array([i])
                start_time1 = time.time()
                # geodesic distance calculation
                # distances = geoalg.geodesicDistances(source_index, target_index)[0]

                # potpourri3d distance calculation
                distances = geoalg.compute_distance(source_index)[target_index]

                # print(f"i: {i}, geodist calculation: {time.time() - start_time1}")
                sorted_index = np.argsort(distances)

                k_n_neighbor = sorted_index[:nodes_connectivity]
                k_n_plus1_neighbor = sorted_index[:nodes_connectivity + 1]
                vert_to_neighbor_dists = np.linalg.norm(
                    vertices[i] - downpcd_points[k_n_plus1_neighbor], axis=-1
                )

                xyz_neighbor_node_idx.append(torch.from_numpy(k_n_neighbor).to(device))

                xyz_neighbor_nodes_weights.append(
                    torch.from_numpy(
                        # np.exp(
                        #     - (np.linalg.norm(vertices[i] - downpcd_points[k_n_neighbor], axis=1) + 1e-5) ** 2 / 2
                        # )
                        # np.linalg.norm(vertices[i] - downpcd_points[k_n_neighbor], axis=1) ** 2
                        (1 - vert_to_neighbor_dists[:nodes_connectivity] / vert_to_neighbor_dists[-1]) ** 2
                    ).float().to(device)
                )
        else:
            raise ValueError("The mode must be eucdisc or geodisc!")

        self._xyz_neighbor_node_idx = torch.stack(xyz_neighbor_node_idx).long().to(device)

        print(torch.max(self._xyz_neighbor_node_idx))
        print(torch.min(self._xyz_neighbor_node_idx))

        self._xyz_neighbor_nodes_weights = torch.stack(xyz_neighbor_nodes_weights).to(device)
        # self._xyz_neighbor_nodes_weights = torch.sqrt(self._xyz_neighbor_nodes_weights)
        # normalize
        self._xyz_neighbor_nodes_weights = (self._xyz_neighbor_nodes_weights
                                            / self._xyz_neighbor_nodes_weights.sum(dim=-1, keepdim=True)
                                            )

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        super().update_step(epoch, global_step, on_load_weights)

        self.dg_timed_attrs = {}
        # self._have_comp_dg_attrs_this_step = False

        self._deformed_vert_positions = {}
        self._deformed_vert_rotations = {}

        # debug
        self.global_step = global_step



def fuse_rotations(
    neighbor_node_idx: Int[Tensor, "N_p N_n"],
    weights: Float[Tensor, "N_p N_n"],
    rotations: Float[pp.LieTensor, "N_t N_v 4"]
):
    """
    q'_i = Exp(\Sigma_{j\in \mathcal{N}(i)} w_{ij} * Log(q_ij))
    """
    rots_log: Float[pp.LieTensor, "N_t N_p N_n 3"] = rotations[:, neighbor_node_idx].Log()
    weighted_rots: Float[pp.LieTensor, "N_t N_p 4"]
    weighted_rots = (weights[None] * rots_log).sum(dim=-2)
    weighted_rots = pp.so3(weighted_rots).Exp()
    return weighted_rots


def dict_temporal_key(timestamp: float = None, frame_idx: int = None):
    if timestamp is None:
        assert frame_idx is not None
        return f"f{frame_idx}"
    elif frame_idx is None:
        return f"t{timestamp}"
    else:
        return f"t{timestamp}_f{frame_idx}"
