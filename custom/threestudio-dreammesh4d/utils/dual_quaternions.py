"""
DualQuaternions operations, interpolation, conversions

"""

import json
import torch
import pypose as pp
from pypose.lietensor.lietensor import SO3Type as Quaternion
from pypose.lietensor.lietensor import LieType, SO3Type

def quat_norm(quat):
    norm = quat.norm(dim=-1, keepdim=True)
    return norm

def quat_conjugate(quat: Quaternion):
    return quat.Inv()

class DualQuaternion(object):
    
    def __init__(self, q_r: Quaternion, q_d: Quaternion, normalize=False):
        if not isinstance(q_r, pp.LieTensor) or not isinstance(q_d, pp.LieTensor):
            raise ValueError("q_r and q_d must be of type SO3Type of pypose. Instead received: {} and {}".format(
                type(q_r), type(q_d)))
        if normalize:
            norm = quat_norm(q_r)
            self.q_r = q_r / norm
            self.q_d = q_d / norm
        else:
            self.q_r = q_r
            self.q_d = q_d
    
    def __mul__(self, other):
        q_r_prod = self.q_r * other.q_r
        q_d_prod = self.q_r * other.q_d + self.q_d * other.q_r
        return DualQuaternion(q_r_prod, q_d_prod)
    
    def __imul__(self, other):
        return self.__mul__(other)
    
    def __rmul__(self, other):
        return DualQuaternion(pp.SO3(self.q_r*other), pp.SO3(self.q_d*other))
    
    def __div__(self, other):
        return self.__truediv__(other)
    
    def __truediv__(self, other):
        other_r_sq = other.q_r * other.q_r
        prod_r = self.q_r * other.q_r / other_r_sq
        prod_d = (other.q_r * self.q_d - self.q_r * other.q_d) / other_r_sq
        return DualQuaternion(prod_r, prod_d)
    
    def __add__(self, other):
        return DualQuaternion(self.q_r + other.q_r, self.q_d + other.q_d)
    
    def __eq__(self, other):
        return (self.q_r == other.q_r or self.q_r == -other.q_r) \
               and (self.q_d == other.q_d or self.q_d == -other.q_d)
    
    def __ne__(self, other):
        return not self == other
    
    def transform_point(self, point_xyz):
        """
        Applying dual-quaternion transformation to points with equation: p' = qpq*

        Here we opt to q* = [q_r*, q_d*], which is the same as "https://github.com/neka-nat/dq3d/blob/master/dq3d/DualQuaternion.h#L149"
        and Ben Kenwright's paper `A Beginners Guide to Dual-Quaternions`
        but different with "https://github.com/Achllle/dual_quaternions/tree/master", which this implementation mainly references to.
        """
        dq_point_array =  torch.cat(
            [
                pp.identity_SO3(*point_xyz.shape[:-1]).to(point_xyz), 
                point_xyz, 
                torch.zeros_like(point_xyz[..., :1])
            ],
            dim=-1
        )
        dq_point = DualQuaternion.from_dq_array(dq_point_array)
        res_dq = self * dq_point * self.quaternion_conjugate()
        
        return res_dq.translation
    
    def transform_point_simple(self, point_xyz):
        """
        Simple linear algebra of p' = Rp + t
        """
        rotation = self.q_r.matrix()
        trans = self.translation
        orig_shape = rotation.shape # [..., 3, 3]
        assert point_xyz.shape[0] == orig_shape[-3]
        transformed_xyz = torch.matmul(rotation, point_xyz.unsqueeze(-1)).squeeze(-1) + trans
        return transformed_xyz

    
    @classmethod
    def from_dq_array(cls, r_xyzw_t_xyzw):
        """
        Create a DualQuaternion instance from two quaternions in list format

        :param r_wxyz_t_wxyz: Tensor of shape [..., 8], with the last dim in order: [q_rx, q_ry, q_rz, q_rw, q_tx, q_ty, q_tz, q_tw]
        """
        return cls(pp.SO3(r_xyzw_t_xyzw[..., :4]), pp.SO3(r_xyzw_t_xyzw[..., 4:]))
    
    @classmethod
    def from_quat_pose_array(cls, r_xyzw_t_xyz):
        """
        Create a DualQuaternion object from an array of a quaternion r and translation t
        sigma = r + eps/2 * t * r

        :param r_wxyz_t_xyz: Tensor of shape [..., 7], with the last dim in order: [q_rx, q_ry, q_rz, q_rw, tx, ty, tz]
        """
        q_r = pp.SO3(r_xyzw_t_xyz[...,:4])
        q_r = pp.SO3(q_r / quat_norm(q_r))
        xyz_part = r_xyzw_t_xyz[..., 4:]
        q_d = pp.SO3(
            0.5 
            * pp.SO3(
                torch.cat([xyz_part, torch.zeros_like(xyz_part[..., :1])], dim=-1))
        )* q_r
        return cls(q_r, q_d)
    
    @classmethod
    def from_translation_vector(cls, t_xyz):
        """
        Create a DualQuaternion object from a cartesian point
        :param t_xyz: Tensor of shape [..., 3]
        """
        return cls.from_quat_pose_array(
            torch.cat(
                [torch.ones_like(t_xyz[..., :1]), torch.zeros_like(t_xyz), t_xyz],
                dim=-1
            )
        )
    
    @classmethod
    def identity(cls, dq_size):
        real_part = pp.identity_SO3(*dq_size)
        dual_part = pp.SO3(torch.zeros_like(real_part))
        return cls(real_part, dual_part)
    
    def quaternion_conjugate(self):
        return DualQuaternion(quat_conjugate(self.q_r), quat_conjugate(self.q_d))
    
    def dual_number_conjugate(self):
        return DualQuaternion(self.q_r, pp.SO3(-self.q_d))
    
    def combined_conjugate(self):
        return DualQuaternion(quat_conjugate(self.q_r), pp.SO3(-quat_conjugate(self.q_d)))
    
    def inverse(self):
        q_r_inv = self.q_r.Inv()
        return DualQuaternion(q_r_inv, -q_r_inv * self.q_d * q_r_inv)
    
    def is_normalized(self):
        return self.q_r.norm(dim=-1).allclose(torch.as_tensor(1.)) and \
            (self.q_r * quat_conjugate(self.q_d) + self.q_d * quat_conjugate(self.q_r)).allclose(torch.as_tensor(0.))
        
    def normalize(self):
        """
        Normalize this dual quaternion

        Modifies in place, so this will not preserve self
        """
        normalized = self.normalized()
        self.q_r = normalized.q_r
        self.q_d = normalized.q_d

    def normalized(self):
        """
        Return a copy of the normalized dual quaternion
        
        ||dq|| = sqrt(dq*dq.quaternion_conjugate())  # assuming dq = r + eps*d
               = sqrt(r^2 - d^2*eps^2)
               = sqrt(r^2)
               = |r|
        which is the absolute value, or the norm since it's a vector
        """
        norm_qr = quat_norm(self.q_r)
        # NOTE: Here simply only divide the dual part q_d with q_r's norm following "https://github.com/Achllle/dual_quaternions/tree/master" and "https://github.com/brainexcerpts/Dual-Quaternion-Skinning-Sample-Codes/tree/master"
        # while the implementation of "https://github.com/neka-nat/dq3d/blob/master/dq3d/DualQuaternion.h#L244" is slightly different
        return DualQuaternion(pp.SO3(self.q_r/norm_qr), pp.SO3(self.q_d/norm_qr))
    
    def pow(self, exponent):
        theta = 2 * torch.arccos(self.q_r[..., -1:])
        if theta.allclose(torch.as_tensor(0., dtype=self.q_r.dtype, device=self.q_r.device)):
            return DualQuaternion.from_translation_vector(exponent*self.translation)
        else:
            s0 = self.q_r[..., :3] / torch.sin(theta/2)
            d = -2. * self.q_d[..., -1:] / torch.sin(theta/2)
            se = (self.q_d[..., :3] - s0 * d/2 * torch.cos(theta/2)) / torch.sin(theta/2)

            q_r = pp.SO3(
                torch.cat(
                    [torch.sin(exponent*theta/2) * s0, torch.cos(exponent*theta/2)], dim=-1
                )
            )
            q_d = pp.SO3(
                torch.cat(
                    [
                        exponent* d/2 * torch.cos(exponent*theta/2) * s0 + torch.sin(exponent*theta/2) * se,
                        -exponent * d/2 * torch.sin(exponent*theta/2)
                    ],
                    dim=-1
                )
            )
            return DualQuaternion(q_r, q_d)

    @property
    def translation(self):
        """Get the translation component of the dual quaternion in vector form

        :return: list [x y z]
        """
        mult = pp.SO3((2.0 * self.q_d)) * quat_conjugate(self.q_r)
        return mult.tensor()[..., :3]
    
    @classmethod
    def sclerp(cls, start, stop, t):
        """Screw Linear Interpolation

        Generalization of Quaternion slerp (Shoemake et al.) for rigid body motions
        ScLERP guarantees both shortest path (on the manifold) and constant speed
        interpolation and is independent of the choice of coordinate system.
        ScLERP(dq1, dq2, t) = dq1 * dq12^t where dq12 = dq1^-1 * dq2

        :param start: DualQuaternion instance
        :param stop: DualQuaternion instance
        :param t: fraction betweem [0, 1] representing how far along and around the
                  screw axis to interpolate
        """
        # ensure we always find closest solution. See Kavan and Zara 2005
        mask = (start.q_r * stop.q_r)[..., -1] < 0
        start.q_r[mask] *= -1
        return start * (start.inverse() * stop).pow(t)
    
    def homogeneous_matrix(self):
        """Homogeneous 4x4 transformation matrix from the dual quaternion

        :return 4 by 4 np.array
        """
        h_mat = self.q_r.matrix() # 3x3
        h_mat = torch.cat([h_mat, self.translation.unsqueeze(-1)], dim=-1)
        h_mat = torch.cat([h_mat, torch.zeros_like(h_mat[..., :1, :])], dim=-2)
        h_mat[..., -1] = 1
        return h_mat
    
    def quat_pose_array(self):
        return torch.cat([self.q_r.tensor(), self.translation], dim=-1)
    
    def dq_array(self):
        return torch.cat([self.q_r, self.q_d], dim=-1)


