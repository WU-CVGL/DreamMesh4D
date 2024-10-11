import numpy as np
import torch
from numpy import ndarray
from collections import defaultdict
from threestudio.utils.typing import *
import open3d as o3d

### Attempt to import svd batch method. If not provided, use default method
### Sourced from https://github.com/KinglittleQ/torch-batch-svd/blob/master/torch_batch_svd/include/utils.h
try:
    from torch_batch_svd import svd as batch_svd
except ImportError:
    print("torch_batch_svd not installed. Using torch.svd instead")
    batch_svd = torch.svd


class ARAPCoach:

    def __init__(
        self,
        verts: Float[Tensor, "N_verts 3"],
        faces: Int[ndarray, "N_face 3"],
        device: torch.device
    ):

        self.verts = verts
        self.device = device
        self.n_verts = len(verts)
        # faces = None

        if faces is not None:
            self.faces = faces
            self.n_faces = len(faces)

            self.one_ring_neighbors = self.get_one_ring_neighbors(self.faces)
            self.max_n_neighbors = max(map(len, self.one_ring_neighbors.values()))

            # _indices_for_nfmt ijn is vertex i connected to vertex j with the nth index(n \in [0, max_n_neighbors-1])
            self._indices_for_nfmt = self.get_indices()
            # edge_cot_weights i are the weights between vertex i and all its one ring connected vertices(0 for no connection)
            self.edge_cot_weights = self.produce_cot_weights_nfmt(self.verts, sparse=False)
            # edge_matrix nfmt ij is the tensor from vertex j to vertex i
            self.edge_matrix_nfmt = self.produce_edge_matrix_nfmt(self.verts)
            pass

        else:
            # use KNN to compute weight
            verts_cpu = verts.cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(verts_cpu)
            kdtree = o3d.geometry.KDTreeFlann(pcd)

            nodes_connectivity = 8
            verts_neighbor_idx = []
            verts_neighbor_weight = []
            for i in range(verts_cpu.shape[0]):
                _, idx, dist2 = kdtree.search_knn_vector_3d(verts_cpu[i], nodes_connectivity + 1)
                verts_neighbor_idx.append(torch.from_numpy(np.asarray(idx))[1:])

                dist2_tensor = torch.from_numpy(np.asarray(dist2))[1:].float().to(device)
                weights = torch.exp(-(dist2_tensor - torch.min(dist2_tensor)) / torch.max(dist2_tensor))
                verts_neighbor_weight.append(weights)

            self.one_ring_neighbors = {k: v.tolist() for k, v in enumerate(verts_neighbor_idx)}
            self.max_n_neighbors = nodes_connectivity

            self._indices_for_nfmt = self.get_indices()
            self.edge_cot_weights = torch.stack(verts_neighbor_weight)
            self.edge_matrix_nfmt = self.produce_edge_matrix_nfmt(self.verts)
            pass

    @classmethod
    def get_one_ring_neighbors(cls, faces) -> Dict[Int, List[Int]]:
        mapping = defaultdict(set)
        for f in faces:
            for j in range(3):  # for each vert in the face
                i, k = (j + 1) % 3, (j + 2) % 3  # get the 2 other vertices
                mapping[f[j]].add(f[i])
                mapping[f[j]].add(f[k])
        orn = {k: list(v) for k, v in mapping.items()}  # convert to list
        return orn

    def get_indices(self):
        # Produce indices
        ii = []
        jj = []
        nn = []
        for i in range(self.n_verts):
            J = self.one_ring_neighbors[i]
            for n, j in enumerate(J):
                ii.append(i)
                jj.append(j)
                nn.append(n)

        ii = torch.LongTensor(ii).to(self.device)
        jj = torch.LongTensor(jj).to(self.device)
        nn = torch.LongTensor(nn).to(self.device)
        return ii, jj, nn

    @torch.no_grad()
    def produce_cot_weights_nfmt(self, xyz_verts=None, sparse: bool = True):
        """Compute cotangent weights for every one-ring neighbors of each vert"""

        # ======== compute a full size weight matrix first ======= #
        V, F = self.n_verts, self.n_faces
        verts = self.verts if xyz_verts is None else xyz_verts
        faces = self.faces

        if not sparse:
            W = torch.zeros((V, V), device=self.device)

        face_verts = verts[faces]  # (F x 3) of verts per face
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

        # Side lengths of each triangle, of shape (sum(F_n),)
        # A is the side opposite v1, B is opposite v2, and C is opposite v3
        A = (v1 - v2).norm(dim=1)
        B = (v0 - v2).norm(dim=1)
        C = (v0 - v1).norm(dim=1)

        # Area of each triangle (with Heron's formula); shape is (F)
        s = 0.5 * (A + B + C)
        # note that the area can be negative (close to 0) causing nans after sqrt()
        # we clip it to a small positive value
        area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

        # Compute cotangents of angles, of shape (F, 3)
        A2, B2, C2 = A * A, B * B, C * C
        cota = (B2 + C2 - A2) / area
        cotb = (A2 + C2 - B2) / area
        cotc = (A2 + B2 - C2) / area
        cot = torch.stack([cota, cotb, cotc], dim=1)
        cot /= 4.0

        if sparse:
            ii = faces[:, [1, 2, 0]]
            jj = faces[:, [2, 0, 1]]
            idx = np.stack([ii, jj], axis=0).reshape(2, F * 3)
            idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
            w = torch.sparse.FloatTensor(idx, cot.flatten(), (V, V))
            w += w.t()
            W = w
        else:
            i = faces[:, [0, 1, 2]].flatten()  # flattened tensor of by face, v0, v1, v2
            j = faces[:, [1, 2, 0]].flatten()  # flattened tensor of by face, v1, v2, v0

            # flatten cot, such that the following line sets
            # w_ij = 0.5 * cot a_ij
            W[i, j] = 0.5 * cot.flatten()
            # to include b_ij, simply add the transpose to itself
            W = W + W.T

        Wn = torch.zeros((V, self.max_n_neighbors), device=self.device)

        # Produce indices
        ii, jj, nn = self._indices_for_nfmt

        if sparse:
            chunk_size = 10_000
            counter = 0
            while counter * chunk_size < len(ii) - chunk_size:
                start = counter * chunk_size
                end = start + chunk_size
                Wn[ii[start:end], nn[start:end]] = (
                    W.index_select(0, ii[start:end]).to_dense()[:, jj[start:end]]
                )
                counter += 0
            start = counter * chunk_size
            Wn[ii[start:], nn[start:]] = (
                W.index_select(0, ii[start:]).to_dense()[:, jj[start:]]
            )
        else:
            Wn[ii, nn] = W[ii, jj]

        return Wn

    def produce_edge_matrix_nfmt(self, xyz_verts: Float[Tensor, "N_verts 3"]):
        E = torch.zeros(self.n_verts, self.max_n_neighbors, 3).to(self.device)
        ii, jj, nn = self._indices_for_nfmt
        E[ii, nn] = xyz_verts[ii] - xyz_verts[jj]
        return E

    def compute_arap_energy(
        self,
        xyz_prime: Float[Tensor, "N_verts 3"],
        vert_rotations: Float[Tensor, "N_verts 3 3"] = None,
    ):
        xyz = self.verts

        w = self.edge_cot_weights

        P = self.edge_matrix_nfmt
        P_prime = self.produce_edge_matrix_nfmt(xyz_prime)

        if vert_rotations is None:
            ### Calculate covariance matrix in bulk
            D = torch.diag_embed(w, dim1=1, dim2=2)
            S = torch.bmm(P.permute(0, 2, 1), torch.bmm(D, P_prime))

            ## in the case of no deflection, set S = 0, such that R = I. This is to avoid numerical errors
            unchanged_verts = torch.unique(torch.where((P == P_prime).all(dim=1))[0])  # any verts which are undeformed
            S[unchanged_verts] = 0

            U, sig, W = batch_svd(S)
            R = torch.bmm(W, U.permute(0, 2, 1))  # compute rotations

            # Need to flip the column of U corresponding to smallest singular value
            # for any det(Ri) <= 0
            entries_to_flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten()  # idxs where det(R) <= 0
            if len(entries_to_flip) > 0:
                Umod = U.clone()
                cols_to_flip = torch.argmin(sig[entries_to_flip], dim=1)  # Get minimum singular value for each entry
                Umod[entries_to_flip, :, cols_to_flip] *= -1  # flip cols
                R[entries_to_flip] = torch.bmm(W[entries_to_flip], Umod[entries_to_flip].permute(0, 2, 1))
        else:
            R = vert_rotations

        # Compute energy
        rot_rigid = torch.bmm(R, P.permute(0, 2, 1)).permute(0, 2, 1)
        stretch_vec = P_prime - rot_rigid  # stretch vector
        stretch_norm = (torch.norm(stretch_vec, dim=2) ** 2)  # norm over (x,y,z) space
        energy = (w * stretch_norm).sum()

        return energy
