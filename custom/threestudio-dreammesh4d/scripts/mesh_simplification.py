import argparse
import os

import open3d as o3d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", required=True, help="path to input mesh")
    parser.add_argument("--scale", default=64, type=int,
                        help="large value for more vertices in simplification model")
    parser.add_argument("--output", required=True, help="path to output mesh")
    args = parser.parse_args()

    mesh_in = o3d.io.read_triangle_mesh(args.mesh_path)
    mesh_in.compute_vertex_normals()

    print(
        f'Input mesh has {len(mesh_in.vertices)} vertices and {len(mesh_in.triangles)} triangles'
    )
    o3d.visualization.draw_geometries([mesh_in])

    voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / args.scale
    print(f'voxel_size = {voxel_size:e}')
    mesh_smp = mesh_in.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)
    print(
        f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
    )
    o3d.visualization.draw_geometries([mesh_smp])

    filename = os.path.basename(args.mesh_path)
    filename = filename.split('.')[0]

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    o3d.io.write_triangle_mesh(os.path.join(args.output, f"{filename}_{args.scale}_{len(mesh_smp.vertices)}.ply"), mesh_smp)
