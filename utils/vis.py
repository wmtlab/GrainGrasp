import torch
import open3d as o3d
import numpy as np
import trimesh
import time


def vis_HandObject(hand, obj, window_name="HandObject"):
    """
    :param hand: list->[1,2,...]
    :param obj: list->[1,2,...]
    :return: None
    """
    assert len(obj) == len(hand)

    for i in range(len(obj)):
        o3d.visualization.draw_geometries([hand[i], obj[i]], window_name=window_name)


def vis_GraspProcess(first_hand, hand_vertex_list, obj, time_sleep=0.05):
    """
    :param first_hand: o3d.geometry.TriangleMesh corresponding to the first hand mesh
    :param hand_vertex_list: list->[[N,3],[N,3],...] don't include the first_hand
    :param obj: o3d.geometry.TriangleMesh
    """
    hand = first_hand
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Grasp_Process", visible=True)
    vis.add_geometry(hand)
    vis.add_geometry(obj)
    vis.update_renderer()
    for i in range(len(hand_vertex_list)):
        hand.vertices = o3d.utility.Vector3dVector(hand_vertex_list[i])
        hand.compute_vertex_normals()
        vis.update_geometry(hand)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(time_sleep)
    vis.run()


def get_o3d_pcd(points, colors=None, vis=False):
    """
    :param points: (B,N,3) or (N,3)
    :param colors: Array or Tensor --->(B,N,3) or (N,3) or None
    :return pcd_batch: list->[pcd1,pcd2,...] or pcd
    """

    def get_o3d_pcd_single(point, color, vis=False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point)
        pcd.colors = o3d.utility.Vector3dVector(color)
        if vis:
            o3d.visualization.draw_geometries([pcd], window_name="points")
        return pcd

    if colors is None:
        colors = np.zeros_like(points)
    else:
        assert points.shape == colors.shape

    if len(points.shape) == 2:
        return get_o3d_pcd_single(points, colors, vis)
    else:
        pcd_batch = []
        for i in range(points.shape[0]):
            pcd = get_o3d_pcd_single(points[i], colors[i], vis)
            pcd_batch.append(pcd)
    return pcd_batch


def get_o3d_mesh(points, faces, paint_color=[0.3, 0.3, 0.3], vertex_colors=None):
    """
    :param points: (B,N,3) or (N,3)
    :param faces: (B,M,3) or (M,3)
    :param paint_color: list->[r,g,b]
    :param vertex_colors: (B,N,3) or (N,3) or None
    :return mesh_batch: list->[mesh1,mesh2,...] or mesh
    """

    def get_o3d_mesh_single(point, face, paint_color, vertex_colors):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(point)
        mesh.triangles = o3d.utility.Vector3iVector(face)
        mesh.compute_vertex_normals()
        if vertex_colors is not None:
            vertex_colors = vertex_colors.reshape(-1, 3)
            vertex_colors[
                np.logical_and(
                    vertex_colors[:, 0] == 0,
                    vertex_colors[:, 1] == 0,
                    vertex_colors[:, 2] == 0,
                )
            ] = np.array(paint_color)
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                vertex_colors.reshape(-1, 3)
            )
        else:
            mesh.paint_uniform_color(paint_color)
        return mesh

    if vertex_colors is not None:
        assert len(points.shape) == len(faces.shape) == len(vertex_colors.shape)
        if len(points.shape) == 3:
            assert points.shape[0] == faces.shape[0] == vertex_colors.shape[0]
        else:
            return get_o3d_mesh_single(points, faces, paint_color, vertex_colors)
    else:
        assert len(points.shape) == len(faces.shape)
        if len(points.shape) == 3:
            assert points.shape[0] == faces.shape[0]
        else:
            return get_o3d_mesh_single(points, faces, paint_color, None)

    mesh_batch = []
    for i in range(points.shape[0]):
        if vertex_colors is None:
            mesh = get_o3d_mesh_single(points[i], faces[i], paint_color, None)
        else:
            mesh = get_o3d_mesh_single(
                points[i], faces[i], paint_color, vertex_colors[i]
            )
        mesh_batch.append(mesh)
    return mesh_batch


def trimesh2o3d(tm_mesh, paint_color=[0.3, 0.3, 0.3], vertex_colors=None):
    mesh = tm_mesh.as_open3d
    mesh.compute_vertex_normals()
    if vertex_colors is not None:
        vertex_colors = vertex_colors.reshape(-1, 3)
        vertex_colors[
            np.logical_and(
                vertex_colors[:, 0] == 0,
                vertex_colors[:, 1] == 0,
                vertex_colors[:, 2] == 0,
            )
        ] = np.array(paint_color)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.reshape(-1, 3))
    else:
        mesh.paint_uniform_color(paint_color)
    return mesh
