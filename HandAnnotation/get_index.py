import sys
import os

sys.path.append(os.getcwd())
import numpy as np
import open3d as o3d
import mano
import torch
import json
import trimesh
import os

# args
vis_pointcloud = False
vis_mesh = True
save = True
root_path = "HandAnnotation"


with torch.no_grad():
    rh_mano = mano.load(
        model_path="mano/models/MANO_RIGHT.pkl",
        model_type="mano",
        use_pca=True,
        num_pca_comps=45,
        batch_size=1,
        flat_hand_mean=True,
    )
    mano_vertices = rh_mano().vertices[0].numpy()
    mano_faces = rh_mano.faces


def index_of_rgb(points_color, RGB):
    R, G, B = RGB
    id0 = np.argwhere(points_color[:, 0] == R).flatten().tolist()
    id1 = np.argwhere(points_color[:, 1] == G).flatten().tolist()
    id2 = np.argwhere(points_color[:, 2] == B).flatten().tolist()
    return list(set(id0).intersection(set(id1), set(id2)))


def get_blender2mano(blender_faces):
    blender2mano = dict()
    for i in range(blender_faces.shape[0]):
        for j in range(3):
            blender2mano[blender_faces[i][j]] = mano_faces[i][j]
    return blender2mano


def get_finger_colors(path="colors.json"):
    with open(path, "r") as f:
        finger_colors = json.loads(f.read())
    return finger_colors


def get_finger_index(blender_colors, finger_colors, blender2mano):
    finger_index = dict()
    for key in finger_colors:
        blender_index = index_of_rgb(blender_colors, finger_colors[key])
        finger_index[key] = list(map(lambda x: int(blender2mano[x]), blender_index))
    return finger_index


def get_save_index_path(root_path, annotation_mesh_path):
    if "tip" in annotation_mesh_path:
        save_index_path = os.path.join(root_path, "tips_index")
    elif "finger" in annotation_mesh_path:
        save_index_path = os.path.join(root_path, "fingers_index")
    else:
        raise ValueError("annotation_mesh_path should contain tip or finger")
    return save_index_path


if __name__ == "__main__":

    annotation_mesh_path = os.path.join(
        root_path, "colored_fingers.ply"
    )  # colored_tips or colored_fingers
    annotation_colors_path = os.path.join(root_path, "colors.json")
    save_index_path = get_save_index_path(root_path, annotation_mesh_path)

    colored_mesh = o3d.io.read_triangle_mesh(annotation_mesh_path)
    # colored_hand from blender, vertices index is disorder with mano.
    blender_faces = np.asarray(colored_mesh.triangles)
    blender2mano = get_blender2mano(blender_faces)
    # print(blender2mano)

    blender_colors = np.asarray(colored_mesh.vertex_colors)
    finger_colors = get_finger_colors(annotation_colors_path)
    finger_index = get_finger_index(blender_colors, finger_colors, blender2mano)

    if vis_mesh or vis_pointcloud:
        mano_colors = np.zeros_like(mano_vertices)
        for key in finger_colors:
            mano_colors[finger_index[key]] = finger_colors[key]

    if vis_pointcloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mano_vertices)
        pcd.colors = o3d.utility.Vector3dVector(mano_colors)
        o3d.visualization.draw_geometries([pcd])

    if vis_mesh:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mano_vertices)
        mesh.vertex_colors = o3d.utility.Vector3dVector(mano_colors)
        mesh.triangles = o3d.utility.Vector3iVector(mano_faces)
        o3d.visualization.draw_geometries([mesh])

    if save:
        save_index_path += ".json"
        with open(save_index_path, "w") as f:
            f.write(json.dumps(finger_index))
        print("have saved to {}".format(save_index_path))
