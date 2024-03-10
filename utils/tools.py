import numpy as np
import json
import open3d as o3d
import trimesh
import os


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def readJson(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def fingerName2fingerId(dict_old):
    """
    change the old key str("thumb", "index", "middle", "ring", "pinky") to int(1, 2, 3, 4, 5)
    """
    dict_new = {}
    dict_new[1] = dict_old["thumb"]
    dict_new[2] = dict_old["index"]
    dict_new[3] = dict_old["middle"]
    dict_new[4] = dict_old["ring"]
    dict_new[5] = dict_old["pinky"]
    return dict_new


def vertices_transformation(vertices, rt):
    """
    rt: 4x4 matrix [R|T]
    """
    p = np.matmul(rt[:3, 0:3], vertices.T) + rt[:3, 3].reshape(-1, 1)
    return p.T


def vertices_rotation(vertices, rt):
    p = np.matmul(rt[:3, 0:3], vertices.T)
    return p.T


def fast_load_obj(file_obj_text, **kwargs):
    """
    Based on the modification of the original code: https://github.com/hwjiang1510/GraspTTA/blob/master/utils/utils.py#35
    Parameters:
    - file_obj_text: A text containing the content of the OBJ file (result of read() method).
    Returns:
    - dict: A representation of the loaded OBJ file.
    """

    # make sure text is utf-8 with only \n newlines
    text = file_obj_text.decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n") + " \n"
    meshes = []

    def append_mesh():
        # append kwargs for a trimesh constructor
        # to our list of meshes
        if len(current["f"]) > 0:
            # get vertices as clean numpy array
            vertices = np.array(current["v"], dtype=np.float64).reshape((-1, 3))
            # do the same for faces
            faces = np.array(current["f"], dtype=np.int64).reshape((-1, 3))
            # get keys and values of remap as numpy arrays
            # we are going to try to preserve the order as
            # much as possible by sorting by remap key
            keys, values = (
                np.array(list(remap.keys())),
                np.array(list(remap.values())),
            )
            # new order of vertices
            vert_order = values[keys.argsort()]
            # we need to mask to preserve index relationship
            # between faces and vertices
            face_order = np.zeros(len(vertices), dtype=np.int64)
            face_order[vert_order] = np.arange(len(vertices), dtype=np.int64)
            # apply the ordering and put into kwarg dict
            loaded = {
                "vertices": vertices[vert_order],
                "faces": face_order[faces],
                "metadata": {},
            }
            # build face groups information
            # faces didn't move around so we don't have to reindex
            if len(current["g"]) > 0:
                face_groups = np.zeros(len(current["f"]) // 3, dtype=np.int64)
                for idx, start_f in current["g"]:
                    face_groups[start_f:] = idx
                loaded["metadata"]["face_groups"] = face_groups
            # we're done, append the loaded mesh kwarg dict
            meshes.append(loaded)

    attribs = {k: [] for k in ["v"]}
    current = {k: [] for k in ["v", "f", "g"]}
    # remap vertex indexes {str key: int index}
    remap = {}
    next_idx = 0
    group_idx = 0
    for line in text.split("\n"):
        line_split = line.strip().split()
        if len(line_split) < 2:
            continue
        if line_split[0] in attribs:
            # v, vt, or vn
            # vertex, vertex texture, or vertex normal
            # only parse 3 values, ignore colors
            attribs[line_split[0]].append([float(x) for x in line_split[1:4]])
        elif line_split[0] == "f":
            # a face
            ft = line_split[1:]
            if len(ft) == 4:
                # hasty triangulation of quad
                ft = [ft[0], ft[1], ft[2], ft[2], ft[3], ft[0]]
            for f in ft:
                # loop through each vertex reference of a face
                # we are reshaping later into (n,3)
                if f not in remap:
                    remap[f] = next_idx
                    next_idx += 1
                    # faces are "vertex index"/"vertex texture"/"vertex normal"
                    # you are allowed to leave a value blank, which .split
                    # will handle by nicely maintaining the index
                    f_split = f.split("/")
                    current["v"].append(attribs["v"][int(f_split[0]) - 1])
                current["f"].append(remap[f])
        elif line_split[0] == "o":
            # defining a new object
            append_mesh()
            # reset current to empty lists
            current = {k: [] for k in current.keys()}
            remap = {}
            next_idx = 0
            group_idx = 0
        elif line_split[0] == "g":
            # defining a new group
            group_idx += 1
            current["g"].append((group_idx, len(current["f"]) // 3))
    if next_idx > 0:
        append_mesh()
    return meshes


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def pc_sample(mesh, num_points=3000):
    sample_pc = trimesh.sample.sample_surface(mesh, num_points)[0]
    return sample_pc
