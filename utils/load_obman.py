import os
import zipfile
import pickle
import trimesh
import numpy as np
from utils import tools


class Load_obman:
    def __init__(self, shapeNet_path, obman_path, mode):
        self.shapeNet_path = shapeNet_path
        self.obman_path = obman_path
        self.shapeNet_zip = zipfile.ZipFile(self.shapeNet_path)
        self.mode = mode
        self.meta_path = os.path.join(self.obman_path, self.mode, "meta")
        self.pklNameList = os.listdir(self.meta_path)

    def set_mode(self, mode):
        self.mode = mode
        self.meta_path = os.path.join(self.obman_path, self.mode, "meta")
        self.pklNameList = os.listdir(self.meta_path)

    def get_meta(self, idx):
        pkl_file = os.path.join(self.meta_path, self.pklNameList[idx])
        meta = pickle.load(open(pkl_file, "rb"))
        return meta

    def get_obj_mesh(self, meta):
        obj_path_seg = meta["obj_path"].split("/")[5:]
        obj_mesh_path = "/".join(obj_path_seg)
        obj_mesh = tools.fast_load_obj(self.shapeNet_zip.read(obj_mesh_path))[0]
        obj_vertices, obj_faces = obj_mesh["vertices"], obj_mesh["faces"]
        obj_vertices = tools.vertices_transformation(obj_vertices, rt=meta["affine_transform"])
        obj_mesh = trimesh.Trimesh(vertices=obj_vertices, faces=obj_faces)
        return obj_mesh

    def get_hand_pc(self, meta):
        return meta["verts_3d"]

    def get_hand_pose(self, meta):
        return meta["hand_pose"]
