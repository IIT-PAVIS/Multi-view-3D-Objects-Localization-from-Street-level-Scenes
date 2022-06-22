import open3d as o3d
import numpy as np


def get_HPR(pts3d,cam_model,print_True = True):
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts3d)
        if print_True: print('Before Removal', len(pts3d))
        diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        print("Define parameters used for hidden_point_removal")
        # camer = [0, 0, diameter]
        camer = list(np.array(cam_model[2].points)[0])
        radius = diameter * 50
        _, pt_map = pcd.hidden_point_removal(camer, radius)
        #pcd = pcd.select_by_index(pt_map)
        if print_True: print('After Removal', len(pt_map))
    except:
        return False
    return pt_map #indices of points visible


def get_SOR(pts3D, nb_neighbors,std_ratio, print_True = True):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3D)
    if print_True: print('Before SOR', len(pts3D))
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    #pcd = pcd.select_by_index(ind)
    if print_True: print('After SOR', len(ind))
    return ind
