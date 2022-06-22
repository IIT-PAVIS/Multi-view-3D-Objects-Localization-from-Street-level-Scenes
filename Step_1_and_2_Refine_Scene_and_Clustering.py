import open3d as o3d
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from colmap_utils import read_camera_model
from utils.numba_points_inside_polygon import parallel_point_in_polygon
from utils.mapillary import getSegmentationWithClass
from utils.mapillary import get_static_objects_dict,SegmentImage
from utils.scene_processing import dbscan,sortwolists,getlabelcolors,get_boundingboxes
from utils.scene_processing import keys2points3D,get_cameras,getAllObjectkeysFromImages_dict
from utils.clean_operators import get_SOR
from config import config




# loading configurations
config_path = 'config/settings.yaml'
config_dict = config.load_config(config_path, config_path)
paths_dict = config.folders_creator(config_dict) # dictionary containing all the paths

#Accessing the required paths
AreaName = paths_dict['AreaName']
root_path = paths_dict['root_path']
segmentation_path = paths_dict['segmentation_path']
path_to_givenimages = paths_dict['path_to_givenimages']
path_to_WriteSeg_on_images = paths_dict['path_to_WriteSeg_on_images']
path_to_sparse = paths_dict['path_to_sparse']
path_toSavePlyFiles = paths_dict['path_toSavePlyFiles']
path_to_intermediate_stuff = paths_dict['path_to_intermediate_stuff']


## Ideally segmentations and other data should be in the same directory
# if you have multiple images sets in separate folder then put and all segmentations at root directory
if os.path.exists(os.path.join(root_path, 'inst_seg')):
    segmentation_path = root_path

# Accessing the set parameters
eps, min_3dpoints = config_dict['Scaning']['eps'], config_dict['Scaning']['min_3dpoints']
nb_neighbors, std_ratio = config_dict['SOR_parameters']['nb_neighbors'], config_dict['SOR_parameters']['std_ratio']
cam_scale = config_dict['miscellaneous']['cam_scale']


if __name__ == '__main__':
    static_objDict = get_static_objects_dict()
    static_objs = list(static_objDict.keys())
    Segmentation_save_true = config_dict['Image_Segmentations']['Write_Seg_on_images']

    cameras, images, points3D = read_camera_model.read_model(path=path_to_sparse, ext=".bin")
    print("num_cameras:", len(cameras))
    print("num_images:", len(images))
    print("num_points3D:", len(points3D))
    imgslist = os.listdir(path_to_givenimages)

## Images wise collecting the information
    xyzs = []
    images_dict = {}   # this dictionary contains valid objects , each image contains label and polygon
    frames = []
    ## main loop interating each image to get all information available
    for idx,key in enumerate(images):
        data = images[key]
        K, T, image_size = read_camera_model.get_cam(data, cameras)
        img_id, img_name, xys, img_xyzs_keys, = data.id, data.name, data.xys, data.point3D_ids
        if not img_name in imgslist:
            continue
        ImagePath = os.path.join(path_to_givenimages,img_name)
        img_name = img_name.split('.')[0]
        ImgInstPath = os.path.join(segmentation_path, 'inst_seg', img_name + '.json')
        #inst_segTrafficPath = os.path.join(segmentation_path, 'traffic_signs', file_name + '.json')
        ImageSavePath = os.path.join(path_to_WriteSeg_on_images, img_name + '.png')
        #ImageSegPath = os.path.join(path_to_WriteSegTexts, img_name + '.txt')
        print('Image Processing: ', img_name, idx, '/', len(images))
        PolygonCoordinates, ObjectLabels = getSegmentationWithClass(ImgInstPath,image_size[0],image_size[1])
        #PolygonCoordinates1, ObjectLabels1  = getSegmentationWithClass(ImagePath,inst_segTrafficPath)
        PolygonCoordinates = PolygonCoordinates #+ PolygonCoordinates1
        ObjectLabels = ObjectLabels #+ ObjectLabels1

        if len(ObjectLabels)==0:  # do not write on image if does not contain any detection
            continue
        if Segmentation_save_true:
            SegmentImage(ImagePath, ImgInstPath, ImageSavePath,PolygonCoordinates,ObjectLabels,static_objDict,draw_2dbbx=True)

        xys_bool = np.full(len(xys), False, dtype=bool)
        static_objs = static_objs#[:12] # selecting only first 12 objects in the list
        objsInImg_dict = {}
        obj_count = 0
        for polygon,obj_lab in zip(PolygonCoordinates,ObjectLabels):
            if obj_lab in static_objs:  # selecting only objects in the given list , discarding irrelivant
                object_bool = parallel_point_in_polygon(xys, np.array(polygon))
                obj_keys = img_xyzs_keys[object_bool]  # getting keys of one object only
                selector = obj_keys != -1  # Ignore points with id == -1, as they are not associated with
                # a 3D point in the reconstructed model.
                obj_keys = obj_keys[selector]
                if len(obj_keys)<=2:
                    continue

                # applying statistical outlier removal
                pts3D = keys2points3D(obj_keys, points3D)
                if len(obj_keys)>=3:
                    ind = get_SOR(pts3D, nb_neighbors, std_ratio, print_True=False)
                    obj_keys = obj_keys[ind]

                obj_count=obj_count+1
                if len(obj_keys) == 0:
                    continue
                objsInImg_dict[obj_count]={'label':obj_lab,'polygon':[[int(poly[0]),int(poly[1])] for poly in polygon],'3Dobj_keys': [int(key) for key in obj_keys]}
                #xys_bool = np.array([object_bool.tolist(),xys_bool.tolist()]).any(0)
        images_dict[img_id] = objsInImg_dict
        cam_model = get_cameras(K, T, image_size, scale=cam_scale)
        frames.extend(cam_model)

    # forming scene first time after statistical removal on object base
    keys_3D = getAllObjectkeysFromImages_dict(images_dict)
    pts3D  = keys2points3D(keys_3D, points3D)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3D)
    pcd.paint_uniform_color([0.2, 0.2, 0.2])
    stuff = [pcd]+frames
    o3d.visualization.draw_geometries(stuff,
                                      zoom=0.2412,
                                      front=[0, -0.2125, -0.5795],
                                      lookat=[1.6172, 0.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024]
                                      )

    filename = os.path.join(path_toSavePlyFiles, '1_selected_scene_elements.ply')
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True, compressed=True, print_progress=True)

    ### Statistical Outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = pcd.select_by_index(ind)
    o3d.visualization.draw_geometries([pcd] + frames,
                                      zoom=0.2412,
                                      front=[0, -0.2125, -0.5795],
                                      lookat=[1.6172, 0.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024]
                                      )

    filename = os.path.join(path_toSavePlyFiles, '2_selected_scene_elements_After_SOR.ply')
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True, compressed=True, print_progress=True)
    ## updating keys aftar statistical removal
    keys_3D = list(np.array(keys_3D)[ind])

    ## applying dbscan
    labels,max_label = dbscan(pcd,eps,min_3dpoints)
    labelcount = np.unique(labels,return_counts=True)  ## now sort labels in decending order and  select most prominant labels
    cluster_labels = labelcount[0][1:]
    cluster_points =  labelcount[1][1:]
    cluster_points, cluster_labels = sortwolists(cluster_points,cluster_labels,reverse=True)
    colordict = getlabelcolors(cluster_labels)
    bbxes = []
    pcds = []
    min_points_to_be_in_cluster = 4
    final_3Dobj_label = 0
    predicted_Objects3D_Keys = {}
    for cluster_label in cluster_labels:
        IndsofCluster = list(np.where(np.isin(labels, cluster_label))[0])
        key_inds = list(np.array(keys_3D)[[int(key) for key in IndsofCluster]])
        if len(key_inds)>= min_points_to_be_in_cluster:
            predicted_Objects3D_Keys[final_3Dobj_label] = [int(key) for key in key_inds]
            pcd = o3d.geometry.PointCloud()
            pts3D = keys2points3D(key_inds,points3D)
            pcd.points = o3d.utility.Vector3dVector(np.array(pts3D))
            pcd.paint_uniform_color(np.array(colordict[cluster_label])/255)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            pcd = pcd.select_by_index(ind)
            aabb = get_boundingboxes(pcd, colordict[cluster_label])
            bbxes.append(aabb)
            pcds.append(pcd)
            print('processed_clusters:', final_3Dobj_label+1,' / ',len(cluster_labels))
            final_3Dobj_label = final_3Dobj_label+1



    o3d.visualization.draw_geometries(pcds+bbxes+frames,
                                      zoom=0.3412,
                                      front=[0, -0.2125, -0.5795],
                                      lookat=[1.6172, 0.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024]
                                      )
    # this .ply contains the clustered 3D points where each cluster is a 3D object
    filename = os.path.join(path_toSavePlyFiles,'3_clustered_scene.ply')


    xyzs = []
    colors = []
    for pcd in pcds:
        xyzs.extend(list(np.asarray(pcd.points)))
        colors.extend(list(np.asarray(pcd.colors)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyzs))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True, compressed=True, print_progress=True)


# saving clusters/3D objects keys in json file
    save_predicted_3DObjects = os.path.join(path_to_intermediate_stuff, 'predicted_3D_Objects_Keys.json')
    with open(save_predicted_3DObjects, 'w') as fp:
        json.dump(predicted_Objects3D_Keys, fp)

# saving computed axis aligned 3D bounding boxes of clusters/3D objects
    bbx_list = []
    for bb in bbxes:
        bbx_list.append(np.asarray(bb.get_box_points()))
    #direct_path = '/home/javedahmad/Dropbox/codes/comparision_mapillary/MapillaryData'
    save_predicted_bbxes =  os.path.join(path_to_intermediate_stuff,'predicted_3D_Objects_BBXES.npy')
    np.save(save_predicted_bbxes,bbx_list)

# saving images_dict which contain objects per image, it would be needed in the further step
    save_images_dict = os.path.join(path_to_intermediate_stuff, 'images_dict.json')
    with open(save_images_dict, 'w') as fp:
        json.dump(images_dict, fp)

