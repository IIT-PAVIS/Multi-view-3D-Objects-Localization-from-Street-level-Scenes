import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import os
import random
import networkx as nx
from colmap_utils.visualize_model import draw_camera

## General Functions
def getimagedataframe(DataFrame,image_id):
    return DataFrame[DataFrame['image_id']==image_id]
def getimagedetection(ImageDataFrame):
    Det_Ids = ImageDataFrame['Det_Idx'].value_counts().keys()
    ## Accessing subDataFrame based on one detection in image
    detections = []
    for Det_id in Det_Ids:
        objDataFrame = ImageDataFrame[ImageDataFrame['Det_Idx']==Det_id]
        points3d = objDataFrame['key'].values
        bbxes2d_xyxy = objDataFrame['bbxes2d_xyxy'].values[0] ## since each key has bbx , just copying first
        imgid = objDataFrame['image_id'].values[0] ## since each key has same image name , just copying first
        detid = objDataFrame['Det_Idx'].values[0] ## since each key has same Detection ID name , just copying first
        detections.append(tuple((points3d,bbxes2d_xyxy,imgid,detid))) ## providing tuple of 3d points, bounding boxes and image id
    return detections

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(0.1, 0.0)
    return False

def outlier_removal(cloud, nb_neighbors,std_ratio):
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors,std_ratio=std_ratio)
    inlier_cloud = cloud.select_by_index(ind)
    return  inlier_cloud


## Related to Clustering
def SelBest(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]

def getAllObjectsDataFrame(DataFrame):
    #forming another data frame name allobjectdataframe that will contains objects of each image but not matched with anyother
    columns = ['image_id','bbxes2d_xyxy','Object_point3D_ids','OBJ_Id_inImage', 'label']
    AllObjectsDataFrame = pd.DataFrame(columns=columns)

    imagesids = DataFrame['image_id'].value_counts().keys()
    DataFrame_iterator = 0
    ###  We have images sorted according to maximum number of objects and 3d points now accessing from top to bottom
    for image_id in imagesids:
        ## Accessing subDataFrame based on just one image
        ImageDataFrame = DataFrame[DataFrame['image_id']==image_id]

        Det_Ids = ImageDataFrame['Det_Idx'].value_counts().keys()

        ## Accessing subDataFrame based on one detection in image
        for Det_id in Det_Ids:
            objDataFrame = ImageDataFrame[ImageDataFrame['Det_Idx']==Det_id]
            points3d = objDataFrame['key'].values
            bbxes2d_xyxy = objDataFrame.bbxes2d_xyxy.values[0]
            label = objDataFrame.label.values[0]
            print('Object ID: ',Det_id,' Image ID: ', image_id)
            AllObjectsDataFrame.loc[DataFrame_iterator]=[image_id,bbxes2d_xyxy,points3d,Det_id,label]
            DataFrame_iterator+=1
    return AllObjectsDataFrame

def keys2pointcloud(keys,points3D):
    all3dpoints=[]
    for key in keys:
        all3dpoints.append(list(points3D[key].xyz))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all3dpoints)
    #pcd.paint_uniform_color([0,0,0])
    return pcd

def keys2points3D(keys,points3D):
    xyzs = []
    for key in keys:
        xyzs.append(list(points3D[key].xyz))
    return  xyzs


def pointcloud2keys(cloud,clusterkeys,points3D):
    keys=[]
    for key in clusterkeys:
        if points3D[key].xyz in cloud:
            keys.append(key)
    return keys

def getAllObjectkeysFromImages_dict(images_dict):
    keys_3D = []
    for img_key in images_dict.keys():
        for obj_key in images_dict[img_key].keys():
            keys_3D.extend(images_dict[img_key][obj_key]['3Dobj_keys'])
    return keys_3D


def get_cameras(K, T, image_size, scale=0.1):
    #Draw Cameras
    cam_width, cam_height = image_size[0], image_size[1]
    R = T[:3, :3]
    t = T[:3, 3]
    # invert
    t = -R.T @ t
    R = R.T
    # create axis, plane and pyramed geometries that will be drawn
    cam_model = draw_camera(K, R, t, cam_width, cam_height, scale=scale)
    return cam_model




def getAvgpointsInObjects(AllObjectsDataFrame):
    Avg_points = int(sum([len(object) for object in AllObjectsDataFrame.Object_point3D_ids])/len(AllObjectsDataFrame.Object_point3D_ids))
    return Avg_points

def dbscan(pcd,eps, Avg_points):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=Avg_points, print_progress=True))
    max_label = labels.max()
    print("clusters:", max_label + 1)
    return labels,max_label

def colorclusters(pcd,labels,max_label):
    cmap = plt.get_cmap("tab20")
    colors = cmap(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd

def colorclusters_self(pcd,labels,colordict):
    #cmap = plt.get_cmap("tab20")
    #colors = cmap(labels / (max_label if max_label > 0 else 1))
    #colors[labels < 0] = 0
    colors=[]
    for lab in labels:
        if int(lab)<0:
            colors.append([0,0,0])
        else:
            colors.append(list(np.array(colordict[int(lab)])/255))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    return pcd


def get_boundingboxes(pcd,color):
    aabb = pcd.get_axis_aligned_bounding_box()
    color = list(np.array(color)/255)
    aabb.color = tuple(color)
    return aabb
def sortwolists(cluster_points,cluster_labels,reverse=True):
    sorted_pairs = sorted(zip(cluster_points,cluster_labels),reverse=reverse)
    tuples = zip(*sorted_pairs)
    cluster_points, cluster_labels = [list(tuple) for tuple in  tuples]
    return cluster_points, cluster_labels


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    #color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 2, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def getlabelcolors(Alllabels):
    colordict = {}
    for label in Alllabels:
        colordict[label] = [random.randint(0, 255) for _ in range(3)]
    return colordict

### Now plotting on images
def WriteOnImages(TrackDataFrame,cluster_labels,colordict,images_sparse_data,ImagesPath,ImagesWithbbxpath):
    frameids = TrackDataFrame['frame_id'].value_counts().keys()
    for frameID in frameids:
        FrameDataFrame = TrackDataFrame[TrackDataFrame['frame_id']==frameID]
        image_id = int(FrameDataFrame.frame_id.values[0])
        img_name =  images_sparse_data[image_id].name
        img0Path = os.path.join(ImagesPath,img_name)
        img0 = cv2.imread(img0Path)
        for i in range(len(FrameDataFrame)):
            track_id = str(FrameDataFrame.track_id.values[i])
            label = FrameDataFrame.track_id.values[i]
            xyxy= [FrameDataFrame.X0.values[i],FrameDataFrame.Y0.values[i],FrameDataFrame.X1.values[i],FrameDataFrame.Y1.values[i]]
            print(img0Path,track_id,xyxy)
            color = colordict[label]
            plot_one_box(xyxy, img0, label=track_id, color=color, line_thickness=4)
        image_save_path = os.path.join(ImagesWithbbxpath,img_name)
        cv2.imwrite(image_save_path, img0)

### Now plotting on images
def WriteOnImages_separatedInFolders(TrackDataFrame,colordict,ImagesPath,ImagesWithbbxpath):
    track_ids = TrackDataFrame['track_id'].value_counts().keys()
    for track_id in track_ids:
        ObjDataFrame = TrackDataFrame[TrackDataFrame['track_id']==track_id]
        indices = ObjDataFrame.index
        for indx in indices:
            img_name = ObjDataFrame.loc[indx].image_name
            img0Path = os.path.join(ImagesPath, img_name)
            img0 = cv2.imread(img0Path)
            xyxy = [ObjDataFrame.loc[indx].X0, ObjDataFrame.loc[indx].Y0, ObjDataFrame.loc[indx].X1,ObjDataFrame.loc[indx].Y1]
            print(img0Path, track_id, xyxy)
            color = colordict[track_id]
            plot_one_box(xyxy, img0, label=str(track_id), color=color, line_thickness=2)
            TrackDirPath = os.path.join(ImagesWithbbxpath, str(track_id))
            if not os.path.exists(TrackDirPath):
                os.mkdir(TrackDirPath)
            image_save_path = os.path.join(TrackDirPath, img_name)
            #try:
            cv2.imwrite(image_save_path, img0)
            #except:
            #    continue

## record some statistics
def record_objects_statistics(TrackDataFrame, path_tosaveStatistics):
    statcolumns = ['object_iD', 'obj_class', 'num_re-Id','']
    statDataFrame = pd.DataFrame(columns=statcolumns)
    track_ids = list(TrackDataFrame['track_id'].value_counts().keys())
    len_frameids = len(TrackDataFrame['frame_id'].value_counts().keys())
    dataframe_iter = 0
    total_re_ids = 0
    for track_id in track_ids:
        ObjDataFrame = TrackDataFrame[TrackDataFrame['track_id']==track_id]
        indices = ObjDataFrame.index
        num_re_ids = len(np.unique(list(ObjDataFrame['frame_id'])))
        total_re_ids = total_re_ids + num_re_ids
        obj_class = ObjDataFrame.loc[indices[0]].track_class
        statDataFrame.loc[dataframe_iter] = [int(track_id),obj_class,num_re_ids,'']
        dataframe_iter += 1
    ## adding another row to show average number of re-identified objects given images
    statDataFrame.loc[dataframe_iter] = ['', '','',''] # just to leave a row blank
    dataframe_iter += 1
    statDataFrame.loc[dataframe_iter] = ['Tot Det.Objs','Tot. Imgs','objs_class', '#Avg Re-Ids',]
    dataframe_iter += 1
    statDataFrame.loc[dataframe_iter] = [int(len(track_ids)), len_frameids,'All', int(np.ceil(total_re_ids/len(track_ids)))]
    tracksfile=os.path.join(path_tosaveStatistics,'states.txt')
    statDataFrame.to_csv(tracksfile, header=True, index=None, sep=' ', mode='a')



def WriteAssociationGraph(TrackDataFrame,path_toSave_MatchedImages):
    import pyvis
    from pyvis.network import Network
    track_ids = TrackDataFrame['track_id'].value_counts().keys()
    for track_id in track_ids:
        ObjDataFrame = TrackDataFrame[TrackDataFrame['track_id']==track_id]
        G = nx.Graph()  # intializing a graph
        indices = ObjDataFrame.index
        for indx in indices:
            #img_name = ObjDataFrame.loc[indx].image_name
            img_id = ObjDataFrame.loc[indx].frame_id
            node1, node2 = 'Obj_' + str(track_id), str(img_id)
            G.add_node(node1)
            G.add_node(node2)
            G.add_edge(node1, node2, weight=1)
        net = Network(notebook=True)
        net.from_nx(G)
        filename = os.path.join(path_toSave_MatchedImages,str(track_id),"assoc_graph.html")
        net.save_graph(filename)
        #plt.figure(figsize=(15, 15))
        #nx.draw_planar(G, with_labels=True)
        #filename = os.path.join(path_toSave_MatchedImages,str(track_id),"assoc_graph.png")
        #plt.savefig(filename)

## functions for modified code
