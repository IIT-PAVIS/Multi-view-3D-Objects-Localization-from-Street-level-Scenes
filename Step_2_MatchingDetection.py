import json
import os
from config import config
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
from utils.mapillary import get_bounding_box
from colmap_utils import read_camera_model
from utils.scene_processing import keys2points3D
from utils.scene_processing import sortwolists,getlabelcolors,WriteOnImages_separatedInFolders
from utils.scene_processing import WriteAssociationGraph, WriteOnImages,record_objects_statistics

# loading configurations
config_path = 'config/settings.yaml'
config_dict = config.load_config(config_path, config_path)
paths_dict = config.folders_creator(config_dict) # dictionary containing all the paths

#Accessing the required paths
path_to_intermediate_stuff = paths_dict['path_to_intermediate_stuff']
path_to_givenimages = paths_dict['path_to_givenimages']
path_to_sparse = paths_dict['path_to_sparse']
path_toSave_MatchedImages = paths_dict['path_toSave_MatchedImages']
path_toSave_MatchedImages_inOneFold = paths_dict['path_toSave_MatchedImages_inOneFold']
path_toSave_MatchedImages_inSepFold = paths_dict['path_toSave_MatchedImages_inSepFold']
path_toSave_Track_TxtFiles = paths_dict['path_toSave_Track_TxtFiles']

#Accessing intermediate paths
path_to_predicted_3DObjects = os.path.join(path_to_intermediate_stuff, 'predicted_3D_Objects_Keys.json')
path_to_images_dict = os.path.join(path_to_intermediate_stuff, 'images_dict.json')


# Accessing the set parameters
intersection_threshold =  config_dict['MatchedDetection']['intersection_threshold']
DiscardObjectsWith_Views =  config_dict['MatchedDetection']['DiscardObjectsWith_Views']

if __name__ == '__main__':
    # loading keys of predicted/computed 3D objects
    with open(path_to_predicted_3DObjects) as fp:
        predicted_Objects3D_Keys= json.load(fp)
   # image dictionary which contains object information for each image
    with open(path_to_images_dict) as fp:
        images_dict = json.load(fp)

    objs3D_2_images_dict = {} # having information about 3d object and association among detection
    for obj3D_id in predicted_Objects3D_Keys.keys():
        obj3D_points_inds  = predicted_Objects3D_Keys[obj3D_id]
        print('processing object',obj3D_id , '/',len(predicted_Objects3D_Keys.keys()))
        temp_dict = {}
        overall_2Ddet_count = 0
        for img_id in images_dict.keys():
            for obj2D_id in images_dict[img_id].keys():
                polygon = images_dict[img_id][obj2D_id]['polygon']
                bbx = get_bounding_box(polygon)
                bbx = [bbx[0][0], bbx[0][1], bbx[1][0], bbx[1][1]]
                obj2D_id_3Dkeys = images_dict[img_id][obj2D_id]['3Dobj_keys']
                obj2D_id_label = images_dict[img_id][obj2D_id]['label']
                #node1, node2 = '3DObj_'+obj3D_id , '2Ddet_' + img_id + '_' + obj2D_id
                intersect = np.round(len(set(obj2D_id_3Dkeys) &  set(obj3D_points_inds))/ len(set(obj2D_id_3Dkeys)),4)
                if intersect > intersection_threshold: # atleast 5 %
                    #print('intersection', intersect)
                    temp_dict[overall_2Ddet_count]={'img_id':img_id,'obj2D_label':obj2D_id_label,'weight':intersect,'bbx':bbx}
                    overall_2Ddet_count=overall_2Ddet_count+1
        if len(temp_dict)< int(DiscardObjectsWith_Views):
            continue
        objs3D_2_images_dict[obj3D_id] = temp_dict


    trackcolumns =  ['frame_id','image_name','track_id','track_class','X0', 'Y0', 'X1', 'Y1']
    TrackDataFrame = pd.DataFrame(columns=trackcolumns)
    dataframe_iter = 0
    _, images, points3D = read_camera_model.read_model(path=path_to_sparse, ext=".bin")


# Sorting and checking object on the basis of weight and labels
    updated_labels = []
    lab_count =0
    final_pred_objects = {}
    for obj3D_key in  objs3D_2_images_dict.keys():
        objs = []
        temp_dict = {}
        # here checking either is there any confusion with any other object
        for det_key in objs3D_2_images_dict[obj3D_key].keys():
            objs.append(objs3D_2_images_dict[obj3D_key][det_key]['obj2D_label'])
        labels,counts = np.unique(np.array(objs),return_counts=True)
        if len(counts)==0:
            continue
        counts,labels = sortwolists(list(counts), list(labels), reverse=True)
        # to check either two multiple detection assigned to one object
        bestlabel = labels[0]
        if len(labels)>2 and (labels[0] == labels[1] and counts[0]<3):
            continue
        else:
            bestlabel = labels[0]

        multiviews_count = 0  # have a check if we have enough views and listed in dataframe
        for det_key in objs3D_2_images_dict[obj3D_key].keys():
            ## discarding if there are less then 2 views and only belong one object
            if objs3D_2_images_dict[obj3D_key][det_key]['obj2D_label'] == bestlabel and len(objs3D_2_images_dict[obj3D_key].keys()) >= int(DiscardObjectsWith_Views) :
                image_id = int(objs3D_2_images_dict[obj3D_key][det_key]['img_id'])
                img_name = images[image_id].name
                bbx = objs3D_2_images_dict[obj3D_key][det_key]['bbx']
                TrackDataFrame.loc[dataframe_iter] = [image_id, img_name, int(lab_count),bestlabel, int(bbx[0]), int(bbx[1]),
                                                      int(bbx[2]), int(bbx[3])]
                dataframe_iter += 1
                multiviews_count += 1
        ## here we are writing finaly the detected 3D objects points , its label and that will be used in evaluation
        if  multiviews_count>=2:
            obj3D_points_inds = predicted_Objects3D_Keys[obj3D_key]
            pts3D = keys2points3D(obj3D_points_inds, points3D)
            final_pred_objects[lab_count]=pts3D
            lab_count = lab_count + 1

    # save clusters/3D objects in npy file
    save_predicted_3DObjects = os.path.join(path_to_intermediate_stuff, 'final_pred_objects.npy')
    np.save(save_predicted_3DObjects, final_pred_objects)

    # if want to write in json file
    # with open(save_predicted_3DObjects, 'w') as fp:
    #     json.dump(predicted_Objects3D_refined, fp)

    # writing on images
    updated_labels=list(final_pred_objects.keys())
    colordict = getlabelcolors(updated_labels)

    tracksfile=os.path.join(path_toSave_Track_TxtFiles,'Matched_Detection.txt')
    TrackDataFrame.to_csv(tracksfile, header=True, index=None, sep=' ', mode='a')
    record_objects_statistics(TrackDataFrame, path_toSave_Track_TxtFiles)
    if config_dict['MatchedDetection']['Write_MatchedDetection_on_images_inOneFolder'] :
        WriteOnImages(TrackDataFrame, _, colordict, images, path_to_givenimages, path_toSave_MatchedImages_inOneFold)
    if config_dict['MatchedDetection']['Write_MatchedDetection_on_images_inSeparateFolder']:
        WriteOnImages_separatedInFolders(TrackDataFrame, colordict, path_to_givenimages, path_toSave_MatchedImages_inSepFold)
    if config_dict['MatchedDetection']['WriteAssociationGraph']:
        WriteAssociationGraph(TrackDataFrame, path_toSave_MatchedImages_inSepFold)



