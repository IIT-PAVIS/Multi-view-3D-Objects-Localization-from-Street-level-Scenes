import yaml
#import logging
#from multiprocessing import Manager
import os

#method directory; for this project we use dbscan
#method_dict = {'dbscan': dbscan}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.
    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def folders_creator(config_dict):
    AreaName= config_dict['data']['AreaName']
    dataset_path = config_dict['data']['dataset_path']
    segmentation_path = config_dict['data']['segmentation_path']
    root_path = os.path.join(dataset_path,AreaName)
    path_to_dense = os.path.join(root_path, 'dense')
    path_to_sparse = os.path.join(root_path, 'sparse', '0')
    path_to_givenimages = os.path.join(root_path, 'images')
    path_to_giveninst = os.path.join(root_path, 'inst_seg')
    path_to_fused_ply = os.path.join(path_to_dense, 'fused.ply')

    path_to_resultsDir = os.path.join(root_path, 'Matching_Results')
    path_to_WriteSeg_on_images= os.path.join(path_to_resultsDir, 'SegmentedImages')
    path_to_intermediate_stuff = os.path.join(path_to_resultsDir, 'IntermediateStuff')

    ## paths to write results when having sparse point cloud
    path_toSave_MatchedImages = os.path.join(path_to_resultsDir, 'MatchedImages')
    path_toSave_MatchedImages_inOneFold = os.path.join(path_toSave_MatchedImages, 'InOneFolder')
    path_toSave_MatchedImages_inSepFold = os.path.join(path_toSave_MatchedImages, 'InSeparateFolder')
    path_toSave_Track_TxtFiles = os.path.join(path_to_resultsDir, 'Track_TxtFiles')
    path_toSavePlyFiles = os.path.join(path_to_resultsDir, 'Visualizations')

    # Create main results folder
    if not os.path.exists(path_to_resultsDir):
        os.mkdir(path_to_resultsDir)

    # Create folder to store images Segmentation given in mappillary dataset
    if config_dict['Image_Segmentations']['creat_folder'] == True:
        if not os.path.exists(path_to_WriteSeg_on_images):
            os.mkdir(path_to_WriteSeg_on_images)


    # Create folder to store final tracks given sparse
    if config_dict['MatchedDetection']['Write_MatchedDetection_on_images'] == True:
        if not os.path.exists(path_toSave_MatchedImages):
            os.mkdir(path_toSave_MatchedImages)
        if config_dict['MatchedDetection']['Write_MatchedDetection_on_images_inOneFolder'] == True:
            if not os.path.exists(path_toSave_MatchedImages_inOneFold):
                os.mkdir(path_toSave_MatchedImages_inOneFold)
        if config_dict['MatchedDetection']['Write_MatchedDetection_on_images_inSeparateFolder'] == True:
            if not os.path.exists(path_toSave_MatchedImages_inSepFold):
                os.mkdir(path_toSave_MatchedImages_inSepFold)



    # Create folder to store 3d visualizations
    if config_dict['MatchedDetection']['Plyfiles'] == True:
        if not os.path.exists(path_toSavePlyFiles):
            os.mkdir(path_toSavePlyFiles)

    # Create folder to store intermediate results in-order to make it moduler
    if not os.path.exists(path_to_intermediate_stuff):
        os.mkdir(path_to_intermediate_stuff)

    # Create folder to save text files containing final predicted tracks and ground-truth if available
    if not os.path.exists(path_toSave_Track_TxtFiles):
        os.mkdir(path_toSave_Track_TxtFiles)

    path_dict = {'AreaName':AreaName,'root_path':root_path,'dataset_path':dataset_path,'segmentation_path':segmentation_path,
                 'path_to_givenimages':path_to_givenimages,'path_to_giveninst':path_to_giveninst, 'path_to_dense':path_to_dense,
                 'path_to_sparse':path_to_sparse,'path_to_fused_ply':path_to_fused_ply,
               'path_to_intermediate_stuff':path_to_intermediate_stuff,'path_to_resultsDir':path_to_resultsDir,
               'path_to_WriteSeg_on_images':path_to_WriteSeg_on_images,'path_toSave_MatchedImages':path_toSave_MatchedImages,
                 'path_toSave_MatchedImages_inOneFold':path_toSave_MatchedImages_inOneFold,'path_toSave_MatchedImages_inSepFold':path_toSave_MatchedImages_inSepFold,
                 'path_toSavePlyFiles':path_toSavePlyFiles,
                 'path_toSave_Track_TxtFiles':path_toSave_Track_TxtFiles
                 }

    return path_dict






