import cv2
import json
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import ImageDraw


def WriteBBx(PolygonCoordinates, ObjectLabels,ImageSavePath,ImageBbxPath,ImagePath, SelectRequiredClass,SelectNotRequiredClass,List_of_objects_required,List_of_objects_Notrequired,list_of_objectsinTraffic,SelectTrafficSigns):
    if not ImageSavePath==None:
        img = cv2.imread(ImagePath)
    for ind, (coord, object) in enumerate(zip(PolygonCoordinates, ObjectLabels)):
        if SelectObjectClass(object,SelectRequiredClass,SelectNotRequiredClass,List_of_objects_required,List_of_objects_Notrequired,list_of_objectsinTraffic,SelectTrafficSigns):
            bbx = get_bounding_box(coord)
            if not ImageSavePath==None:
                plot_one_box(bbx[0],bbx[1], img, color=None, label=None, line_thickness=None)
            WriteBBXtxtfile(object,bbx,ImageBbxPath)
    if not ImageSavePath==None:
        cv2.imwrite(ImageSavePath,img)


def getSegmentationWithClass(ImgInstPath,img_width,img_hight):
    PolygonCoordinates = []
    ObjectLabels = []
    #raw_img = Image.open(ImagePath).convert('RGBA')
    with open(ImgInstPath) as json_file:
        seg_data = json.load(json_file)
    if 'features' in seg_data.keys():
        for i, insidefeature in enumerate(seg_data['features']):
            if 'shape' in list(insidefeature['properties'].keys()) and insidefeature['properties']['shape']['type'] == 'Polygon':
                try:
                    coordinates = insidefeature['properties']['shape']['coordinates'][0]
                    for point in coordinates:
                         point[0] = int(point[0] * img_width)
                         point[1] = int(point[1] * img_hight)
                    PolygonCoordinates.append(coordinates)  # appending polygon coordinates
                    ObjectLabels.append(insidefeature['properties']['value']) # appending objectlabel of above polygon
                except:
                    continue
    return PolygonCoordinates, ObjectLabels

def plot_objects_statistics_in_mapillary(All_objects,path_to_saveplots):
    # counting and sorting
    labelcount = np.unique(All_objects, return_counts=True)
    labels,values = labelcount[0],labelcount[1]
    sorted_pairs = sorted(zip(values,labels), reverse=True)
    tuples = zip(*sorted_pairs)
    values, labels = [list(tuple) for tuple in tuples]

    # separating in categories
    objs_dict = {'object--vehicle':{},'human':{},'animal':{},'object--':{},'nature--':{},'not_defined':{}}
    for obj,sz in zip(labels,values):
        if 'object--vehicle' in obj:
            objs_dict['object--vehicle'].update({obj:sz})
        elif 'human' in obj:
            objs_dict['human'].update({obj:sz})
        elif 'animal' in obj:
            objs_dict['animal'].update({obj:sz})
        elif 'object--' in obj and not 'object--vehicle' in obj:
            objs_dict['object--'].update({obj:sz})
        elif 'nature--' in obj:
            objs_dict['nature--'].update({obj:sz})
        else:
            objs_dict['not_defined'].update({obj:sz})
    for cat in objs_dict:
        names = list(objs_dict[cat])
        values = list(objs_dict[cat].values())
        if len(names)==0:
            continue
        plt.figure()
        plt.bar(names,values)
        plt.xlabel('Detections')
        plt.ylabel('Frequency')
        plt.xticks(names,rotation=90)
        plt.title('Object Category: '+ cat +  ', Avg is ' + str(sum(values) // len(values)))
        plt.tight_layout()
        plt.savefig(os.path.join(path_to_saveplots,cat+'.png'),dpi=400)
    return objs_dict

def get_bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

colours = {
    'driveway   ': (12, 160, 25),
    'marking--discrete--arrow--left': (75, 0, 175),
    'marking--discrete--arrow--right': (75, 0, 175),
    'marking--discrete--arrow--split-left-or-straight': (75, 0, 175),
    'marking--discrete--arrow--split-right-or-straight': (75, 0, 175),
    'marking--discrete--arrow--straight': (75, 0, 175),
    #'crosswalk-plain': (75, 0, 175),
    #'crosswalk-zebra': (75, 0, 175),
    #'give-way-row': (75, 0, 175),
    #'give-way-single': (75, 0, 175),
    #'stop-line': (75, 0, 175),
    'marking--discrete--symbol--bicycle': (75, 0, 175),
    #'marking--discrete--text': (75, 0, 175),
    #'other-marking': (75, 0, 175),
    'object--banner': (0, 100, 100),
    'object--sign--advertisement': (100, 0, 0),
    'object--sign--information': (100, 0, 0),
    'object--sign--store': (100, 0, 0),
    'object--support--traffic-sign-frame': (100, 0, 0),
    'object--traffic-sign': (100, 0, 0),
    'object--traffic-sign--direction': (100, 0, 0),
    'object--traffic-sign--information-parking': (100, 0, 0),
    'object--trash-can': (15, 10, 125),
    'object--bench': (0, 0, 255),
    'object--street-light': (50, 155, 0),
    'object--traffic-light': (50, 155, 0),
    'object--support--pole': (15, 50, 75),
    'object--traffic-light--general-upright': (155, 50, 0),
    'object--traffic-light--general-horizontal': (155, 50, 0),
    'object--traffic-light--general-single': (155, 50, 0),
    'object--traffic-light--general-other': (155, 50, 0),
    'object--traffic-light--pedestrians': (155, 50, 0),
    'object--traffic-light--cyclists': (155, 50, 0),
    'object--manhole': (100, 25, 25),  # Static objects: YELLOW.
    #'object--junction-box': (155, 155, 30),
    'object--water-valve': (100, 0, 0),
    #'object--bike-rack': (20, 255, 20),
    'object--catch-basin': (55, 100, 0),
    #'object--cctv-camera': (25, 50, 50),
    'object--fire-hydrant': (25, 50, 50),
    #'object--mailbox': (155, 155, 30),
    'object--parking-meter': (155, 155, 30),
    'object--phone-booth': (155, 155, 30),
    'object--support--utility-pole': (175, 0, 175)
  }


default_colour = (128, 128, 128)
def get_static_objects_dict():
    return colours



def SegmentImage(ImgReadPath,segmentation_file,ImgSavePath,PolygonCoordinates,ObjectLabels,static_objDict,draw_2dbbx=True):
    # Load image.
    img = cv2.imread(ImgReadPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if draw_2dbbx:
        for polygon, label in zip(PolygonCoordinates, ObjectLabels):
            if label in list(static_objDict.keys()):
                bbx = get_bounding_box(polygon)
                color = static_objDict[label]
                plot_one_box(bbx[0], bbx[1], img, color=color, label=label, line_thickness=2)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raw_img = Image.fromarray(img)
    #raw_img = Image.open(ImgReadPath).convert('RGBA')
    segm_img = raw_img.copy()
    # Load segmentation information.
    with open(segmentation_file) as json_file:
        seg_data = json.load(json_file)
    if 'features' in seg_data.keys():
        for feat in seg_data['features']:
            if feat['properties']['shape']['type'] == 'Polygon':
                coordinates = feat['properties']['shape']['coordinates'][0]  # Each row is one point, X, Y, normalised.
                for point in coordinates:
                    point[0] = int(point[0] * raw_img.width)
                    point[1] = int(point[1] * raw_img.height)
                coo_tuples = [tuple(x) for x in coordinates]

                draw = ImageDraw.Draw(segm_img)
                if feat['properties']['value'] in colours:
                    draw.polygon(coo_tuples, fill=(colours[feat['properties']['value']]))
                else:
                    print(feat['properties']['value'])
                    draw.polygon(coo_tuples, fill=default_colour)

        overlaid_img = Image.blend(raw_img, segm_img, 0.5)
        overlaid_img.save(ImgSavePath)


def check2dpoints_insidebbxes_for_complete_image(xys,objs_bbxes):
    #having bbxes and xys points, give information either point is inside or not
    return ((xys >= np.array(objs_bbxes)[:, None, :2]) & (xys <= np.array(objs_bbxes)[:, None, 2:])).all(2)

def plot_one_box(c1,c2, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    #c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



#
# colours_pre = {
#     'driveway   ': (12, 160, 25),
#     'marking--discrete--arrow--left': (75, 0, 175),
#     'marking--discrete--arrow--right': (75, 0, 175),
#     'marking--discrete--arrow--split-left-or-straight': (75, 0, 175),
#     'marking--discrete--arrow--split-right-or-straight': (75, 0, 175),
#     'marking--discrete--arrow--straight': (75, 0, 175),
#     'crosswalk-plain': (75, 0, 175),
#     'crosswalk-zebra': (75, 0, 175),
#     'give-way-row': (75, 0, 175),
#     'give-way-single': (75, 0, 175),
#     'stop-line': (75, 0, 175),
#     'marking--discrete--symbol--bicycle': (75, 0, 175),
#     'marking--discrete--text': (75, 0, 175),
#     'other-marking': (75, 0, 175),
#     'object--banner': (0, 100, 100),
#     'object--sign--advertisement': (100, 0, 0),
#     'object--sign--information': (100, 0, 0),
#     'object--sign--store': (100, 0, 0),
#     'object--support--traffic-sign-frame': (100, 0, 0),
#     'object--traffic-sign': (100, 0, 0),
#     'object--traffic-sign--direction': (100, 0, 0),
#     'object--traffic-sign--information-parking': (100, 0, 0),
#     'object--trash-can': (15, 10, 125),
#     'object--bench': (0, 0, 255),
#     'object--street-light': (50, 155, 0),
#     'object--traffic-light': (50, 155, 0),
#     'object--support--pole': (0, 255, 20),
#     'object--traffic-light--general-upright': (155, 50, 0),
#     'object--traffic-light--general-horizontal': (155, 50, 0),
#     'object--traffic-light--general-single': (155, 50, 0),
#     'object--traffic-light--general-other': (155, 50, 0),
#     'object--traffic-light--pedestrians': (155, 50, 0),
#     'object--traffic-light--cyclists': (155, 50, 0),
#     'object--manhole': (100, 25, 25),  # Static objects: YELLOW.
#     'object--junction-box': (155, 155, 30),
#     'object--water-valve': (100, 0, 0),
#     'object--bike-rack': (20, 255, 20),
#     'object--catch-basin': (55, 100, 0),
#     'object--cctv-camera': (25, 50, 50),
#     'object--fire-hydrant': (25, 50, 50),
#     'object--mailbox': (155, 155, 30),
#     'object--parking-meter': (155, 155, 30),
#     'object--phone-booth': (155, 155, 30),
#     'object--support--utility-pole': (175, 0, 175)
#
#   }
