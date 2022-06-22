# Multi-view 3D Objects Localization from Street Level Scenes

## Contact
Any questions or suggestions are welcome! 

Javed Ahmad 
[javed.ahmad@iit.it](mailto:javed.ahmad@iit.it) , 
[javed.sial91@gmail.com](mailto:javed.sial91@gmail.com)

## Abstract:
This paper presents a method to localize street-level objects
in 3D from images of an urban area. Our method processes 3D sparse
point clouds reconstructed from multi-view images and leverages 2D
instance segmentation to find all objects within the scene and to generate
for each object the corresponding cluster of 3D points and matched
2D detections. The proposed approach is robust to changes in image
sizes, viewpoint changes, and changes in the object’s appearance across
different views. We validate our approach on challenging street-level
crowd-sourced images from the Mapillary platform, showing a significant
improvement in the mean average precision of object localization for the
available Mapillary annotations. These results showcase our method’s
effectiveness in localizing objects in 3D, which could potentially be used
in applications such as high-definition map generation of urban environments.



#### Mapillary street-level scenes

You can **download** the images captured at a particular area/scene of any city where Mapillary service is available by following this blog post [https://blog.mapillary.com/update/2021/12/03/mapillary-python-sdk.html](https://blog.mapillary.com/update/2021/12/03/mapillary-python-sdk.html).
#### Configuration
Set the path of the downloaded scene and the other hyper-parameters in the following file

```config/setting.yaml```

#### Process data and localize/find 3D objects available in the scene 
Use the following script, it loads instance segmentation, images, and sparse reconstruction to refine the sparse point clouds and cluster them to find the objects in the scene.
```bash
python Step_1_and_2_Refine_Scene_and_Clustering.py
```
#### Provide matched 2D detection
Use the following script, it loads localized 3D objects in the scene and matches 2D detections based on that, and writes all matched detection of an instance in one folder. The settings can be changed to have all matched detection together.

```bash
python Step_2_MatchingDetection.py
```
