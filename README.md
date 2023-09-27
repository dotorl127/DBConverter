# KetiDBconverter
for convert open source datasets KITTI/KITTI-like, Waymo, NuScenes and Udacity

## Sep 2023 Update
- [x] refactoring

## July 2023 Update
- [x] added kitti-like label convert
- [x] added kakao dataset converter

## December 2022 Update
- [x] added Udacity dataset

## October 2022 Update
- [x] convert coordinates system among KITTI, Waymo, Nuscenes 
- [x] extract sensor data into tfrecord file
- [x] parse json metadata files and extract each data
- [x] convert label format, class names for each dataset
- [x] visualization point cloud with 3D label
- [x] visualization image with 2D label

## Requirements
- equal or higher **Python3.8**
- please reference to **requirements.txt**

## KetiDBconverter directory hierarchy
```commandline
KetiDBconverter
├─ converter
│  ├─ kitti_converter.py
│  ├─ nuscenes_converter.py
│  ├─ waymo_converter.py
│  └─ udacity_converter.py
├─ dictionary
│  ├─ class_dictionary.py
│  └─ rotation_dictionary.py
├─ utils
│  ├─ label.py
│  ├─ util.py
│  └─ visulize.py
├─ db_infos.yaml
├─ demo.py
├─ KetiDBconverter.py
├─ README.md
└─ requirements.txt
```

## Description directory hierarchy
- **converter**
  - {dataset name}_converter : for convert from {dataset name} to another datasets
- **dictionary**
  - class_dictionary.py : for convert class names each dataset
  - rotation_dictionary.py : for align sensor, label rotation among datasets
  - Udacity Dataset only support LiDAR rotation matrix for extract 2D BBox on Nuscenes
- **utils**
  - label.py : label class
  - util.py : parse each dataset label, check validation matrix shape
  - visulize.py : plot 2D BBOX on image or 3D BBOX on 3D space with point cloud, Udacity dataset has not support 3D visualize
- **db_infos.yaml** : information about each dataset(dataset name, sensor list, class name list), Udacity Dataset hasn't included
- **deme.py** : visualization dataset
- **requirements.txt** : required pip package list

## How to use
### Convert
**Arguments**
```text
-i, --src_db_dir  : Directory to load Dataset
-o, --tgt_db_dir  : Directory to save converted Dataset
-t, --tgt_db_type : Dataset type to convert [KITTI, Waymo, Nuscenes, Udacity]
-c, --config_path : Dataset configuration yaml file path
```
**example command line**
```commandline
python KetiDBconverter.py -i /path/to/KITTI -o /path/to/save -t nuscenes
```
### Visualization
**Arguments**
```text
-i,  --root_path    : Directory to load Dataset
-dt, --dataset_type : Type Name of Dataset to Visulization
-vt, --vis_type     : Type of visualization [2d, 3d]
```
**example command line**
```commandline
python demo.py -i /path/to/KITTI -dt kitti -vt 3d
```
**2d**
![2d](https://github.com/dotorl127/KetiDBconverter/assets/35759912/58e65e12-cf6d-47e7-8371-bda403174431)
**3d**  
![3d](https://github.com/dotorl127/KetiDBconverter/assets/35759912/9c34a784-b870-4303-b804-720cd1b0e2cd)
**project**  
![project](https://github.com/dotorl127/KetiDBconverter/assets/35759912/b95dd94b-8763-4372-b051-59956f8cf8db)

## [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
### Directory hierarchy
```commandline
root
├─ image_2
├─ label_2
├─ calib
└─ velodyne
```
### Label format
```commandline
type, truncated, occluded, alpha, bbox(left, top, right, bottom), dimensions(height, width, lengh), localtion(x, y, z), rotation_y
```
### Calibration format
- P{n} : projection matrix for each {n} camera
- R0_rect : rectifying matrix for P2 camera
- Tr_velo_to_cam : RT matrix for roof LiDAR to camera
- Tr_imu_to_cam : RT matrix for imu(or base_link) to camera
### Coordinate system
- Camera : x(right), y(bottom), z(forward)
- LiDAR : x(forward), y(left), z(up)


## [Waymo](https://waymo.com/open/licensing/)
### Directory hierarchy
```commandline
root
└─ *.tfrecord
```
### Label format
check [proto file](https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto)  
```commandline
bbox(center x, y, width, height), speed(x, y), type, id, location(x, y, z), dimensions(width, length, height), heading, num_lidar_points_in_box
```
### Calibration format
- {camera_name}_intrinsic : {camera_name}'s intrinsic matrix
- {camera_name}_extrinsic : {camera_name}'s RT matrix for camera to base_link
- {camera_name}_width : image width size
- {camera_name}_height : image height size
- {camera_name}_rolling_shutter_direction : check [proto file](https://github.com/waymo-research/waymo-open-dataset/blob/17f070076dad149766357b31e25d27cf8b5da6ac/waymo_open_dataset/dataset.proto#L109)
- {LiDAR_name}_beam_inclinations : each channel's vertical angle of {LiDAR_name}
- {LiDAR_name}_beam_inclination_min : min vertical fov of {LiDAR_name}
- {LiDAR_name}_beam_inclination_max : max vertical fov of {LiDAR_name}
- {LiDAR_name}_extrinsic : {LiDAR_name}'s RT matrix for LiDAR to base_link
### Coordinate system
- Camera : x(forward), y(left), z(up)
- LiDAR : x(forward), y(left), z(up)


## [Nuscenes](https://www.nuscenes.org/nuscenes)
### Directory hierarchy
```commandline
root
├─ v1.0-mini
│  ├─ maps
│  ├─ samples
│  │  ├─ CAM_{name}
│  │  ├─ ...
│  │  ├─ LIDAR_TOP
│  │  ├─ RADAR_{name}
│  │  └─ ...
│  ├─ sweeps
│  │  ├─ CAM_{name}
│  │  ├─ ...
│  │  ├─ LIDAR_TOP
│  │  ├─ RADAR_{name}
│  │  └─ ...
│  ├─ v1.0-mini
│  │  ├─ attribute.json
│  │  ├─ calibrated_sensor.json
│  │  ├─ category.json
│  │  ├─ ego_pose.json
│  │  ├─ instance.json
│  │  ├─ log.json
│  │  ├─ map.json
│  │  ├─ sample.json
│  │  ├─ sample_annotation.json
│  │  ├─ sample_data.json
│  │  ├─ scene.json
│  │  ├─ sensor.json
│  └─ └─ visibility.json
└─ v1.0-trainval
   ├─ maps
   ├─ samples
   ├─ sweeps
   └─ v1.0-trainval
```
### Label format
check sample_annotation json file  
```commandline
type, translation(center x, y, z), size(width, height, length), rotation(quaternion), num_lidar_pts, num_radar_pts, bbox(left, top, right, bottom)
```
### Calibration format
- {camera_name}_translation : {camera_name}'s location to base_link
- {camera_name}_rotation : {camera_name}'s rotation to base_link
- {camera_name}_intrinsic : {camera_name}'s intrinsic matrix
- {LiDAR_name}_translation : {LiDAR_name}'s location to base_link
- {LiDAR_name}_rotation : {LiDAR_name}'s rotation to base_link
### Coordinate system
- Camera : x(right), y(down), z(forward)
- LiDAR : x(right), y(forward), z(up)


## [Udacity](http://bit.ly/udacity-annoations-crowdai)
### Directory hierarchy
```commandline
root
└─ object-detection-crowdai
   ├─ {Frame}.jpg
   ├─ ...
   └─ labels.csv
```
### Label format
```commandline
xmin, ymin, xmax, ymax, Frame, Type, Preview URL
```

## KAKAO
### Directory hierarchy
```commandline
root
├─ sensor
│  ├─ camera[00]
│  │  ├─ {timestamp}.jpg
│  │  ├─ ...
│  │  └─ {timestamp}.jpg
│  ├─ ...
│  ├─ camera[05]
│  ├─ lidar[00]
│  │  ├─ {timestamp}.pcd
│  │  ├─ ...
│  └─ └─ {timestamp}.pcd
└─ meta
   ├─ dataset.json
   ├─ ego_pose.json
   ├─ frame.json
   ├─ frame_annotation.json
   ├─ frame_data.json
   ├─ instance.json
   ├─ log.json
   ├─ sensor.json
   └─ preset.yaml
```
### Label format
check frame_annotation json file  
bbox_image3d:
```commandline
type, corners(x,y coordinates of the 8 vertices of the cuboid)
```
bbox_pcd3d:
```commandline
type, x, y, z, w, l, h, orientation(quaternion)
```

### Calibration format
- {camera_name}_intrinsic : {camera_name}'s intrinsic matrix
- {camera_name}_extrinsic : {camera_name}'s intrinsic matrix
- {LiDAR_name}_extrinsic : {LiDAR_name}'s location to base_link
### Coordinate system
- Camera : x(right), y(down), z(forward)
- LiDAR : x(forward), y(left), z(up)

## kitti-like
### Label format
```commandline
Values  Name         Description
----------------------------------------------------------------------------
   1    type         A string which describes the type of object.
                     The number of type is not limited.
                     The string type name of the original input dataset is used without
                     any modification. 
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
                     (If not available, -1 is inserted)
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown (or if not available)
   1    alpha        Observation angle of object, ranging [-pi..pi]
                     (If not available, -99 is inserted) 
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
                     Available for CAMERA sensors only (If otherwise, -1 is inserted)
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in SENSOR coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in SENSOR coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
   1    track ID     Index for tracking. If not available, -1 is inserted.
```
### Calibration format
```commandline
K: raw intrinsic matrix, default zeros (3, 3)
P: rectified intrinsic matrix, default zeros (3, 4)
D: distortion matrix, default zeros (1, 5)
R0_rect: rotation matrix from origin cam to target cam, default identity (3, 3)
Tr_velo_to_cam: transform matirx from LiDAR to target cam, default identity (4, 4)
Tr_imu_to_velo: transform matirx from ogirin to target cam, default identity (4, 4)
```
### Coordinate system
- leave original, not convert