# KetiDBconverter
for convert open source datasets KITTI, Waymo, NuScenes and Udacity

## July 2023 Update
- [ ] add KAKAO dataset

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
```
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
```
python KetiDBconverter.py --src_db_dir {source dataset path to load} --tgt_db_dir {target dataset path to save} --tgt_db_type {dataset name to convert[kitti, waymo, nuscenes, udacity]}
```
### Visualization
```
python demo.py --root_dir {dataset path to load} --dataset_type {dataset type name to visualize} --vis_type {visualize type name[2d, 3d]}
```

## KITTI
### Directory hierarchy
```
root
├─ image_2
├─ label_2
├─ calib
└─ velodyne
```
### Label format
```
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


## Waymo
### Directory hierarchy
```
root
└─ *.tfrecord
```
### Label format
check [proto file](https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto)<br />
```
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


## Nuscenes
### Directory hierarchy
```
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
├─ v1.0-trainval
│  ├─ maps
│  ├─ samples
│  ├─ sweeps
└─ └─ v1.0-trainval
```
### Label format
check sample_annotation json file<br />
```
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


## Udacity
### Directory hierarchy
```
root
├─ object-detection-crowdai
│  ├─ {Frame}.jpg
│  ├─ ...
└─ └─ labels.csv
```
### Label format
```
xmin, ymin, xmax, ymax, Frame, Type, Preview URL
```

## KAKAO
### Directory hierarchy
```
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
check frame_annotation json file<br />
bbox_image3d:
```
type, corners(x,y coordinates of the 8 vertices of the cuboid)
```
bbox_pcd3d:
```
type, x, y, z, w, l, h, orientation(quaternion) 
```

### Calibration format
- {camera_name}_translation : {camera_name}'s location to base_link
- {camera_name}_rotation : {camera_name}'s rotation to base_link
- {camera_name}_intrinsic : {camera_name}'s intrinsic matrix
- {LiDAR_name}_translation : {LiDAR_name}'s location to base_link
- {LiDAR_name}_rotation : {LiDAR_name}'s rotation to base_link
### Coordinate system
- Camera : x(right), y(down), z(forward)
- LiDAR : x(forward), y(left), z(up)