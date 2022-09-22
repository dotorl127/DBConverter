# KetiDBconverter
for convert open source datasets KITTI, NuScenes, Waymo

## Related work
- [x] convert coordinates system among KITTI, Waymo, Nuscenes
- [x] extract sensor data in tfrecord file
- [x] convert label format, class names for each dataset
- [x] visualization point cloud with 3D label
- [x] visualization image with 2D label

## KetiDBconverter directory hierarchy
```
KetiDBconverter
├─ converter
│  ├─ kitti_converter.py
│  ├─ nuscenes_converter.py
│  └─ waymo_converter.py
├─ dictionary
│  ├─ class_dictionary.py
│  └─ rotation_dictionary.py
├─ utils
│  ├─ label.py
│  ├─ util.py
│  └─ visulize.py
├─ db_infos.yaml
├─ damo.py
├─ KetiDBconverter.py
├─ README.md
└─ requirements.txt
```
- {dataset name}_converter : 

## How to use
### Convert
```
python KetiDBconverter.py --src_db_dir {source dataset path to load} --tgt_db_dir {target dataset path to save} --tgt_db_type {dataset name to convert[kitti, waymo, nuscenes]}
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
type, truncated, occluded, alpha, bbox(left, top, right, bottom), dimensions(height, width, lengh), localtion(x, y, z), rotation_y
### Coordinate system
Camera : x(right), y(bottom), z(forward)<br />
LiDAR : x(forward), y(left), z(up)

## WAYMO
### Directory hierarchy
```
root
└─ *.tfrecord
```
### Label format
check [proto file](https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto)
### Coordinate system
Camera : x(forward), y(left), z(up)<br />
LiDAR : x(forward), y(left), z(up)

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
token, sample_token, instance_token, attribute_tokens, visibility_token, translation(center x, y, z), size(width, length, height), rotation(quaternion), num_lidar_pts, num_radar_pts, next, prev
### Coordinate system
Camera : x(right), y(down), z(forward)<br />
LiDAR : x(right), y(forward), z(up)