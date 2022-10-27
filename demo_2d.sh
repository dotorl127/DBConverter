#! /bin/bash
{
python demo.py --root_dir ./WORKSPACE/converted_dataset/kitti_to_nuscenes --dataset_type nuscenes --vis_type 2d;
python demo.py --root_dir ./WORKSPACE/converted_dataset/kitti_to_waymo --dataset_type waymo --vis_type 2d;
python demo.py --root_dir ./WORKSPACE/converted_dataset/kitti_to_udacity --dataset_type udacity --vis_type 2d;

python demo.py --root_dir ./WORKSPACE/converted_dataset/waymo_to_kitti --dataset_type kitti --vis_type 2d;
python demo.py --root_dir ./WORKSPACE/converted_dataset/waymo_to_nuscenes --dataset_type nuscenes --vis_type 2d;
python demo.py --root_dir ./WORKSPACE/converted_dataset/waymo_to_udacity --dataset_type udacity --vis_type 2d;

python demo.py --root_dir ./WORKSPACE/converted_dataset/nuscenes_to_kitti --dataset_type kitti --vis_type 2d;
python demo.py --root_dir ./WORKSPACE/converted_dataset/nuscenes_to_waymo --dataset_type waymo --vis_type 2d;
python demo.py --root_dir ./WORKSPACE/converted_dataset/nuscenes_to_udacity --dataset_type udacity --vis_type 2d;

python demo.py --root_dir ./WORKSPACE/converted_dataset/udacity_to_kitti --dataset_type kitti --vis_type 2d;
python demo.py --root_dir ./WORKSPACE/converted_dataset/udacity_to_waymo --dataset_type waymo --vis_type 2d;
python demo.py --root_dir ./WORKSPACE/converted_dataset/udacity_to_nuscenes --dataset_type nuscenes --vis_type 2d;
}
