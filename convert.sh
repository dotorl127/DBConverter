#! /bin/bash
{
python KetiDBconverter.py --src_db_dir ./WORKSPACE/original_dataset/kitti/training --tgt_db_dir ./WORKSPACE/converted_dataset/kitti_to_waymo --tgt_db_type waymo;
python KetiDBconverter.py --src_db_dir ./WORKSPACE/original_dataset/kitti/training --tgt_db_dir ./WORKSPACE/converted_dataset/kitti_to_nuscenes --tgt_db_type nuscenes;
python KetiDBconverter.py --src_db_dir ./WORKSPACE/original_dataset/kitti/training --tgt_db_dir ./WORKSPACE/converted_dataset/kitti_to_udacity --tgt_db_type udacity;

python KetiDBconverter.py --src_db_dir ./WORKSPACE/original_dataset/waymo --tgt_db_dir ./WORKSPACE/converted_dataset/waymo_to_kitti --tgt_db_type kitti;
python KetiDBconverter.py --src_db_dir ./WORKSPACE/original_dataset/waymo --tgt_db_dir ./WORKSPACE/converted_dataset/waymo_to_nuscenes --tgt_db_type nuscenes;
python KetiDBconverter.py --src_db_dir ./WORKSPACE/original_dataset/waymo --tgt_db_dir ./WORKSPACE/converted_dataset/waymo_to_udacity --tgt_db_type udacity;

python KetiDBconverter.py --src_db_dir ./WORKSPACE/original_dataset/nuscenes/v1.0-mini --tgt_db_dir ./WORKSPACE/converted_dataset/nuscenes_to_kitti --tgt_db_type kitti;
python KetiDBconverter.py --src_db_dir ./WORKSPACE/original_dataset/nuscenes/v1.0-mini --tgt_db_dir ./WORKSPACE/converted_dataset/nuscenes_to_waymo --tgt_db_type waymo;
python KetiDBconverter.py --src_db_dir ./WORKSPACE/original_dataset/nuscenes/v1.0-mini --tgt_db_dir ./WORKSPACE/converted_dataset/nuscenes_to_udacity --tgt_db_type udacity;

python KetiDBconverter.py --src_db_dir ./WORKSPACE/original_dataset/udacity --tgt_db_dir ./WORKSPACE/converted_dataset/udacity_to_kitti --tgt_db_type kitti;
python KetiDBconverter.py --src_db_dir ./WORKSPACE/original_dataset/udacity --tgt_db_dir ./WORKSPACE/converted_dataset/udacity_to_waymo --tgt_db_type waymo;
python KetiDBconverter.py --src_db_dir ./WORKSPACE/original_dataset/udacity --tgt_db_dir ./WORKSPACE/converted_dataset/udacity_to_nuscenes --tgt_db_type nuscenes;
}
