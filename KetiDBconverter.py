import os
import argparse
import yaml

from converter import kitti_converter, nuscenes_converter, waymo_converter, udacity_converter, kakao_converter


class KetiDBconverter(object):
    def __init__(self, src_dir: str, tgt_dir: str, db_type: str, config_path: str):
        # parse DB type
        self.db = None
        src_db_type = None
        lst_dir = os.listdir(src_dir)
        if 'image_2' in lst_dir:
            src_db_type = 'kitti'
        elif 'v1.0-mini' in lst_dir:
            src_db_type = 'nuscenes'
        elif 'object-detection-crowdai' in lst_dir:
            src_db_type = 'udacity'
        else:
            src_db_type = 'waymo'

        if src_dir[-1] != '/':
            src_dir += '/'
        self.src_path = src_dir
        self.src_db_type = src_db_type
        assert self.src_db_type is not None, f'Invalid Source Directory.\n "{src_dir}" Please Check.'

        if tgt_dir[-1] != '/':
            tgt_dir += '/'
        self.tgt_path = tgt_dir
        self.tgt_db_type = None
        self.check_db_type(db_type)
        assert self.tgt_db_type is not None, f'Invalid Target Dataset Type.\n "{db_type}" Please Check.'

        self.ret_dir = ['camera', 'lidar', 'label', 'calib']

        # load yaml config file
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            print('config loaded.')

        # get DB directory list
        if self.src_db_type in self.config:
            self.camera_dir_lst = self.config[self.src_db_type]['sensor_name_list']['camera']
            self.lidar_dir_lst = self.config[self.src_db_type]['sensor_name_list']['lidar']
            self.radar_dir_lst = None
            if 'radar' in self.config[self.src_db_type]['sensor_name_list']:
                self.radar_dir_lst = self.config[self.src_db_type]['sensor_name_list']['radar']

        self.create_dir()
        self.load_src_dataset()

    def check_db_type(self, db_type: str):
        check_lst = ['kitti', 'waymo', 'nuscenes', 'udacity']
        if db_type.lower() in check_lst:
            self.tgt_db_type = db_type.lower()

    def create_dir(self):
        if self.src_db_type == 'udacity':
            path = self.tgt_path + 'camera/front'
            if not os.path.isdir(path):
                os.makedirs(path)
            path = self.tgt_path + 'label/front'
            if not os.path.isdir(path):
                os.makedirs(path)
        else:
            for dir_name in self.ret_dir:
                path = self.tgt_path + dir_name
                if not os.path.isdir(path):
                    os.makedirs(path)
            for cam_dir_name in self.camera_dir_lst:
                path = self.tgt_path + 'camera/' + cam_dir_name
                if not os.path.isdir(path):
                    os.makedirs(path)
                path = self.tgt_path + 'label/' + cam_dir_name
                if not os.path.isdir(path):
                    os.makedirs(path)
                path = self.tgt_path + 'calib/' + cam_dir_name
                if not os.path.isdir(path):
                    os.makedirs(path)
            for lid_dir_name in self.lidar_dir_lst:
                path = self.tgt_path + 'lidar/' + lid_dir_name
                if not os.path.isdir(path):
                    os.makedirs(path)
            if self.radar_dir_lst is not None:
                for radar_dir_name in self.radar_dir_lst:
                    path = self.tgt_path + 'calib/' + radar_dir_name
                    if not os.path.isdir(path):
                        os.makedirs(path)
                    path = self.tgt_path + 'radar/' + radar_dir_name
                    if not os.path.isdir(path):
                        os.makedirs(path)

    def load_src_dataset(self):
        module = None
        if self.src_db_type == 'kitti':
            module = kitti_converter
        elif self.src_db_type == 'waymo':
            module = waymo_converter
        elif self.src_db_type == 'nuscenes':
            module = nuscenes_converter
        elif self.src_db_type == 'udacity':
            module = udacity_converter
        elif self.src_db_type == 'kakao':
            module = kakao_converter
        self.db = getattr(module, self.src_db_type)(self.src_path, self.tgt_path, self.tgt_db_type)

    def convert(self):
        convert_func = getattr(self.db, 'convert')
        convert_func()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_db_dir', help='Directory to load Dataset')
    parser.add_argument('--tgt_db_dir', help='Directory to save converted Dataset')
    parser.add_argument('--tgt_db_type', help='Dataset type to convert [KITTI, Waymo, Nuscenes, Udacity]')
    parser.add_argument('--config_path', default='db_infos.yaml', help='Dataset configuration yaml file path')
    args = parser.parse_args()

    print(args.src_db_dir)

    converter = KetiDBconverter(args.src_db_dir, args.tgt_db_dir, args.tgt_db_type, args.config_path)
    converter.convert()

    print('Dataset Convert has Done.')
