import os
import os.path as osp
import numpy as np
from shutil import copyfile
from tqdm import tqdm

from dictionary.class_dictionary import kakao_dict
from dictionary.rotation_dictionary import cam_rot, lid_rot
from scipy.spatial.transform import Rotation
from utils.util import check_valid_mat


class kakao_parser:
    def __init__(self,
                 src_dir: str = None):
        self.src_dir = src_dir
        self.table_root = osp.join(self.src_dir, 'meta')
        self.table_names = ['dataset', 'ego_pose', 'frame', 'frame_annotation',
                            'frame_data', 'instance', 'log', 'sensor']
        self.dataset = self.__load_table__('dataset')
        self.ego_pose = self.__load_table__('ego_pose')
        self.frame = self.__load_table__('frame')
        self.frame_annotation = self.__load_table__('frame_annotation')
        self.frame_data = self.__load_table__('frame_data')
        self.instance = self.__load_table__('instance')
        self.log = self.__load_table__('log')
        self.sensor = self.__load_table__('sensor')
        self.__make_reverse_index__()

    def __load_table__(self, table_name) -> dict:
        with open(osp.join(self.table_root, '{}.json'.format(table_name))) as f:
            table = json.load(f)
        return table

    def __make_reverse_index__(self) -> None:
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            table_ = getattr(self, table)
            if isinstance(table_, dict):
                table_ = [table_]

            for ind, member in enumerate(table_):
                self._token2ind[table][member['uuid']] = ind

        # Decorate (adds short-cut) sample_annotation table with for category name.
        for record in self.frame_annotation:
            inst = self.get('instance', record['instance_uuid'])
            record['category_name'] = inst['category_name']

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.frame_data:
            sensor_record = self.get('sensor', record['sensor_uuid'])
            record['name'] = sensor_record['name']
            record['type'] = sensor_record['type']

        # Reverse-index samples with sample_data and annotations.
        for record in self.frame:
            record['data'] = {}
            record['anns'] = []

        for record in self.frame_data:
            if record['is_key_frame']:
                sample_record = self.get('frame', record['frame_uuid'])
                sample_record['data'][record['name']] = record['uuid']

        for ann_record in self.frame_annotation:
            frame_data_record = self.get('frame_data', ann_record['frame_data_uuid'])
            sample_record = self.get('frame', frame_data_record['frame_uuid'])
            sample_record['anns'].append(ann_record['uuid'])

    def get(self, table_name: str, token: str) -> dict:
        assert table_name in self.table_names, "Table {} not found".format(table_name)
        return getattr(self, table_name)[self.getind(table_name, token)]

    def getind(self, table_name: str, token: str) -> int:
        return self._token2ind[table_name][token]


def quat2euler(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z


class kakao:
    def __init__(self,
                 src_dir: str = None,
                 dst_dir: str = None,
                 dst_db_type: str = None):
        assert src_dir is not None or dst_dir is not None or dst_db_type is not None, \
            f'Invalid Parameter Please Check {src_dir}\n{dst_dir}\n{dst_db_type}'

        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.dst_db_type = dst_db_type
        self.kakaodb = kakao_parser(self.src_dir)
        print(f'Set Destination Dataset Type {self.dst_db_type}')

    def convert(self):
        print(f'Convert kakao to {self.dst_db_type} Dataset.')

        for frame in self.kakaodb.frame:
            for anno_uuid in frame['anns']:
                frame_annotation = self.kakaodb.get('frame_annotation', anno_uuid)
                frame_data = self.kakaodb.get('frame_data', frame_annotation['frame_data_uuid'])
                sensor_data = self.kakaodb.get('sensor', frame_data['sensor_uuid'])
                ego_pose_data = self.kakaodb.get('ego_pose', frame_data['ego_pose_uuid'])

                # TODO: save sensor calibration data
                # TODO: convert coordinates system
                # TODO: save sensor raw data

                if 'kitti' in self.dst_db_type:
                    cls = kakao_dict[f'to_kitti'][frame_annotation['category_name']]

                    with open(f'{osp.join(self.dst_dir, "labels", frame_data["name"])}', 'w') as f:
                        if frame_annotation['annotation_type_name'] == 'bbox_pcd3d':
                            xyz = ', '.join(frame_annotation['geometry']['center'])
                            wlh = ', '.join(frame_annotation['geometry']['wlh'])
                            _, _, yaw = quat2euler(*frame_annotation['geometry']['orientation'])
                            f.write(f'{cls}, {xyz}, {wlh}, {yaw}\n')
                        elif frame_annotation['annotation_type_name'] == 'bbox_image3d':
                            x1 = min(frame_annotation['geometry']['corners'][0])
                            y1 = min(frame_annotation['geometry']['corners'][1])
                            x2 = max(frame_annotation['geometry']['corners'][0])
                            y2 = max(frame_annotation['geometry']['corners'][1])
                            f.write(f'{cls}, {x1}, {y1}, {x2}, {y2}\n')
                elif self.dst_db_type == 'waymo':
                    pass
                elif self.dst_db_type == 'nuscenes':
                    pass
                elif self.dst_db_type == 'udacity':
                    pass
