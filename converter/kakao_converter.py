import os
import os.path as osp
import numpy as np
from shutil import copyfile
from tqdm import tqdm

from dictionary.class_dictionary import kakao_dict
from dictionary.rotation_dictionary import cam_rot, lid_rot
from scipy.spatial.transform import Rotation
from utils.util import check_valid_mat

from pyquaternion import Quaternion as Q


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
        self.cam_rot = cam_rot['kakao'][dst_db_type]
        self.cam_rot = check_valid_mat(self.cam_rot)
        self.lid_rot = lid_rot['kakao'][dst_db_type]
        self.lid_rot = check_valid_mat(self.lid_rot)
        print(f'Set Destination Dataset Type {self.dst_db_type}')

    def convert(self):
        print(f'Convert kakao to {self.dst_db_type} Dataset.')

        for sensor in self.kakaodb.sensor:
            calib_file_path = osp.join(self.dst_dir, 'calibration', sensor['type'],
                                       sensor['name'] + '.txt')
            if not osp.exists(calib_file_path):
                with open(calib_file_path, 'r') as f:
                    loc = list(map(float, sensor['translation']))
                    rot = list(map(float, sensor['rotation']))
                    intrinsic = sensor['intrinsic']['parameter']
                    extrinsic = np.zeros((3, 4))
                    extrinsic[:3, :3] = Q(*rot).rotation_matrix
                    extrinsic[:, 3] = loc
                    f.write(f'{sensor["name"]}_intrinsic : {intrinsic}\n')
                    f.write(f'{sensor["name"]}_extrinsic : {extrinsic}\n')

        for frame in tqdm(self.kakaodb.frame):
            for anno_uuid in frame['anns']:
                frame_annotation = self.kakaodb.get('frame_annotation', anno_uuid)
                frame_data = self.kakaodb.get('frame_data', frame_annotation['frame_data_uuid'])
                sensor_data = self.kakaodb.get('sensor', frame_data['sensor_uuid'])
                ego_pose_data = self.kakaodb.get('ego_pose', frame_data['ego_pose_uuid'])

                f_name = f'{frame_data["file_name"]}.{frame["file_format"]}'
                src_path = osp.join(self.src_dir, 'sensor', frame_data['name'], f_name)
                dst_path = osp.join(self.dst_dir, frame_data['type'], frame_data['name'], f_name)
                copyfile(src_path, dst_path)


                x1, y1, x2, y2 = -1, -1, -1, -1
                x, y, z = 'None', 'None', 'None'
                w, l, h = 'None', 'None', 'None'
                yaw = 'None'

                if frame_annotation['annotation_type_name'] == 'bbox_pcd3d':
                    x, y, z = list(map(float, frame_annotation['geometry']['center']))
                    w, l, h = list(map(float, frame_annotation['geometry']['wlh']))
                    _, _, yaw = quat2euler(*frame_annotation['geometry']['orientation'])

                    if self.dst_db_type == 'kitti':
                        z -= h / 2
                        rot -= np.pi / 2

                    # project bounding box to the virtual reference frame
                    if self.dst_db_type == 'kitti':
                        x, y, z, _ = self.cam_rot @ np.array([x, y, z, 1]).T
                    else:
                        x, y, z, _ = self.lid_rot @ np.array([x, y, z, 1]).T
                elif frame_annotation['annotation_type_name'] == 'bbox_image3d':
                    x, y = frame_annotation['geometry']['corners']
                    x1 = min(x)
                    y1 = min(y)
                    x2 = max(x)
                    y2 = max(y)

                if self.dst_db_type != 'udacity' and yaw != 'None':
                    with open(f'{osp.join(self.dst_dir, "labels", frame_data["name"])}', 'a') as f:
                        if self.dst_db_type == 'kitti' and yaw != 'None':
                            cls = kakao_dict[f'to_kitti'][frame_annotation['category_name']]
                            f.write(f'{cls}, 0, 0, -10, {x1}, {y1}, {x2}, {y2}, {x}, {y}, {z}, {h}, {w}, {l}, {yaw}\n')
                        elif self.dst_db_type == 'waymo':
                            cls = kakao_dict[f'to_waymo'][frame_annotation['category_name']]
                            w = x2 - x1
                            h = y2 - y1
                            cx = x1 + w / 2
                            cy = y1 + h / 2
                            f.write(f'{cx}, {cy}, {w}, {h}, 0, 0, {cls}, -1, '
                                    f'{x}, {y}, {z}, {w}, {l}, {h}, {yaw}\n')
                        elif self.dst_db_type == 'nuscenes':
                            cls = kakao_dict[f'to_nuscenes'][frame_annotation['category_name']]
                            rot = Rotation.from_euler('xyz', [0, 0, rot])
                            rot_quat = rot.as_quat()
                            f.write(f'{cls}, {x}, {y}, {z}, {w}, {h}, {l}, '
                                    f'{rot_quat[0]}, {rot_quat[1]}, {rot_quat[2]}, {rot_quat[3]}, '
                                    f'0, 0, -1, -1, -1, -1\n')
                elif self.dst_db_type == 'udacity' and (x1, y1, x2, y2) != (-1, -1, -1, -1):
                    with open(f'{osp.join(self.dst_dir, "labels", frame_data["name"])}', 'a') as f:
                        cls = kakao_dict[f'to_udacity'][frame_annotation['category_name']]
                        f.write(f'{x1}, {y1}, {x2}, {y2}, {cls}\n')
