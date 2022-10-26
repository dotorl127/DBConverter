import os
from shutil import copyfile
from tqdm import tqdm
import csv
from dictionary.class_dictionary import udacity_dict


class udacity:
    def __init__(self,
                 src_dir: str = None,
                 dst_dir: str = None,
                 dst_db_type: str = None):
        """
        :param src_dir: something.
        :param dst_dir: something.
        """
        assert src_dir is not None or dst_dir is not None or dst_db_type is not None, \
            f'Invalid Parameter Please Check {src_dir}\n{dst_dir}\n{dst_db_type}'

        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.dst_db_type = dst_db_type
        print(f'Set Destination Dataset Type {self.dst_db_type}')

    def convert(self):
        img_path = self.src_dir + 'object-detection-crowdai/'
        label_path = self.src_dir + 'object-detection-crowdai/labels.csv'
        filenames = sorted(os.listdir(img_path))
        filenames = [filename.rstrip('.jpg') for filename in filenames]
        label_dict = {}

        with open(label_path, 'r', encoding='utf-8') as f:
            rdr = csv.reader(f)
            for line in rdr:
                file = line[4].rstrip('.jpg')
                if file in label_dict:
                    label_dict[file].append([*line[:4], line[5]])
                else:
                    label_dict[file] = [[*line[:4], line[5]]]

        for frame, file in enumerate(tqdm(filenames[:10])):
            copyfile(f'{img_path}{file}.jpg', f'{self.dst_dir}camera/front/{frame:06d}.jpg')

            labels = label_dict[file]
            lines = ''

            for label in labels:
                x1, y1, x2, y2 = map(int, label[:4])
                type = udacity_dict[f'to_{self.dst_db_type}'][label[4]]
                line = ''

                if self.dst_db_type == 'kitti':
                    line = f'{type}, 0, 0, -10, ' \
                           f'{x1}, {y1}, {x2}, {y2}, ' \
                           f'0, 0, 0, 0, 0, 0, 0\n'
                elif self.dst_db_type == 'waymo':
                    width = x2 - x1
                    height = y2 - y1
                    cnt_x = x1 + width // 2
                    cnt_y = y1 + height // 2
                    line = f'{cnt_x}, {cnt_y}, {width}, {height}, ' \
                           f'0, 0, {type}, -1, ' \
                           f'0, 0, 0, 0, 0, 0, 0\n'
                elif self.dst_db_type == 'nuscenes':
                    line = f'{type}, 0, 0, 0, 0, 0, 0, ' \
                           f'0, 0, 0, 0, {-1}, {-1}, ' \
                           f'{x1}, {y1}, {x2}, {y2}\n'

                lines += line

            with open(f'{self.dst_dir}label/front/{frame:06d}.txt', 'w') as f:
                f.write(lines)
