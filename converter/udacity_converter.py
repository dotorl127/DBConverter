import os
from shutil import copyfile
from tqdm import tqdm
import csv


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

        for frame, file in enumerate(tqdm(filenames)):
            copyfile(f'{img_path}{file}.jpg', f'{self.dst_dir}front_camera/{frame:06d}.jpg')

            labels = label_dict[file]

            for label in labels:
                x1, y1, x2, y2 = label[:4]
                type = label[4]
                line = f'{x1}, {y1}, {x2}, {y2}, {type}\n'

                if self.dst_db_type == 'waymo':
                    width = x2 - x1
                    height = y2 - y1
                    cx = x1 + (width // 2)
                    cy = y1 + (height // 2)
                    line = f'{cx}, {cy}, {width}, {height}, {type}\n'

                with open(f'{self.dst_dir}label/{frame:06d}.txt', 'w') as f:
                    f.write(line)
