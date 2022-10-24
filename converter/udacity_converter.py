import os
import numpy as np
from shutil import copyfile
from tqdm import tqdm
import csv


class Udacity:
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
        self.image = None
        self.labels = []
        print(f'Set Destination Dataset Type {self.dst_db_type}')

    def convert(self):
        img_path = self.src_dir + 'object-detection-crowdai/'
        label_path = self.src_dir + 'object-detection-crowdai/labels.csv'
        filenames = sorted(os.listdir(img_path))
        filenames = [filename.rstrip('.jpg') for filename in filenames]
        label_dict = {}

        with open('data.csv', 'r', encoding='utf-8') as f:
            rdr = csv.reader(f)
            for line in rdr:
                if line[4] in label_dict:
                    label_dict[line[4]].append([*line[:4], line[5]])
                else:
                    label_dict[line[4]] = [*line[:4], line[5]]

        for file in filenames:
            labels = label_dict[file]

            for label in labels:
                x1, y1 = label[:2]
                x2, y2 = label[2:4]
                type = label[-1]

                if self.dst_db_type == 'waymo':
                    width = x2 - x1
                    height = y2 - y1
                    cx = x1 + (width // 2)
                    cy = y1 + (height // 2)
                    # @ TODO: save cx, cy, width, height, type
                else:
                    # @ TODO : save x1, y1, x2, y2, type
                    pass
