import os
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


