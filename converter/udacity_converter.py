import os
import numpy as np
from shutil import copyfile
from tqdm import tqdm

from utils import label


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
        self.calib_dict = {}
        self.image = None
        self.labels = []
        print(f'Set Destination Dataset Type {self.dst_db_type}')

    def convert(self):

