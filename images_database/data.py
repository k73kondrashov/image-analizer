import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError
from tqdm import tqdm

IMGS_EXT = ['.jpeg', '.jpg', '.png']


@dataclass
class ImageInfo:
    path: Path
    hash: str
    embedding: np.ndarray = None
    faces: List = None
    caption: str = None
    text: List = None


class Database:
    def __init__(self, images_dir):

        self.images_dir = Path(images_dir)
        self.corrupted_files = set()

        self.images_list = self._collect_images_path()
        self.images = {}
        self.name2hash = {}

    def _collect_images_path(self):
        return [path for path in self.images_dir.rglob('*') if path.suffix.lower() in IMGS_EXT]

    def _check_corrupted_files(self):
        for path in tqdm(self.images_list, desc="Checking corrupted files"):
            try:
                Image.open(path)
            except UnidentifiedImageError:
                print(path)
                self.corrupted_files.add(path)

    def save(self, path):
        with open(path, 'wb') as file_:
            pickle.dump(self, file_)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file_:
            data = pickle.load(file_)
        return data




def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
