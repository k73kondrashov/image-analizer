from dataclasses import dataclass
from pathlib import Path
from typing import List
import hashlib
import pickle
from PIL import Image
from PIL import UnidentifiedImageError
from tqdm import tqdm
from images_database.tasks.ocr_task import OCRTask
from images_database.tasks.clip_task import ClipTask
from images_database.tasks.caption_task import CaptionTask
from images_database.tasks.face_task import FaceTask

import numpy as np

IMGS_EXT = ['.jpeg', '.jpg', '.png']

TASKS_DICT = {
    'face': FaceTask,
    'embedding': ClipTask,
    'text': OCRTask,
    'caption': CaptionTask
}


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
        self._check_corrupted_files()
        self.images = {}
        name2hash = {}

        self.tasks = {task_name: task() for task_name, task in TASKS_DICT.items()}

    def _collect_images_path(self):
        return [path for path in self.images_dir.rglob('*') if path.suffix.lower() in IMGS_EXT]

    def _check_corrupted_files(self):
        for path in tqdm(self.images_list, desc="Checking corrupted files"):
            try:
                Image.open(path)
            except UnidentifiedImageError:
                print(path)
                self.corrupted_files.add(path)

    def process_images_dir(self):
        for path in tqdm(self.images_list):
            if path in self.corrupted_files:
                continue
            hash_value = md5(path)
            image = Image.open(path).convert("RGB")
            extracted_info = {}
            for task_name, task in self.tasks.items():
                extracted_info.update(task._process_image(image))

            img_info = ImageInfo(
                path=path,
                hash=hash_value,
                **extracted_info
            )
            self.images[hash_value] = img_info

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
