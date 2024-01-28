import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from images_database.tasks.base import BaseTask
import numpy as np
from PIL import ImageDraw, Image


class FaceTask(BaseTask):
    def __init__(self):

        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.index_mapping = {}

    def _process_image(self, image):
        faces = self.app.get(np.array(image))

        return {'faces': [face.__dict__ for face in faces]}

    def _get_index_vectors(self, images_info):
        vectors = []
        for image_info in images_info.values():
            for idx, face in enumerate(image_info.faces):
                self.index_mapping[len(vectors)] = (image_info.hash, idx)
                vectors.append(face['embedding'])
        return np.array(vectors)

    @staticmethod
    def draw(image_info, face_idx=None):
        img = Image.open(image_info.path)
        img_draw = ImageDraw.Draw(img)
        faces = image_info.faces if face_idx is None else [image_info.faces[face_idx]]
        for face in faces:
            img_draw.rectangle(face['bbox'].tolist(), outline=(255, 0, 0), width=max(img.size) // 200)
        return img

    def draw_match(self, image1, idx1, image2, idx2):

        img1 = self.draw(image1, idx1)
        img2 = self.draw(image2, idx2)
        return img1, img2
