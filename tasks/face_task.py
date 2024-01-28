import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from images_database.tasks.base import BaseTask
import numpy as np


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
