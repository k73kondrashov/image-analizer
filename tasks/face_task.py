import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import numpy as np


class FaceTask:
    def __init__(self):

        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def _process_image(self, image):
        faces = self.app.get(np.array(image))

        return {'faces': [face.__dict__ for face in faces]}
