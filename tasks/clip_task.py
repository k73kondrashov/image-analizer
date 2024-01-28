import torch
import clip
from images_database.tasks.base import BaseTask
import numpy as np

class ClipTask(BaseTask):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.index_mapping = {}

    def _process_image(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)[0]

        return {'embedding': image_features.cpu().numpy()}

    def _get_index_vectors(self, images_info):
        vectors = []
        for image_info in images_info.values():
            self.index_mapping[len(vectors)] = (image_info.hash, 0)
            vectors.append(image_info.embedding)
        return np.array(vectors)
