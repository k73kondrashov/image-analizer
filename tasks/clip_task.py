from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from images_database.tasks.base import BaseTask


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

    def find_by_text(self, text, score_idx=1):

        text = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text).cpu().numpy()

        res = self.index.search(text_features, score_idx)
        return self.index_mapping[res[1][0, -1]][0]

    def find_by_image(self, source, score_idx=1):

        if isinstance(source, (str, Path)):
            image = Image.open(source)
        elif isinstance(source, np.ndarray):
            #TODO Add dim checker
            image = Image.fromarray(source)
        elif isinstance(source, Image.Image):
            image = source
        else:
            raise TypeError(f"source must be a path to image, np.array or PIL.Image, got {type(source)}")

        image_emb = self._process_image(image)['embedding']
        res = self.index.search(image_emb[None, ...], score_idx)
        return self.index_mapping[res[1][0, -1]][0]
