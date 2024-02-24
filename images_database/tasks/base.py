from abc import abstractmethod

import faiss
import numpy as np


class BaseTask:
    def __init__(self):
        self.index = None
    
    def process_image(self, image: np.ndarray) -> object:
        pass

    @abstractmethod
    def _get_index_vectors(self, images_info):
        """Find target vectors in database"""

    def build_index(self, images_info):
        vectors = self._get_index_vectors(images_info)
        # if len(vectors.shape) == 1:
        #     vectors = [None, ...]
        _, dim = vectors.shape

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
