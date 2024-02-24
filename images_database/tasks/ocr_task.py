import easyocr
import numpy as np
from images_database.tasks.base import BaseTask


class OCRTask(BaseTask):
    def __init__(self):
        self.reader = easyocr.Reader(['ru', 'en'])

    def _process_image(self, image):
        result = self.reader.readtext(np.array(image))
        return {'text': result}
