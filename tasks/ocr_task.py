import easyocr
import numpy as np


class OCRTask:
    def __init__(self):
        self.reader = easyocr.Reader(['ru', 'en'])

    def _process_image(self, image):
        result = self.reader.readtext(np.array(image))
        return {'text': result}
