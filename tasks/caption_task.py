from transformers import pipeline
from images_database.tasks.base import BaseTask


class CaptionTask(BaseTask):
    def __init__(self):
        self.captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0)

    def _process_image(self, image):
        result = self.captioner(image)[0]
        return {'caption': result['generated_text']}
