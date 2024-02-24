import numpy as np
import gradio as gr
from pathlib import Path
import pickle
from PIL import Image, ImageDraw
from images_database.tasks.face_task import FaceTask
from images_database.data import Database

class FaceApp:
    def __init__(self, database, name2path):
        self.database = database
        self.name2path = name2path
        self.input_box = []
        self.orig_image = None
        self.face_task = FaceTask()
        self.face_task.build_index(database.images)

        self.count = 1

    def _next_image(self, input_image):
        self.count += 1
        return self._predict(input_image)

    def _prev_image(self, input_image):
        if self.count >= 2:
            self.count -= 1

        return self._predict(input_image)

    def chose_box(self, img, evt: gr.SelectData):
        if len(self.input_box) == 4:
            self.input_box = []

        if self.orig_image is None:
            self.orig_image = img
        self.input_box.extend(evt.index)

        if len(self.input_box) == 4:
            box = np.array(self.input_box).reshape((2, 2))
            box = np.concatenate([box.min(0), box.max(0)]).tolist()
            self.draw_box(img, box)
            return img
        return self.orig_image

    def draw_box(self, img, box):
        draw = ImageDraw.Draw(img)
        draw.rectangle(box, outline=(255, 0, 0), width=2)

    def _predict(self, img):
        face_data = self.face_task._process_image(img)
        print(face_data)
        hash, idx = self.face_task.find(face_data["faces"][0]['embedding'], self.count)
        print(hash, idx)
        img = Image.open(self.name2path[self.database.images[hash].path.name])
        return img, "ll"

    def app(self):
        with gr.Row():
            inp_img = gr.Image(type="pil", height=400, width=512)
            out_img = gr.Image(height=400, width=512)
            inp_img.select(self.chose_box, inp_img, inp_img)
        with gr.Row():
            query = gr.Text(label="Query")
            out_text = gr.Text(label="Path")
        with gr.Row():
            clear_btn = gr.Button("Clear")
            submit_btn = gr.Button("Submit")
            prev_btn = gr.Button("prev")
            next_btn = gr.Button("next")

            submit_btn.click(self._predict, inp_img, [out_img, out_text])
            prev_btn.click(self._prev_image, inp_img, [out_img, out_text])
            next_btn.click(self._next_image, inp_img, [out_img, out_text])

if __name__ == "__main__":
    database = Database("/home/nikolay/SOFT/VK-SavedImagesDownloader/Alina_saved_images/")
    with open("images.pickle", 'rb') as f:
        database.images = pickle.load(f)
    name2path = {path.name: path for path in Path("/home/nikolay/SOFT/VK-SavedImagesDownloader/Alina_saved_images/").glob('*')}
    face_app = FaceApp(database, name2path)
    with gr.Blocks() as demo:
        with gr.Tab("Image search"):
            face_app.app()
            # clear_btn.click(self.clear, inp_img, inp_img)
        demo.launch()