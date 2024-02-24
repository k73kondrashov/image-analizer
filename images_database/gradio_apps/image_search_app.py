import numpy as np
import gradio as gr
from images_database.data import Database
from images_database.tasks.clip_task import ClipTask
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser


class ImageSearchApp:
    def __init__(self, database, name2path):

        self.database = database
        self.name2path = name2path
        self.clip_task = ClipTask()
        self.clip_task.build_index(database.images)
        self.count = 1

    def _next_image(self, input_image, text):
        self.count += 1
        return self._predict(input_image, text)

    def _prev_image(self, input_image, text):
        if self.count >= 2:
            self.count -= 1

        return self._predict(input_image, text)

    def _predict(self, input_img, text):
        print(">>>>", text, self.count)
        if text:
            hash_ = self.clip_task.find_by_text(text, self.count)
            return (np.array(
                Image.open(self.name2path[self.database.images[hash_].path.name])),
                str(self.name2path[self.database.images[hash_].path.name]))
        elif input_img is not None:
            hash_ = self.clip_task.find_by_image(input_img, self.count)
            return (np.array(
                Image.open(self.name2path[self.database.images[hash_].path.name])),
                str(self.name2path[self.database.images[hash_].path.name]))

        return np.zeros((512, 512, 3), dtype=np.uint8), ""

    def app(self):
        with gr.Row():
            inp_img = gr.Image(height=400, width=512)
            out_img = gr.Image(height=400, width=512)
        with gr.Row():
            query = gr.Text(label="Query")
            out_text = gr.Text(label="Path")
        with gr.Row():
            clear_btn = gr.Button("Clear")
            submit_btn = gr.Button("Submit")
            prev_btn = gr.Button("prev")
            next_btn = gr.Button("next")
            submit_btn.click(self._predict, [inp_img, query], [out_img, out_text])
            prev_btn.click(self._prev_image, [inp_img, query], [out_img, out_text])
            next_btn.click(self._next_image, [inp_img, query], [out_img, out_text])
                # clear_btn.click(self.clear, inp_img, inp_img)
