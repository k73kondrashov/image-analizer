import pickle
import sys
sys.path.append('/home/nikolay/Projects/')
import numpy as np
import gradio as gr
from images_database.data import Database
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser


class InferTask:
    def __init__(self):
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
            hash_ = database.tasks['embedding'].find_by_text(text, self.count)
            return np.array(Image.open(name2path[database.images[hash_].path.name])), str(name2path[database.images[hash_].path.name])
        elif input_img is not None:
            hash_ = database.tasks['embedding'].find_by_image(input_img, self.count)
            return np.array(Image.open(name2path[database.images[hash_].path.name])), str(name2path[database.images[hash_].path.name])

        return np.zeros((512, 512, 3), dtype=np.uint8), ""

    def run(self):
        with gr.Blocks() as demo:
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
            demo.launch()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--images_dir", type=str)
    parser.add_argument("--db_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    database = Database(args.images_dir)
    with open(args.db_path, 'rb') as f:
        database.images = pickle.load(f)

    name2path = {path.name: path for path in Path(args.images_dir).glob('*')}

    database.tasks['embedding'].build_index(database.images)

    app = InferTask()
    app.run()
