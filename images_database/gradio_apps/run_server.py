import pickle
import numpy as np
import gradio as gr
from images_database.data import Database
from images_database.gradio_apps.image_search_app import ImageSearchApp
from images_database.gradio_apps.face_app import FaceApp
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser


class InferTask:
    def __init__(self, database, name2path):
        self.database = database
        self.search_app = ImageSearchApp(database, name2path)
        self.face_app = FaceApp(database, name2path)

    def run(self):
        with gr.Blocks() as demo:
            with gr.Tab("Image search"):
                self.search_app.app()
            with gr.Tab("Face search"):
                self.face_app.app()
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

    app = InferTask(database, name2path)
    app.run()
