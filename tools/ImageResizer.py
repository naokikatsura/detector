import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

INPUT_DIR = "C:/Users/naoki/DETECTOR/data/validation/positive_formatted/"
SIZE = 320
RESIZE = 128
CH = 3

def show_image(img, title=""):
    plt.imshow(img.numpy().astype(np.uint8))
    plt.title(title)
    plt.axis("off")
    plt.show()

def resize_image(path):

    # 画像読み込み
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=CH)

    show_image(img, "original")

    img = tf.image.resize(
        img,
        (RESIZE, RESIZE),
        method="area"
    )

    show_image(img, "resized")

    return img


for filename in os.listdir(INPUT_DIR)[:10]:

    if not filename.lower().endswith(".png"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)

    img = resize_image(input_path)

print("完了")