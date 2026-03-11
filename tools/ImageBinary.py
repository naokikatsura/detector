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

def binary_image(path):

    # 画像読み込み
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=CH)

    show_image(img, "original")

    # グレースケール
    img = tf.image.rgb_to_grayscale(img)

    show_image(img, "binary")

    # float化
    img = tf.cast(img, tf.float32) / 255.0

    # Sobel edge
    edges = tf.image.sobel_edges(img[None])

    gx = edges[0,:,:,0,0]
    gy = edges[0,:,:,0,1]

    edge_mag = tf.sqrt(gx**2 + gy**2)

    edge_mag = edge_mag / tf.reduce_max(edge_mag) * 255

    show_image(edge_mag, "edge")

    # 2値化
    threshold = 40
    binary = tf.where(edge_mag > threshold, 255.0, 0.0)

    show_image(binary, "binary")

    return binary


for filename in os.listdir(INPUT_DIR)[20:30]:

    if not filename.lower().endswith(".png"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)

    img = binary_image(input_path)

print("完了")