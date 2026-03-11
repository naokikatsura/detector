import os
import numpy as np
from PIL import Image

INPUT_DIR = "C:/Users/naoki/DETECTOR/data/validation/negative/"
OUTPUT_DIR = "C:/Users/naoki/DETECTOR/data/validation/negative_formatted/"
SIZE = 320

os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_image(path, size=320):

    img = Image.open(path).convert("RGB")
    img = np.array(img)

    h, w, c = img.shape

    # -------------------------
    # 小さい場合 → reflect padding
    # -------------------------
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)

    if pad_h > 0 or pad_w > 0:

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="reflect"
        )

        h, w, _ = img.shape

    # -------------------------
    # 大きい場合 → 中央切り出し
    # -------------------------
    start_y = (h - size) // 2
    start_x = (w - size) // 2

    img = img[start_y:start_y+size, start_x:start_x+size]

    return img


for filename in os.listdir(INPUT_DIR):

    if not filename.lower().endswith(".png"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    img = normalize_image(input_path, SIZE)

    Image.fromarray(img).save(output_path)

    print("saved:", output_path)

print("完了")