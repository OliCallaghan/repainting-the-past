from PIL import Image
import os
import sys

# Import settings
IM_WIDTH = 224
IM_HEIGHT = 224
DATA_DIR = 'resources'
IM_PER_BIN = 100

for filename in os.listdir(DATA_DIR):
    print(filename)

    # Load image
    jpeg_file = Image.open(os.path.join(DATA_DIR, filename))
    # Crop to correct size
    shorter_side = min(jpeg_file.size)
    x_cut = (jpeg_file.size[0] - shorter_side) / 2
    y_cut = (jpeg_file.size[1] - shorter_side) / 2

    jpeg_cropped = jpeg_file.crop((x_cut, y_cut, jpeg_file.size[0] - x_cut, jpeg_file.size[1] - y_cut))
    jpeg_resized = jpeg_cropped.resize((IM_WIDTH, IM_HEIGHT))
    jpeg_resized.save(os.path.join(DATA_DIR, filename))
