from PIL import Image
import os
from skimage import color
import numpy as np
import sys

# Import settings
IM_WIDTH = 224
IM_HEIGHT = 224
DATA_DIR = 'resources'
IM_PER_BIN = 100

images_loaded = 0

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
    # Import RAW RGB data
    raw = np.reshape(jpeg_resized.getdata(), (IM_WIDTH, IM_HEIGHT, 3)).astype(float)
    # Convert RGB to LAB
    lab = color.rgb2lab(raw / 256)
    # Flatten LAB to Pixel Data
    data = lab.flatten()

    # print sys.getsizeof(data[0])

    # Write image to binary file
    file = open('bin/data' + str(images_loaded / IM_PER_BIN) + '.bin', 'ab')

    for pix_col in data:
        file.write(pix_col)

    file.write(bytes(data))
    file.close()

    # Iterate images_loaded
    images_loaded += 1
