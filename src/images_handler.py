import matplotlib.pyplot as plt
from PIL import Image
import seaborn
import numpy as np
import json


class ImageHandler:
    def __init__(self):
        self.image = None
        # TODO: define these as properties or arguments
        self.resize_to = 256
        self.crop_to = 224
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def process_image(self, image_path):
        """
        Scales, crops, and normalizes a PIL image for a PyTorch model
        :return: returns an Numpy array
        """
        image = Image.open(image_path)
        img = self.__resize(image, self.resize_to)
        img = self.__crop_center(img, self.crop_to)
        np_img = np.array(img) / 255  # Scaling by 255 (floats between 0 and 1)
        np_img = (np_img - self.mean) / self.std  # Normalize
        np_img = np_img.transpose((2, 0, 1))  # move the third index to the first
        return np_img

    def __resize(self, image, short_size):
        width, height = image.size
        if width >= height:
            image.thumbnail((999999, short_size))
        else:
            image.thumbnail((short_size, 999999))
        return image

    def __crop_center(self, image, size):
        width, height = image.size
        left = (width - size) / 2
        top = (height - size) /2
        right = (width + size) / 2
        bottom = (height + size) / 2
        return image.crop((left, top, right, bottom))

    def imshow(self, image, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots()

        if title:
            plt.title(title)

        image = image.transpose((1, 2, 0))
        image = self.std * image + self.mean
        image = np.clip(image, 0, 1)
        ax.imshow(image)
        return ax
