import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image
# from torchvision.transforms import Normalize
# from utilities.constants import IMG_NORM_MEAN, IMG_NORM_STD


def read_image(path: str, convert=True, convert_code='COLOR_BGR2RGB', astype='uint8'):
    """
    Read an image at given location. 
    Note: cv2 loads image in BGR channels by default. 
        Use `convert='COLOR_BGR2RGB'` to convert into RGB.

    Args:
        path (str): path to the image file
        convert_code (str): required channel conversion code, defaults to 'COLOR_BGR2RGB'
            Note: Check [this](https://docs.opencv.org/master/d8/d01/group__imgproc__color__conversions.html)
            for an exhaustive list. This applies only when `convert=True`

    Returns:
        np.ndarray: loaded image
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, getattr(cv2, convert_code)) if convert else image

    if (image.max() > 1.0 or image.min() < 0) and astype == 'float':
        warnings.warn("Caution! You are reading an image with `astype='float'` while the image is not in [0., 1.0]")

    image = image.astype(astype)

    return image


def resize_image(image: np.ndarray, size: tuple):

    assert len(size) == 2 and isinstance(size, tuple)
    assert isinstance(image, np.ndarray)

    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deeplab Segmentation')
    parser.add_argument('-i', '--image_path', type=str,
        required=True,help='Path to image file that needs to be resized')
    args=parser.parse_args()

    image = read_image(args.image_path)
    image = resize_image(image, (1920, 1080))
    plt.imsave(args.image_path, image)

