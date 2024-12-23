from PIL import Image
import cv2
import numpy as np
from typing import NamedTuple
from functools import lru_cache


class Midline(NamedTuple):
    height: int
    width: int


def img_to_cv(img: Image.Image):
    if img.mode == "LA":
        img = img.copy().convert("L")  # Discard alpha, keep as grayscale
    elif img.mode == "RGBA":
        img = img.copy().convert("RGB")  # Discard alpha, convert to RGB

    img_array = np.array(img)

    # Check the number of dimensions and channels in the image
    if img_array.ndim == 2:
        # Grayscale image (1 channel)
        cv_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.ndim == 3:
        if img_array.shape[2] == 3:
            # RGB image (3 channels)
            cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif img_array.shape[2] == 4:
            # RGBA image (4 channels)
            cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            raise ValueError(f"Unsupported number of channels: {img_array.shape[2]}")
    else:
        raise ValueError("Invalid image format")

    return cv_img


@lru_cache
def get_face_classifier() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
    )


def get_midline_from_image(img: Image.Image) -> Midline | None:
    cv_img = img_to_cv(img)
    gray_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    faces = get_face_classifier().detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    faces = list(faces)

    if not faces:
        return None

    faces.sort(key=lambda x: x[2] * x[3], reverse=True)

    biggest_face = faces[0]

    midline_width = biggest_face[0] + biggest_face[2] // 2
    midline_height = biggest_face[1] + biggest_face[3] // 2

    return Midline(height=midline_height, width=midline_width)


def resize_on_centerline(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """
    Resize the image to given size, cropping more landscape images and centering
    on the eyeline of the face.
    """
    midline = get_midline_from_image(image)

    if midline is None:
        return image.resize(size)

    original_ratio = image.width / image.height
    target_ratio = size[0] / size[1]

    if original_ratio > target_ratio:
        # landscape
        # crop the sides around midline.width
        height = image.height
        width = int(image.height * target_ratio)
        left = midline.width - width // 2
        right = midline.width + width // 2

        if left < 0:
            left = 0
            right = width

        if right > image.width:
            left = image.width - width
            right = image.width

        top = 0
        bottom = height
    elif original_ratio < target_ratio:
        # portrait
        # crop the top and bottom around midline.height
        width = image.width
        height = int(image.width / target_ratio)
        top = midline.height - height // 2
        bottom = midline.height + height // 2

        # if centering would make blank space, just make sure we include the face
        if top < 0:
            top = 0
            bottom = height
        if bottom > image.height:
            top = image.height - height
            bottom = image.height

        left = 0
        right = width
    else:
        return image.resize(size)

    return image.crop((left, top, right, bottom)).resize(size)
