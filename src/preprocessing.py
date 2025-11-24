"""Image preprocessing utilities reused from Assignment 3."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf


def resize_and_normalize(image: tf.Tensor, target_size: Tuple[int, int] = (128, 128)) -> tf.Tensor:
    image = tf.image.resize(image, target_size)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def augment_image(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image

__all__ = ["resize_and_normalize", "augment_image"]
