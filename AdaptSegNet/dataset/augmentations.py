# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import imgaug.augmenters as iaa

# ImageNet code should change this value
IMAGE_SIZE = 32


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

def gaussianblur(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.GaussianBlur(sigma=(0, level))
    return fun(images=pil_img).transpose(1, 2, 0)


def poisson(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.AdditivePoissonNoise(40, per_channel=True)
    return fun(images=pil_img).transpose(1, 2, 0)


def snowflake(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03))
    return fun(images=pil_img).transpose(1, 2, 0)

def rain(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03))
    return fun(images=pil_img).transpose(1, 2, 0)

def multi(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.MultiplyElementwise((0.5, 1.5))
    return fun(images=pil_img).transpose(1, 2, 0)

def dropout(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.CoarseDropout(0.02, size_percent=0.5, per_channel=True)
    return fun(images=pil_img).transpose(1, 2, 0)

def blendalpha(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.BlendAlpha((0.0, 1.0), iaa.Grayscale(1.0))
    return fun(images=pil_img).transpose(1, 2, 0)


def canny(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.Canny(alpha=(0.0, 0.5))
    img = fun(images=pil_img).transpose(1, 2, 0)
    #print(img.shape)
    return img

extensive_aug = [gaussianblur, snowflake, rain, multi, dropout, canny, poisson]

style_augment = [solarize, posterize, brightness]
noise_augment = extensive_aug + [autocontrast, equalize, color, sharpness, brightness]

augmentations = [
    autocontrast, equalize, posterize, solarize, color]

extensive_plus_normal = extensive_aug + augmentations
#augmentations = [
#    autocontrast, equalize, posterize, solarize, color, contrast, brightness, sharpness]


augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]
