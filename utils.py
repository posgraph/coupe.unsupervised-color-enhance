"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import copy
from skimage import color
import os

def to_Lab(I):
    # AB 98.2330538631 -86.1830297444 94.4781222765 -107.857300207
    lab = color.rgb2lab(I)
    l = (lab[:, :, 0] / 100.0) * 255.0    # L component ranges from 0 to 100
    a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) * 255.0         # a component ranges from -127 to 127
    b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) * 255.0         # b component ranges from -127 to 127
    # return np.dstack([l, a, b]).astype(np.uint8)
    return np.dstack([l, a, b])

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_test_data(image_path, load_size=512, fine_size=256):
    img = imread(image_path)
    # img = __scale_shortest(img, load_size)
    h, w, c = img.shape
    img = img[:h-(h%4),:w-(w%4)]
    # img = scipy.misc.imresize(img, [fine_size, fine_size])
    img = to_Lab(img)
    img = img/127.5 - 1
    return img

def rgb2gray3(image):
    g = color.rgb2gray(image)
    rgb = np.stack((g, g, g)).transpose((1,2,0))
    return rgb

def load_train_data(image_path, load_size=512, fine_size=256, is_testing=False):
    img_A = rgb2gray3(imread(image_path[0]))
    img_B = imread(image_path[1])
    if not is_testing:
        img_A = __scale_shortest(img_A, load_size)
        img_B = __scale_shortest(img_B, load_size)
        ah, aw, ac = img_A.shape
        bh, bw, bc = img_B.shape
        h1 = int(np.floor(np.random.uniform(1e-2, ah-fine_size)))
        w1 = int(np.floor(np.random.uniform(1e-2, aw-fine_size)))
        h2 = int(np.floor(np.random.uniform(1e-2, bh-fine_size)))
        w2 = int(np.floor(np.random.uniform(1e-2, bw-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h2:h2+fine_size, w2:w2+fine_size]
        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    else:
        img_A = __scale_shortest(img_A, load_size)
        img_B = __scale_shortest(img_B, load_size)
        ah, aw, ac = img_A.shape
        bh, bw, bc = img_B.shape
        h1 = int(np.floor(np.random.uniform(1e-2, ah-fine_size)))
        w1 = int(np.floor(np.random.uniform(1e-2, aw-fine_size)))
        h2 = int(np.floor(np.random.uniform(1e-2, bh-fine_size)))
        w2 = int(np.floor(np.random.uniform(1e-2, bw-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h2:h2+fine_size, w2:w2+fine_size]

    img_A = to_Lab(img_A)
    img_B = to_Lab(img_B)
    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def __scale_shortest(img, target_shortest):
    oh, ow, oc = img.shape
    if ow > oh: # set oh to target_shortest
        # if (oh == target_shortest):
        #     return img
        h = target_shortest
        w = int(target_shortest * ow / oh)
        return scipy.misc.imresize(img, [h, w], 'bicubic')
    else: # set ow to target_shortest
        # if (ow == target_shortest):
        #     return img
        w = target_shortest
        h = int(target_shortest * oh / ow)
        return scipy.misc.imresize(img, [h, w], 'bicubic')
# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path, mode='RGB').astype(np.uint8)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    img = to_RGB(img)
    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, scipy.misc.toimage(merge(images, size)*255, cmin=0, cmax=255))

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def to_RGB(I):
    # print(I)
    l = I[:, :, 0] * 100.0
    a = I[:, :, 1] * (98.2330538631 + 86.1830297444) - 86.1830297444
    b = I[:, :, 2] * (94.4781222765 + 107.857300207) - 107.857300207
    # print(np.dstack([l, a, b]))
    rgb = color.lab2rgb(np.dstack([l, a, b]).astype(np.float64))
    return rgb

def inverse_transform(images):
    return (images+1.)/2.
