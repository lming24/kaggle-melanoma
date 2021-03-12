"""
Module that contains all transformation classes. Data augmentation operations will
be implemented here
"""

import math
import random

import numpy as np
import torch
import cv2
from PIL import Image


@torch.jit.unused
def _parse_fill(fill, img):
    """Helper function to get the fill color for rotate and perspective transforms.
    Args:
        fill (n-tuple or int or float): Pixel fill value for area outside the transformed
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands.
        img (PIL Image): Image to be filled.
        min_pil_version (str): The minimum PILLOW version for when the ``fillcolor`` option
            was first introduced in the calling function. (e.g. rotate->5.2.0, perspective->5.0.0)
    Returns:
        dict: kwarg for ``fillcolor``
    """

    num_bands = len(img.getbands())
    if fill is None:
        fill = 0
    if isinstance(fill, (int, float)) and num_bands > 1:
        fill = tuple([fill] * num_bands)
    if not isinstance(fill, (int, float)) and len(fill) != num_bands:
        msg = ("The number of elements in 'fill' does not match the number of " "bands of the image ({} != {})")
        raise ValueError(msg.format(len(fill), num_bands))

    return {"fillcolor": fill}


def _get_perspective_coeffs(startpoints, endpoints):
    # pylint: disable=invalid-name
    """Helper function to get the coefficients (a, b, c, d, e, f, g, h) for the perspective transforms.
    In Perspective Transform each pixel (x, y) in the original image gets transformed as,
     (x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )
    Args:
        List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
        List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image
    Returns:
        octuple (a, b, c, d, e, f, g, h) for transforming each pixel.
    """
    matrix = []

    for p1, p2 in zip(endpoints, startpoints):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = torch.tensor(matrix, dtype=torch.float)  # pylint: disable=not-callable
    B = torch.tensor(startpoints, dtype=torch.float).view(8)  # pylint: disable=not-callable
    res = torch.lstsq(B, A)[0]
    return res.squeeze_(1).tolist()


def perspective(img, startpoints, endpoints, interpolation=Image.BICUBIC, fill=None):
    """Perform perspective transform of the given PIL Image.
    Args:
        img (PIL Image): Image to be transformed.
        startpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the original image
        endpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image
        interpolation: Default- Image.BICUBIC
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            This option is only available for ``pillow>=5.0.0``.
    Returns:
        PIL Image:  Perspectively transformed Image.
    """

    opts = _parse_fill(fill, img)

    coeffs = _get_perspective_coeffs(startpoints, endpoints)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, interpolation, **opts)


class RandomPerspective:
    """Performs Perspective transformation of the given PIL Image randomly with a given probability.

    Args:
        interpolation : Default- Image.BICUBIC

        p (float): probability of the image being perspectively transformed. Default value is 0.5

        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.

        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively. Default value is 0.
    """
    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=Image.BICUBIC, fill=0):
        # pylint: disable=invalid-name
        self.p = p
        self.interpolation = interpolation
        self.distortion_scale = distortion_scale
        self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be Perspectively transformed.

        Returns:
            PIL Image: Random perspectivley transformed image.
        """
        if random.random() < self.p:
            width, height = img.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            return perspective(img, startpoints, endpoints, self.interpolation, self.fill)
        return img

    @staticmethod
    def get_params(width, height, distortion_scale):
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width : width of the image.
            height : height of the image.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1,
                                   width - 1), random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class DictionaryTransform:
    """
    Takes a list of keys and a list of transform instances and applies
    each transform on the items of the dictionary based on the `keys` list
    e.g keys=['img'] and transforms=[img_transform] will apply the function img_transform
    to the dictionary['img'] and return the updated dictionary. This allows for easy
    composition of many different transforms
    """
    def __init__(self, keys, transforms):
        self.keys = keys
        self.transforms = transforms

    def __call__(self, dictionary):
        for key, transform in zip(self.keys, self.transforms):
            dictionary[key] = transform(dictionary[key])

        return dictionary


class StringToInt:
    """
    Maps strings to integers. Useful for converting categorical variables to an int
    """
    def __init__(self, dictionary, missing_value):
        self.dictionary = dictionary
        self.missing_value = missing_value

    def __call__(self, x):
        try:
            return float(self.dictionary[x])
        except KeyError:
            # print("Warning: KeyError converted to", len(self.dictionary), x, 'not found in', self.dictionary)
            return float(self.dictionary[self.missing_value])


class StringToOneHot:
    """
    Maps strings to integers. Useful for converting categorical variables to an int
    """
    def __init__(self, dictionary, missing_value):
        self.dictionary = dictionary
        self.len_dict = len(set(dictionary.values()))
        self.missing_value = missing_value

    def __call__(self, x):
        try:
            value = int(self.dictionary[x])
        except KeyError:
            # print("Warning: KeyError converted to", len(self.dictionary), x, 'not found in', self.dictionary)
            value = int(self.dictionary[self.missing_value])
        onehot = [0.] * self.len_dict
        onehot[value] = 1.
        return onehot


class AggregateAndNormalizeToTensor:
    """
    Aggragates values from multiple keys from dictionary, normalizing them
    in the process
    """
    def __init__(self, keys, mean, std, replace_nan):
        self.keys = keys
        self.mean = torch.tensor(mean, dtype=torch.float)  # pylint: disable=not-callable
        self.std = torch.tensor(std, dtype=torch.float)  # pylint: disable=not-callable
        self.replace_nan = replace_nan

    def _replace_nan_and_pack(self, value, key, index):
        _ = index
        if not isinstance(value, (tuple, list)):
            if math.isnan(value):
                # print("Warning: NaN found in", key)
                value = self.replace_nan[key]
            value = [value]
        return value

    def __call__(self, dictionary):
        dictionary.pop("path", None)
        features = [
            x for i, key in enumerate(self.keys) for x in self._replace_nan_and_pack(dictionary.pop(key), key, i)
        ]
        features = torch.tensor(features, dtype=torch.float)  # pylint: disable=not-callable
        dictionary["features"] = (features - self.mean) / self.std
        return dictionary


class Microscope:
    """
    Cutting out the edges around the center circle of the image
    Imitating a picture, taken through the microscope

    Taken from https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet

    Args:
        p (float): probability of applying an augmentation
    """
    def __init__(self, p: float = 0.5):
        self.p = p  # pylint: disable=invalid-name

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to apply transformation to.

        Returns:
            PIL Image: Image with transformation.
        """
        if random.random() < self.p:
            img = np.array(img)
            circle = cv2.circle(
                (np.ones(img.shape) * 255).astype(np.uint8),  # image placeholder
                (img.shape[1] // 2, img.shape[0] // 2 + 1),  # center point of circle
                random.randint(img.shape[0] // 2 - 3, img.shape[0] // 2 + 15),  # radius
                (0, 0, 0),  # color
                -1)

            mask = circle - 255
            img = np.multiply(img, mask)
            img = Image.fromarray(img)

        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


class DrawHair:
    """
    Draw a random number of pseudo hairs

    Args:
        hairs (int): maximum number of hairs to draw
        width (tuple): possible width of the hair in pixels
        p (float): probability of drawing hairs
    """
    def __init__(self, hairs: int = 4, width: tuple = (1, 2), p: float = 0.3):  # pylint: disable=invalid-name
        self.hairs = hairs
        self.width = width
        self.p = p  # pylint: disable=invalid-name

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        if not self.hairs:
            return img

        if random.random() >= self.p:
            return img

        img = np.array(img)
        height, width, _ = img.shape

        for _ in range(random.randint(0, self.hairs)):
            # The origin point of the line will always be at the top half of the image
            origin = (random.randint(0, width), random.randint(0, height // 2))
            # The end of the line
            end = (random.randint(0, width), random.randint(0, height))
            color = (0, 0, 0)  # color of the hair. Black.
            cv2.line(img, origin, end, color, random.randint(self.width[0], self.width[1]))

        img = Image.fromarray(img)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, width={self.width})'


class ColorConstancy:
    """
    Normalize colors and apply gamma correction
    """
    def __init__(self, power=6, gamma=None):
        self.power = power
        self.gamma = gamma
        if gamma:
            self.look_up_table = np.ones((256, 1), dtype='uint8') * 0
            for i in range(256):
                self.look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)

    def __call__(self, img):
        power = self.power
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_dtype = img.dtype

        if self.gamma is not None:
            img = img.astype('uint8')
            img = cv2.LUT(img, self.look_up_table)

        img = img.astype('float32')
        img_power = np.power(img, power)
        rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
        rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
        rgb_vec = rgb_vec / rgb_norm
        rgb_vec = 1 / (rgb_vec * np.sqrt(3))
        img = np.multiply(img, rgb_vec)

        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        return Image.fromarray(img.astype(img_dtype))


class RemoveHair:
    """
    Remove hairs from image and fill image using inpainting
    """
    def __init__(self, threshold=10):
        self.threshold = threshold

    def __call__(self, img):
        img = np.array(img)

        # Convert the original image to grayscale
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Kernel for the morphological filtering
        kernel = cv2.getStructuringElement(1, (17, 17))

        # Perform the blackHat filtering on the grayscale image to find the
        # hair countours
        blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)

        # intensify the hair countours in preparation for the inpainting
        # algorithm
        _, thresh2 = cv2.threshold(blackhat, self.threshold, 255, cv2.THRESH_BINARY)

        # inpaint the original image depending on the mask
        img = cv2.inpaint(img, thresh2, 1, cv2.INPAINT_TELEA)
        return Image.fromarray(img)


class MultipleAugmentations:
    """
    Add one additional dimension where different augmentations are applied to the same
    image

    The output of the augmentations needs to be a tensor while the input needs to be
    a PIL image
    """
    def __init__(self, transform, num_of_augmentations=3):
        self.transform = transform
        self.num_of_augmentations = num_of_augmentations

    def __call__(self, img):
        outputs = []
        for _ in range(self.num_of_augmentations):
            img_temp = img.copy()
            img_temp = self.transform(img_temp)
            outputs.append(img_temp)

        return torch.stack(outputs, dim=0)


class ApplyOnIterable:
    """Apply a transform on a list of samples"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, imgs):
        if imgs:
            return torch.stack([self.transform(img) for img in imgs], dim=0)
        else:
            return imgs


class AggregateFromDictAndApplyTransform:
    """
    Apply a transform on a list of images aggregated from a list of keys
    """
    def __init__(self, transform, keys):
        self.transform = transform
        self.keys = keys

    def __call__(self, x):
        counter = 0
        counters = [0]
        is_list = []
        aggregate = []
        for key in self.keys:
            if torch.is_tensor(x[key]):
                aggregate.append(x[key])
                is_list.append(False)
                counter += 1
            else:
                aggregate.extend(x[key])
                is_list.append(True)
                counter += len(x[key])
            counters.append(counter)

        aggregate = self.transform(aggregate)

        for start, end, key, is_list_tmp in zip(counters[0:-1], counters[1:], self.keys, is_list):
            if is_list_tmp:
                x[key] = aggregate[start:end]
            else:
                assert (end - start) == 1
                x[key] = aggregate[start]
        return x
