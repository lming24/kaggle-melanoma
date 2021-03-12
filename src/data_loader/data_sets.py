"""
Contains all data set classes. They should all inherit from torch.utils.data.Dataset
"""

import io
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
from PIL import Image


def get_skincolor_image(img):
    """
    Returns an image with the average skin color
    """
    img = np.array(img)

    img_hsv = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    img_rgb = torch.from_numpy(img)

    mask = img_hsv[:, :, 0] <= 25
    mask &= img_hsv[:, :, 1] >= 59
    mask &= img_hsv[:, :, 1] <= 173
    mask &= img_rgb[:, :, 0] > 95
    mask &= img_rgb[:, :, 1] > 40
    mask &= img_rgb[:, :, 2] > 20
    mask &= img_rgb[:, :, 0] > img_rgb[:, :, 1]
    mask &= img_rgb[:, :, 0] > img_rgb[:, :, 2]
    mask &= (torch.abs(img_rgb[:, :, 0] - img_rgb[:, :, 1]) > 15)
    channels = img_rgb.view(-1, 3)[mask.view(-1), :]
    mean = channels.float().mean(0)
    skin = torch.empty_like(img_rgb, dtype=torch.uint8)
    skin[:, :, :] = mean.byte().view(1, 1, -1)

    return Image.fromarray(skin.numpy())


class MelanomaDataset(Dataset):
    """
    Expected folder structure:
    - train.csv
    - images
        - <image1>.jpg
        - <image2>.jpg
        - ...
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, path, transform=None, preload_to_mem=False, add_context=False, max_context_imgs=3):
        # pylint: disable=too-many-arguments,too-many-branches
        self.path = pathlib.Path(path).resolve()

        self.image_folder = self.path / 'images'
        self.metadata = pd.read_csv(str(self.path / 'metadata.csv'))

        try:
            self.malignant = len(self.metadata[self.metadata['target'] == 1])
            self.benign = len(self.metadata[self.metadata['target'] == 0])
        except KeyError:
            self.malignant = 0
            self.benign = 0

        # A function that performs pre-processing of data
        # e.g conversion to tensor or data-augmentation (random rotations etc)
        self.transform = transform

        self.num_images = len(self.metadata)

        self.add_context = add_context
        self.patient_2_idx = {}
        for idx in range(self.num_images):
            metadata = self.metadata.iloc[idx]
            patient_id = metadata["patient_id"]
            try:
                self.patient_2_idx[patient_id].add(idx)
            except KeyError:
                self.patient_2_idx[patient_id] = set([idx])

        self.memory = []
        if preload_to_mem:
            for idx in range(self.num_images):
                metadata = self.metadata.iloc[idx]
                image_name = metadata.pop("image_name") + '.jpg'
                img_path = self.image_folder / image_name

                # Load raw binary data but do not decode/decompress (keep memory low)
                with img_path.open('rb') as file:
                    file_content = file.read()
                    self.memory.append(file_content)
        self.max_context_imgs = max_context_imgs

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # pylint: disable=too-many-branches
        if idx >= len(self):
            raise IndexError("Dataset index out of range")

        metadata = self.metadata.iloc[idx]
        image_name = metadata.pop("image_name") + '.jpg'

        res = {
            "image_name": image_name,
            "patient_id": metadata["patient_id"],
            "sex": metadata["sex"],
            "age": metadata["age_approx"],
            "location": metadata["anatom_site_general_challenge"]
        }

        try:
            res["target"] = metadata["target"]
        except KeyError:
            pass

        img_path = str(self.image_folder / image_name)

        if self.memory:
            img_bytes = io.BytesIO(self.memory[idx])
            img = Image.open(img_bytes)
        else:
            img = Image.open(img_path)

        res["img"] = img

        # Add context
        if self.add_context:
            ctx = []
            try:
                choices = np.random.choice(self.patient_2_idx[res["patient_id"]],
                                           replace=True,
                                           size=self.max_context_imgs)
            except ValueError:
                choices = []
            for cidx in choices:
                if cidx == idx:
                    continue
                context_img = self.metadata.iloc[cidx]["image_name"] + '.jpg'
                if self.memory:
                    cimg_bytes = io.BytesIO(self.memory[cidx])
                    cimg = Image.open(cimg_bytes)
                else:
                    cimg_path = str(self.image_folder / context_img)
                    cimg = Image.open(cimg_path)
                ctx.append(cimg)

            ctx.append(get_skincolor_image(res["img"]))

            res["context"] = ctx
            res["context_lengths"] = len(ctx)
        else:
            res["context"] = []
            res["context_lengths"] = 0

        if self.transform:
            res = self.transform(res)
        else:
            # Pillow opens files lazily and waits until they are processed.
            # If not transformation is required force loading of the images here
            # to avoid open file pointers. This also closes the file
            img.load()

        return res

    def get_targets(self):
        """Returns target labels as a tensor"""
        return torch.tensor(self.metadata["target"].tolist())  # pylint: disable=not-callable


class MelanomaDataset2(MelanomaDataset):
    """
    Expected folder structure:
    - train.csv
    - images
        - <image1>.jpg
        - <image2>.jpg
        - ...
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 path,
                 exclude_ids=None,
                 transform=None,
                 preload_to_mem=False,
                 add_context=False,
                 max_context_imgs=3):
        # pylint: disable=too-many-arguments,too-many-branches, super-init-not-called, too-many-locals
        if not isinstance(path, list):
            path = [path]

        path = [pathlib.Path(p).resolve() for p in path]
        self.path = path

        if not exclude_ids:
            exclude_ids = [[] for _ in range(len(path))]

        self.metadata = None
        for subpath, ex_id in zip(path, exclude_ids):
            metadata = pd.read_csv(str(subpath / 'metadata.csv'))

            if ex_id:
                try:
                    mask = (metadata.tfrecord == ex_id[0])
                    for id_ in ex_id[1:]:
                        mask = (mask | (metadata.tfrecord == id_))
                    metadata = metadata[~mask]
                except AttributeError:
                    pass

            metadata.loc[:, 'path'] = str(subpath / 'images')

            if self.metadata is None:
                self.metadata = metadata
            else:
                self.metadata = pd.concat([self.metadata, metadata], ignore_index=True)

        try:
            self.malignant = len(self.metadata[self.metadata['target'] == 1])
            self.benign = len(self.metadata[self.metadata['target'] == 0])
        except KeyError:
            self.malignant = 0
            self.benign = 0

        # A function that performs pre-processing of data
        # e.g conversion to tensor or data-augmentation (random rotations etc)
        self.transform = transform

        self.num_images = len(self.metadata)

        self.add_context = add_context
        self.patient_2_idx = {}
        for idx in range(self.num_images):
            metadata = self.metadata.iloc[idx]
            patient_id = metadata["patient_id"]
            try:
                self.patient_2_idx[patient_id].add(idx)
            except KeyError:
                self.patient_2_idx[patient_id] = set([idx])

        self.memory = []
        if preload_to_mem:
            for idx in range(self.num_images):
                metadata = self.metadata.iloc[idx]
                image_name = metadata.pop("image_name") + '.jpg'
                subpath = pathlib.Path(metadata.pop("path"))
                img_path = subpath / image_name

                # Load raw binary data but do not decode/decompress (keep memory low)
                with img_path.open('rb') as file:
                    file_content = file.read()
                    self.memory.append(file_content)
        self.max_context_imgs = max_context_imgs

    def __getitem__(self, idx):
        # pylint: disable=too-many-branches
        if idx >= len(self):
            raise IndexError("Dataset index out of range")

        metadata = self.metadata.iloc[idx]
        image_name = metadata.pop("image_name") + '.jpg'
        image_folder_str = metadata.pop("path")
        image_folder = pathlib.Path(image_folder_str)

        res = {
            "image_name": image_name,
            "path": image_folder_str,
            "patient_id": str(metadata["patient_id"]),
            "sex": metadata["sex"],
            "age": metadata["age_approx"],
            "location": metadata["anatom_site_general_challenge"]
        }

        try:
            res["target"] = metadata["target"]
        except KeyError:
            pass

        img_path = str(image_folder / image_name)

        if self.memory:
            img_bytes = io.BytesIO(self.memory[idx])
            img = Image.open(img_bytes)
        else:
            img = Image.open(img_path)

        res["img"] = img

        # Add context
        if self.add_context:
            ctx = []
            try:
                choices = np.random.choice(self.patient_2_idx[res["patient_id"]],
                                           replace=True,
                                           size=self.max_context_imgs)
            except ValueError:
                choices = []
            for cidx in choices:
                if cidx == idx:
                    continue
                context_img = self.metadata.iloc[cidx]["image_name"] + '.jpg'
                context_path = self.metadata.iloc[cidx]["path"]
                if self.memory:
                    cimg_bytes = io.BytesIO(self.memory[cidx])
                    cimg = Image.open(cimg_bytes)
                else:
                    cimg_path = str(context_path / context_img)
                    cimg = Image.open(cimg_path)
                ctx.append(cimg)

            ctx.append(get_skincolor_image(res["img"]))

            res["context"] = ctx
            res["context_lengths"] = len(ctx)
        else:
            res["context"] = []
            res["context_lengths"] = 0

        if self.transform:
            res = self.transform(res)
        else:
            # Pillow opens files lazily and waits until they are processed.
            # If not transformation is required force loading of the images here
            # to avoid open file pointers. This also closes the file
            img.load()

        return res


class MelanomaCountDataset(Dataset):
    """
    Generates bags of pictures
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals
    def __init__(self,
                 path,
                 transform=None,
                 preload_to_mem=False,
                 mean_bag_size=8,
                 var_bag_size=2,
                 num_bags=250,
                 seed=1,
                 random_target_first=False):
        self.path = path
        self.transform = transform
        self.preload_to_mem = preload_to_mem
        self.mean_bag_size = mean_bag_size
        self.var_bag_size = var_bag_size

        dataset = MelanomaDataset(path, transform, preload_to_mem)
        targets = dataset.get_targets()
        self.malignant = targets.nonzero().view(-1)
        self.benign = (targets == 0).nonzero().view(-1)

        self.r = np.random.RandomState(seed)  # pylint: disable=invalid-name
        self.bags = []
        for _ in range(num_bags):
            bag_length = np.int(self.r.normal(mean_bag_size, var_bag_size, 1))
            if bag_length < 1:
                bag_length = 1

            if not random_target_first:
                indices = torch.LongTensor(self.r.randint(0, len(dataset), bag_length))
                target_number = (targets[indices] == 1).sum().item()
            else:
                target_number = self.r.randint(0, bag_length + 1, 1)[0]
                if self.malignant.size(0) == 0:
                    target_number = 0
                    malignant_indices = self.malignant.clone()
                else:
                    malignant_indices = self.malignant[self.r.randint(0, self.malignant.size(0), target_number)]
                benign_indices = self.benign[self.r.randint(0, self.benign.size(0), bag_length - target_number)]
                indices = torch.cat([malignant_indices, benign_indices])

            self.bags.append((indices, target_number))

        self.dataset = dataset

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        indices, target = self.bags[idx]

        imgs = []
        for i in indices:
            imgs.append(self.dataset[i.item()]["img"])

        return {"imgs": torch.stack(imgs, dim=0), "target": torch.tensor(target), "len": torch.tensor(len(indices))}  # pylint: disable=not-callable
