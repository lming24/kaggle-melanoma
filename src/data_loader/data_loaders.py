"""
Module that contains all available dataloaders
"""

import torchvision.transforms as vision_transforms
import torchvision.datasets as vision_datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

import data_loader.data_sets as datasets
import data_loader.transforms as transforms
import logger.logger as logger


def concatenate_collate(batch):
    """
    Same a default collate but concatenates batches instead of stacking
    """
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)  # pylint: disable=protected-access
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    return batch


def melanoma_counter_collate(batch):
    """Colate function used by MelanomaCountDataloader"""
    imgs = concatenate_collate([d.pop("imgs") for d in batch])
    res = torch.utils.data.dataloader.default_collate(batch)
    res["imgs"] = imgs

    return res


def melanoma_context_collate(batch):
    """Colate function used by MelanomaCountDataloader"""
    imgs = concatenate_collate([d.pop("context") for d in batch])
    res = torch.utils.data.dataloader.default_collate(batch)
    res["context"] = imgs

    return res


class MnistDataLoader(DataLoader):
    """
    MNIST data loading demo
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True):
        # pylint: disable=too-many-arguments
        trsfm = vision_transforms.Compose(
            [vision_transforms.ToTensor(),
             vision_transforms.Normalize((0.1307, ), (0.3081, ))])
        self.data_dir = data_dir
        self.dataset = vision_datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class MelanomaDataloader(DataLoader):
    """
    Melanoma dataset
    """
    def __init__(self,
                 path,
                 batch_size,
                 shuffle=True,
                 exclude_ids=None,
                 num_workers=1,
                 malignant_sampling_weight=None,
                 preload_to_mem=False,
                 augmentation=False,
                 hue_augmentation=0.0,
                 target_size=(512, 512),
                 resize_type='random_resized_crop',
                 tta_augmentations=1,
                 add_context=False,
                 max_context_imgs=5,
                 ignore_location=False):
        # pylint: disable=too-many-arguments, too-many-locals
        #Mean of images: tensor([0.8063, 0.6212, 0.5919])
        #StdDev of images: tensor([0.1499, 0.1756, 0.2020])
        #Mean of other features: tensor([4.8160e-01, 4.8529e+01, 5.2544e-01, 2.5223e-01, 1.5141e-01, 5.5425e-02,
        #1.1483e-02, 4.0158e-03])
        #StdDev of other features: tensor([ 0.4997, 14.2342,  0.4994,  0.4343,  0.3584,  0.2288,  0.1065,  0.0632])

        logging = logger.get_logger("melanoma_dataloader", verbosity=2)
        pixel_mean = tuple([int(255 * x) for x in [0.8330, 0.6405, 0.6078]])
        assert resize_type in ['random_resized_crop', 'resize']
        if resize_type == 'random_resized_crop':
            resize_op = vision_transforms.RandomResizedCrop(tuple(target_size))
        else:
            resize_op = vision_transforms.Resize(tuple(target_size))
        if augmentation:
            img_trsfm = vision_transforms.Compose([
                transforms.DrawHair(p=0.2),
                # transforms.Microscope(p=0.2),
                vision_transforms.RandomHorizontalFlip(),
                vision_transforms.RandomVerticalFlip(),
                vision_transforms.RandomRotation((90, -90), fill=pixel_mean),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3, fill=pixel_mean),
                vision_transforms.ColorJitter(brightness=32. / 255., saturation=0.5, hue=hue_augmentation),
                resize_op,
                vision_transforms.ToTensor(),
                vision_transforms.Normalize([0.8330, 0.6405, 0.6078], [0.1456, 0.1852, 0.2123])
            ])
        else:
            img_trsfm = vision_transforms.Compose([
                vision_transforms.Resize(tuple(target_size)),
                vision_transforms.ToTensor(),
                vision_transforms.Normalize([0.8330, 0.6405, 0.6078], [0.1456, 0.1852, 0.2123])
            ])
        if tta_augmentations > 1:
            img_trsfm = transforms.MultipleAugmentations(img_trsfm, tta_augmentations)
        male_female = transforms.StringToInt({'male': 0, 'female': 1}, 'male')
        location = transforms.StringToOneHot(
            {
                'torso': 0,
                'posterior torso': 0,
                'anterior torso': 0,
                'lower extremity': 1,
                'upper extremity': 2,
                'head/neck': 3,
                'palms/soles': 4,
                'oral/genital': 5
            },
            missing_value='torso')
        if not ignore_location:
            transform1 = transforms.DictionaryTransform(
                ["img", "sex", "location", "context"],
                [img_trsfm, male_female, location,
                 transforms.ApplyOnIterable(img_trsfm)])
            transform2 = transforms.AggregateAndNormalizeToTensor(
                ["sex", "age", "location"],
                mean=[4.8160e-01, 4.8529e+01, 5.2544e-01, 2.5223e-01, 1.5141e-01, 5.5425e-02, 1.1483e-02, 4.0158e-03],
                std=[0.4997, 14.2342, 0.4994, 0.4343, 0.3584, 0.2288, 0.1065, 0.0632],
                replace_nan={'age': 50.})
        else:
            transform1 = transforms.DictionaryTransform(
                ["img", "sex", "location", "context"],
                [img_trsfm, male_female, location,
                 transforms.ApplyOnIterable(img_trsfm)])
            transform2 = transforms.AggregateAndNormalizeToTensor(["sex", "age"],
                                                                  mean=[4.8160e-01, 4.8529e+01],
                                                                  std=[0.4997, 14.2342],
                                                                  replace_nan={'age': 50.})
        final_transform = vision_transforms.Compose([transform1, transform2])

        self.path = path
        self.dataset = datasets.MelanomaDataset2(self.path,
                                                 exclude_ids=exclude_ids,
                                                 transform=final_transform,
                                                 preload_to_mem=preload_to_mem,
                                                 add_context=add_context,
                                                 max_context_imgs=max_context_imgs)

        num_malignant = self.dataset.malignant
        num_benign = self.dataset.benign
        total = num_malignant + num_benign
        if malignant_sampling_weight is None:
            sampler = None
        elif malignant_sampling_weight == 'equal':
            class_weights = torch.tensor([num_benign, num_malignant], dtype=torch.float)  # pylint: disable=not-callable
            class_weights = 1.0 / class_weights
            distribution = class_weights[self.dataset.get_targets()]
            sampler = WeightedRandomSampler(distribution, len(self.dataset))
            if shuffle is False:
                logging.warning("Shuffling will happen even if shuffle is False when malignant_sampling_weight is set")
            else:
                shuffle = False
        else:
            assert isinstance(malignant_sampling_weight, (float, int))
            class_weights = torch.tensor([1. / total, 1. / total])  # pylint: disable=not-callable
            class_weights[1] = class_weights[1] * malignant_sampling_weight
            class_weights = class_weights / class_weights.sum()
            distribution = class_weights[self.dataset.get_targets()]
            sampler = WeightedRandomSampler(distribution, len(self.dataset))
            if shuffle is False:
                logging.warning("Shuffling will happen even if shuffle is False when malignant_sampling_weight is set")
            else:
                shuffle = False

        super().__init__(self.dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         sampler=sampler,
                         num_workers=num_workers,
                         collate_fn=melanoma_context_collate,
                         pin_memory=True)


class MelanomaCountDataloader(DataLoader):
    """
    Melanoma count dataset
    """
    def __init__(self,
                 path,
                 batch_size,
                 mean_bag_size=8,
                 var_bag_size=2,
                 num_bags=250,
                 seed=1,
                 random_target_first=False,
                 shuffle=True,
                 num_workers=1,
                 preload_to_mem=False,
                 augmentation=False,
                 hue_augmentation=0.0):
        # pylint: disable=too-many-arguments, too-many-locals
        #Mean of images: tensor([0.8063, 0.6212, 0.5919])
        #StdDev of images: tensor([0.1499, 0.1756, 0.2020])
        #Mean of other features: tensor([4.8160e-01, 4.8529e+01, 5.2544e-01, 2.5223e-01, 1.5141e-01, 5.5425e-02,
        #1.1483e-02, 4.0158e-03])
        #StdDev of other features: tensor([ 0.4997, 14.2342,  0.4994,  0.4343,  0.3584,  0.2288,  0.1065,  0.0632])

        # logging = logger.get_logger("melanoma_count_dataloader", verbosity=2)
        if augmentation:
            img_trsfm = vision_transforms.Compose([
                transforms.DrawHair(p=0.3),
                transforms.Microscope(p=0.5),
                vision_transforms.RandomHorizontalFlip(),
                vision_transforms.RandomVerticalFlip(),
                vision_transforms.RandomRotation((90, -90)),
                vision_transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                vision_transforms.ColorJitter(brightness=32. / 255., saturation=0.5, hue=hue_augmentation),
                vision_transforms.ToTensor(),
                vision_transforms.Normalize([0.8063, 0.6212, 0.5919], [0.1499, 0.1756, 0.2020])
            ])
        else:
            img_trsfm = vision_transforms.Compose([
                vision_transforms.ToTensor(),
                vision_transforms.Normalize([0.8063, 0.6212, 0.5919], [0.1499, 0.1756, 0.2020])
            ])

        transform = transforms.DictionaryTransform(["img"], [img_trsfm])

        self.path = path
        self.dataset = datasets.MelanomaCountDataset(self.path,
                                                     transform=transform,
                                                     preload_to_mem=preload_to_mem,
                                                     mean_bag_size=mean_bag_size,
                                                     var_bag_size=var_bag_size,
                                                     num_bags=num_bags,
                                                     seed=seed,
                                                     random_target_first=random_target_first)

        super().__init__(self.dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         pin_memory=True,
                         collate_fn=melanoma_counter_collate)
