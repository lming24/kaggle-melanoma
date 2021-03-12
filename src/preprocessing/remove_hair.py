"""
Calculates statistics for the training dataset such as mean and stddev.
"""

import argparse
import pathlib

import torchvision.transforms as vision_transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_loader.transforms as transforms
import data_loader.data_sets as datasets


def main():
    """
    Main function
    """
    # pylint: disable=too-many-locals, too-many-statements
    args = argparse.ArgumentParser(description='Image Reconstruction')
    args.add_argument('-p', '--path', type=str, help='path to folder with images')

    args = args.parse_args()

    paths = [pathlib.Path(p) for p in args.path.split(',')]
    exclude_ids = [[-1] for _ in paths]

    img_trsfm = vision_transforms.Compose([transforms.ColorConstancy(gamma=2.2), transforms.RemoveHair()])
    male_female = transforms.StringToInt({'male': 0, 'female': 1}, 'male')
    location = transforms.StringToOneHot(
        {
            'torso': 0,
            'lower extremity': 1,
            'upper extremity': 2,
            'head/neck': 3,
            'palms/soles': 4,
            'oral/genital': 5
        },
        missing_value='torso')
    transform1 = transforms.DictionaryTransform(["img", "sex", "location"], [img_trsfm, male_female, location])
    # Let's assume the average age of the person is 40 so we replace NaNs with 40
    transform2 = transforms.AggregateAndNormalizeToTensor(["sex", "age", "location"],
                                                          mean=[0, 0, 0, 0, 0, 0, 0, 0],
                                                          std=[1, 1, 1, 1, 1, 1, 1, 1],
                                                          replace_nan={'age': 50.})
    final_transform = vision_transforms.Compose([transform1, transform2])
    dataset = datasets.MelanomaDataset2(paths, exclude_ids, final_transform, preload_to_mem=True)

    for sample in tqdm(dataset):
        image_name = sample["image_name"]
        subpath = pathlib.Path(sample["path"])
        img = sample["img"]
        img_path = str(subpath / image_name)
        img.save(img_path)


if __name__ == '__main__':
    main()
