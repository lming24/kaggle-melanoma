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


def calc_mean(dataset):
    """
    Calculate mean
    """
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)

    mean = 0.
    n_samples = 0.
    sum_of_features = 0.
    for data_all in tqdm(dataloader):
        data = data_all['img']
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)  # Flatten images, retain channels
        mean += data.mean(2).sum(0)  # Sum across samples
        n_samples += batch_samples
        sum_of_features += data_all['features'].sum(0)

    return mean / n_samples, sum_of_features / n_samples


def calc_var(dataset, mean, other_mean):
    """
    Calculate sum of variance
    """
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)

    var = 0.
    n_pixels = 0.
    n_samples = 0.
    sum_of_sqr_diff = 0.
    for data_all in tqdm(dataloader):
        data = data_all['img']
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)  # Flatten images, retain channels
        var += ((data - mean.unsqueeze(1))**2).sum([0, 2])  # Sum over samples and pixels
        n_pixels += batch_samples * data.size(2)
        n_samples += batch_samples
        sum_of_sqr_diff += ((data_all['features'] - other_mean)**2).sum(0)

    return var / n_pixels, sum_of_sqr_diff / n_samples


def main():
    """
    Main function
    """
    # pylint: disable=too-many-locals, too-many-statements
    args = argparse.ArgumentParser(description='Image Reconstruction')
    args.add_argument('-p', '--path', type=str, help='path to folder with images')

    args = args.parse_args()

    path = pathlib.Path(args.path)

    img_trsfm = vision_transforms.Compose([vision_transforms.ToTensor()])
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
    dataset = datasets.MelanomaDataset(path, final_transform)

    mean, other_mean = calc_mean(dataset)
    var, other_var = calc_var(dataset, mean, other_mean)

    std = torch.sqrt(var)
    other_std = torch.sqrt(other_var)

    print("Mean of images:", mean)
    print("StdDev of images:", std)
    print("Mean of other features:", other_mean)
    print("StdDev of other features:", other_std)


if __name__ == '__main__':
    main()
