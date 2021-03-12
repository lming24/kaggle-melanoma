"""
Resize PIL images
"""

import argparse
import pathlib

from tqdm import tqdm
import torchvision.transforms as vision_transforms

import data_loader.transforms as transforms
import data_loader.data_sets as datasets


def main():
    """
    Main function
    """
    # pylint: disable=too-many-locals, too-many-statements
    args = argparse.ArgumentParser(description='Image Reconstruction')
    args.add_argument('-p', '--path', type=str, help='path to folder with images')
    args.add_argument('-s', '--scale', type=float, help='how many times to scale down')

    args = args.parse_args()

    original_size = [4000., 6000.]
    new_size = [int(size / args.scale) for size in original_size]
    print("Resizing images to size: ", new_size)

    path = pathlib.Path(args.path)

    img_trsfm = vision_transforms.Compose([vision_transforms.Resize(new_size)])
    transform = transforms.DictionaryTransform(["img"], [img_trsfm])
    dataset = datasets.MelanomaDataset(path, transform)
    for data in tqdm(dataset):
        name, pil_image = data["image_name"], data["img"]
        pil_image.save(dataset.path / 'images' / name)


if __name__ == '__main__':
    main()
