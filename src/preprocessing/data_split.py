"""
Move 10% of train images into data/images/val/
"""

import os
import pathlib
import pandas as pd

# Folder structure
# - images
#   - image1
#   - image2
#   - .....
# - metadata.csv
SOURCE_FOLDER = pathlib.Path(__file__).parent.parent.parent / 'data' / 'all_train'
TRAIN_FOLDER = pathlib.Path(__file__).parent.parent.parent / 'data' / 'train'
VAL_FOLDER = pathlib.Path(__file__).parent.parent.parent / 'data' / 'val'


def main():
    """
    Main function
    """
    val_metadata_path = (pathlib.Path(__file__).parent.parent.parent / 'data' / 'val_metadata.csv').resolve()
    val_metadata = pd.read_csv(val_metadata_path)
    val_images = list(val_metadata['image_name'].to_numpy())

    source_metadata = pd.read_csv(str(SOURCE_FOLDER / 'metadata.csv'))
    train_metadata = source_metadata[~source_metadata['image_name'].isin(val_images)].copy()
    train_images = list(train_metadata['image_name'].to_numpy())

    print("Images in training dataset: ", len(train_metadata))
    print("Images in validation dataset: ", len(val_metadata))
    print("Percent of validation:", 100 * len(val_metadata) / (len(val_metadata) + len(train_metadata)))

    TRAIN_FOLDER.mkdir(exist_ok=True)
    VAL_FOLDER.mkdir(exist_ok=True)

    (TRAIN_FOLDER / 'images').mkdir(exist_ok=True)
    (VAL_FOLDER / 'images').mkdir(exist_ok=True)

    train_metadata.to_csv(str(TRAIN_FOLDER / 'metadata.csv'))
    val_metadata.to_csv(str(VAL_FOLDER / 'metadata.csv'))

    for image in train_images:
        src_image = str(SOURCE_FOLDER / 'images' / (image + '.jpg'))
        dst_image = str(TRAIN_FOLDER / 'images' / (image + '.jpg'))
        os.rename(src_image, dst_image)

    for image in val_images:
        src_image = str(SOURCE_FOLDER / 'images' / (image + '.jpg'))
        dst_image = str(VAL_FOLDER / 'images' / (image + '.jpg'))
        os.rename(src_image, dst_image)


if __name__ == '__main__':
    main()
