"""
Script to create a submission for kaggle
"""

import argparse
import pathlib

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

import model.model as models
import data_loader.data_loaders as dataloaders
from lib.utils import all_tensors_to


def main():
    """
    Main function
    """
    # pylint: disable=too-many-locals, too-many-statements
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = argparse.ArgumentParser(description='Convert video to slowmo')
    args.add_argument('-p', '--path', type=str, help='path to test folder')
    args.add_argument('--checkpoint', type=str, help='path to model checkpoint')
    args.add_argument('--use-gpu', action='store_true', help='Use GPU for processing')
    args.add_argument('--batch-size', default=2, type=int, help='how many pairs of frames to load at once')
    args.add_argument('--preload-to-mem', action='store_true', help='Preload pictures to memory')
    args.add_argument('--tta', default=3, type=int, help="Number of test time augmentations")
    args.add_argument('--context-size', default=5, type=int, help="Number of context images")
    args.add_argument('--size', default=None, nargs=2, metavar=('height', 'width'), help="Image size")
    args.add_argument('--softmax', action='store_true')
    args.add_argument('--ignore-loc', action='store_true')

    args = args.parse_args()

    if args.size is None:
        args.size = [384, 384]

    test_folder = pathlib.Path(args.path)
    checkpoint_path = pathlib.Path(args.checkpoint)

    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    state_dict = checkpoint['state_dict']
    model_name = checkpoint['arch']
    model_params = checkpoint['config']['arch']['args']

    model = getattr(models, model_name)(**model_params)

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    add_context = bool(args.context_size)
    # Transforms
    dataloader = dataloaders.MelanomaDataloader(test_folder,
                                                args.batch_size,
                                                shuffle=False,
                                                num_workers=4,
                                                malignant_sampling_weight=None,
                                                preload_to_mem=args.preload_to_mem,
                                                augmentation=True,
                                                hue_augmentation=0.05,
                                                tta_augmentations=args.tta,
                                                resize_type="resize",
                                                target_size=(args.size[0], args.size[1]),
                                                add_context=add_context,
                                                max_context_imgs=args.context_size,
                                                ignore_location=args.ignore_loc)

    output_path = test_folder / 'submission.csv'
    print("Writing result to", str(output_path))

    res = {"image_name": [], "target": []}

    with torch.no_grad():
        for data in tqdm(dataloader):
            data = all_tensors_to(data, device=device, non_blocking=True)
            out = model(data)
            if not args.softmax:
                output = torch.sigmoid(out)
            else:
                output = torch.nn.functional.softmax(out, dim=1)[:, 1]

            res["image_name"].extend([name.split('.')[0] for name in data["image_name"]])
            res["target"].extend(output.cpu().tolist())

    res = pd.DataFrame(res)
    res.to_csv(str(output_path), index=False)


if __name__ == '__main__':
    main()
