import os
import glob
import argparse
from tqdm import tqdm

import torch

from tarfile import TarFile

from collections import OrderedDict

import pickle


def get_argument_parser():
    parser = argparse.ArgumentParser(description='Extract features from Vision-Language datasets.')
    parser.add_argument('--dataset_dir', required=True, help='Dataset directory.', type=str)
    parser.add_argument('--filelist', required=True, help='Output feature directory.', type=str)
    parser.add_argument('--tsv_padzero', default=5, help='TSV file name padding.', type=int)
    parser.add_argument('--batch_size', default=128, help='Batch size.', type=int)
    parser.add_argument('--workers', default=8, help='Number of workers.', type=int)
    parser.add_argument('--print_freq', default=20, help='Print frequency.', type=int)
    return parser


def collate_fn(batch):
    images, texts, metas = list(zip(*batch))
    images = torch.stack(images, dim=0)
    texts = torch.stack(texts, dim=0)
    return images, texts, metas


def main(args):
    tarfile_list = sorted(f for f in os.listdir(args.dataset_dir) if f.endswith('.tar'))

    filelist = OrderedDict()

    pbar = tqdm(enumerate(tarfile_list), total=len(tarfile_list))
    for tsv_idx, tarfile_name in pbar:
        assert tarfile_name == f'{tsv_idx:0{args.tsv_padzero}d}.tar'
        pbar.set_description(tarfile_name)

        tarf = TarFile(os.path.join(args.dataset_dir, f'{tsv_idx:0{args.tsv_padzero}d}.tar'))
        filelist[f'{tsv_idx:0{args.tsv_padzero}d}'] = []
        cur_list = filelist[f'{tsv_idx:0{args.tsv_padzero}d}']

        for f in tarf:
            if not f.name.endswith('.jpg'):
                continue
            cur_list.append(f.name)

    pickle.dump(filelist, open(args.filelist, "wb"))


if __name__ == "__main__":
    args = get_argument_parser().parse_args()

    main(args)
