import os
import glob
import argparse
from tqdm import tqdm

import torch
from torch.utils import data
import clip

import webdataset as wds

from react_retrieval.data.tsv import TSVWriter
from react_retrieval.data.datasets import *


def get_argument_parser():
    parser = argparse.ArgumentParser(description='Extract features from Vision-Language datasets.')
    parser.add_argument('--model', default='ViT-B/32', help='VL model name.', type=str)
    parser.add_argument('--dataset_dir', required=True, help='Dataset directory.', type=str)
    parser.add_argument('--save_dir', required=True, help='Output feature directory.', type=str)
    parser.add_argument('--tsv_chunk_idx', default=0, help='TSV file index.', type=int)
    parser.add_argument('--tsv_chunks', default=8, help='TSV file index.', type=int)
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


def main(args, model, preprocess):
    pairs_wds = wds.WebDataset(os.path.join(args.dataset_dir, f'{args.tsv_idx:0{args.tsv_padzero}d}.tar')) \
        .decode("pilrgb") \
        .rename(image="jpg;png;jpeg;webp", text="txt") \
        .map_dict(image=preprocess, text=lambda text: clip.tokenize([text], context_length=77, truncate=True)[0]) \
        .to_tuple("image", "text", "json")

    pairs_wds_loader = data.DataLoader(pairs_wds, batch_size=args.batch_size,
                                       num_workers=args.workers, drop_last=False, pin_memory=True,
                                       shuffle=False, collate_fn=collate_fn)

    feature_file = os.path.join(args.save_dir, f'{args.tsv_idx:0{args.tsv_padzero}d}.tsv')

    feature_writer = TSVWriter(feature_file)

    for images, texts, metas in pairs_wds_loader:
        images = images.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)

        with torch.no_grad():
            image_embeddings = model.encode_image(images)
            text_embeddings = model.encode_text(texts)

            for i in range(image_embeddings.shape[0]):
                feature_writer.write([
                    metas[i]['key'],
                    metas[i]['sha256'],
                    encode_as_string(image_embeddings[i]),
                    encode_as_string(text_embeddings[i]),
                ])

    feature_writer.close()


if __name__ == "__main__":
    args = get_argument_parser().parse_args()
    n_shards = len(glob.glob(os.path.join(args.dataset_dir, '*.tar')))
    n_shard_per_chunk = n_shards // args.tsv_chunks

    torch.cuda.set_device(args.tsv_chunk_idx)

    os.makedirs(args.save_dir, exist_ok=True)

    if args.model in clip.available_models():
        model, preprocess = clip.load(args.model)
    model.cuda().eval()

    pbar = tqdm(range(n_shard_per_chunk*args.tsv_chunk_idx, min(n_shard_per_chunk*(args.tsv_chunk_idx+1), n_shards)))
    for i in pbar:
        args.tsv_idx = i
        pbar.set_description(f'{args.tsv_idx:0{args.tsv_padzero}d}.tar')
        main(args, model, preprocess)
